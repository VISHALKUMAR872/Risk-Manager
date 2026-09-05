from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func, select
from risk_persistence.postgres.models.transaction import Transaction
from risk_api.dependencies.database import get_db_session
from risk_persistence.postgres.models.risk_decision import RiskDecision

from risk_persistence.neo4j import Neo4jClient
from risk_persistence.neo4j.transaction_graph import TransactionGraph
from risk_persistence.postgres.repositories import (
    RiskDecisionRepository,
    TransactionRepository,
)
from risk_persistence.postgres.session import AsyncSessionLocal

from risk_api.api.schemas import (
    DashboardSummaryResponse,
    DashboardTransactionResponse,
    RiskDecisionResponse,
    TransactionNetworkResponse,
    TransactionResponse,
)


router = APIRouter(
    prefix="/transactions",
    tags=["transactions"],
)


async def get_session():
    async with AsyncSessionLocal() as session:
        yield session


@router.get(
    "",
    response_model=list[TransactionResponse],
)
async def list_transactions(
    limit: int = Query(default=50, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
) -> list[TransactionResponse]:
    repository = TransactionRepository(session)

    transactions = await repository.list_all(limit=limit)

    return transactions


@router.get(
    "/dashboard",
    response_model=list[DashboardTransactionResponse],
)
async def dashboard_transactions(
    limit: int = Query(default=50, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
) -> list[DashboardTransactionResponse]:
    repository = TransactionRepository(session)

    rows = await repository.list_with_risk(limit=limit)

    response: list[DashboardTransactionResponse] = []

    for transaction, risk in rows:
        response.append(
            DashboardTransactionResponse(
                transaction_id=transaction.transaction_id,
                event_id=transaction.event_id,
                event_time=transaction.event_time,
                customer_id=transaction.customer_id,
                merchant_id=transaction.merchant_id,
                amount=transaction.amount,
                currency=transaction.currency,
                device_id=transaction.device_id,
                ip_address=transaction.ip_address,
                payment_method=transaction.payment_method_id,
                merchant_category=transaction.merchant_category,
                country=transaction.country,
                channel=transaction.channel,
                status=transaction.status,
                created_at=transaction.created_at,
                fraud_probability=(
                    float(risk.fraud_probability)
                    if risk is not None
                    else None
                ),
                expected_loss=(
                    float(risk.expected_loss)
                    if risk is not None
                    else None
                ),
                risk_level=risk.risk_level if risk else None,
                decision=risk.decision if risk else None,
                reason_codes=(
                    risk.reason_codes
                    if risk and risk.reason_codes
                    else []
                ),
                policy_version=(
                    risk.policy_version
                    if risk
                    else None
                ),
                model_version=(
                    risk.model_version
                    if risk
                    else None
                ),
                calibration_version=(
                    risk.calibration_version
                    if risk
                    else None
                ),
            )
        )

    return response


@router.get(
    "/{transaction_id}/network",
    response_model=TransactionNetworkResponse,
)
async def get_transaction_network(
    transaction_id: str,
) -> TransactionNetworkResponse:
    client = Neo4jClient()

    try:
        graph = TransactionGraph(client)

        network = await graph.get_transaction_network(
            transaction_id
        )

        if network is None:
            raise HTTPException(
                status_code=404,
                detail="Transaction network not found",
            )

        return TransactionNetworkResponse(**network)

    finally:
        await client.close()


@router.get(
    "/{transaction_id}",
    response_model=TransactionResponse,
)
async def get_transaction(
    transaction_id: str,
    session: AsyncSession = Depends(get_session),
) -> TransactionResponse:
    repository = TransactionRepository(session)

    transaction = await repository.get_by_transaction_id(
        transaction_id
    )

    if transaction is None:
        raise HTTPException(
            status_code=404,
            detail="Transaction not found",
        )

    return TransactionResponse(
        transaction_id=transaction.transaction_id,
        event_id=transaction.event_id,
        event_time=transaction.event_time,
        customer_id=transaction.customer_id,
        merchant_id=transaction.merchant_id,
        amount=transaction.amount,
        currency=transaction.currency,
        device_id=transaction.device_id,
        ip_address=transaction.ip_address,
        payment_method=transaction.payment_method_id,
        merchant_category=transaction.merchant_category,
        country=transaction.country,
        channel=transaction.channel,
        status=transaction.status,
        created_at=transaction.created_at,
    )


@router.get(
    "/{transaction_id}/risk",
    response_model=RiskDecisionResponse,
)
async def get_transaction_risk(
    transaction_id: str,
    session: AsyncSession = Depends(get_session),
) -> RiskDecisionResponse:
    repository = RiskDecisionRepository(session)

    decision = await repository.get_by_transaction_id(
        transaction_id
    )

    if decision is None:
        raise HTTPException(
            status_code=404,
            detail="Risk decision not found",
        )

    return decision

@router.get(
    "/dashboard/summary",
    response_model=DashboardSummaryResponse,
)
async def get_dashboard_summary(
    session: AsyncSession = Depends(get_db_session),
) -> DashboardSummaryResponse:
    transaction_count = await session.scalar(
        select(func.count(Transaction.id))
    )

    risk_decided_count = await session.scalar(
        select(func.count(RiskDecision.id))
    )

    pending_count = await session.scalar(
        select(func.count(Transaction.id))
        .select_from(Transaction)
        .outerjoin(
            RiskDecision,
            RiskDecision.transaction_id == Transaction.transaction_id,
        )
        .where(
            Transaction.status.in_(["RECEIVED", "PROCESSING"]),
            RiskDecision.id.is_(None),
        )
    )

    failed_count = await session.scalar(
        select(func.count(Transaction.id)).where(
            Transaction.status == "FAILED"
        )
    )

    average_amount = await session.scalar(
        select(func.avg(Transaction.amount))
    )

    expected_loss = await session.scalar(
        select(
            func.coalesce(
                func.sum(RiskDecision.expected_loss),
                0,
            )
        )
    )

    intervention_count = await session.scalar(
        select(func.count(RiskDecision.id)).where(
            RiskDecision.decision.in_(
                ["VERIFY", "REVIEW", "HOLD"]
            )
        )
    )

    high_risk_count = await session.scalar(
        select(func.count(RiskDecision.id)).where(
            RiskDecision.risk_level.in_(
                ["HIGH", "CRITICAL"]
            )
        )
    )

    decision_rows = (
        await session.execute(
            select(
                RiskDecision.decision,
                func.count(RiskDecision.id),
            ).group_by(RiskDecision.decision)
        )
    ).all()

    risk_rows = (
        await session.execute(
            select(
                RiskDecision.risk_level,
                func.count(RiskDecision.id),
            ).group_by(RiskDecision.risk_level)
        )
    ).all()

    decisions = {
        "APPROVE": 0,
        "VERIFY": 0,
        "REVIEW": 0,
        "HOLD": 0,
    }

    for decision, count in decision_rows:
        if decision in decisions:
            decisions[decision] = count

    risk_levels = {
        "LOW": 0,
        "MEDIUM": 0,
        "HIGH": 0,
        "CRITICAL": 0,
    }

    for risk_level, count in risk_rows:
        if risk_level in risk_levels:
            risk_levels[risk_level] = count

    return DashboardSummaryResponse(
        transaction_count=transaction_count or 0,
        risk_decided_count=risk_decided_count or 0,
        pending_count=pending_count or 0,
        failed_count=failed_count or 0,
        intervention_count=intervention_count or 0,
        expected_loss=float(expected_loss or 0),
        high_risk_count=high_risk_count or 0,
        decisions=decisions,
        risk_levels=risk_levels,
        average_transaction_amount=float(
            average_amount or 0
        ),
    )