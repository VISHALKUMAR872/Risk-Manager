from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from risk_persistence.postgres.models.risk_decision import RiskDecision
from risk_persistence.postgres.models.transaction import Transaction


class TransactionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, transaction: Transaction) -> Transaction:
        self.session.add(transaction)
        await self.session.flush()
        return transaction

    async def get_by_transaction_id(
        self,
        transaction_id: str,
    ) -> Transaction | None:
        result = await self.session.execute(
            select(Transaction).where(
                Transaction.transaction_id == transaction_id
            )
        )
        return result.scalar_one_or_none()

    async def get_by_id(
        self,
        transaction_id: UUID,
    ) -> Transaction | None:
        result = await self.session.execute(
            select(Transaction).where(
                Transaction.id == transaction_id
            )
        )
        return result.scalar_one_or_none()

    async def update_status(
        self,
        transaction_id: str,
        status: str,
    ) -> Transaction | None:
        transaction = await self.get_by_transaction_id(transaction_id)

        if transaction is None:
            return None

        transaction.status = status
        await self.session.flush()

        return transaction

    async def list_all(
        self,
        limit: int = 50,
    ) -> list[Transaction]:
        result = await self.session.execute(
            select(Transaction)
            .order_by(
                Transaction.event_time.desc(),
                Transaction.created_at.desc(),
            )
            .limit(limit)
        )

        return list(result.scalars().all())

    async def list_with_risk(
        self,
        limit: int = 50,
    ) -> list[tuple[Transaction, RiskDecision | None]]:
        result = await self.session.execute(
            select(Transaction, RiskDecision)
            .outerjoin(
                RiskDecision,
                RiskDecision.transaction_id
                == Transaction.transaction_id,
            )
            .order_by(
                Transaction.event_time.desc(),
                Transaction.created_at.desc(),
            )
            .limit(limit)
        )

        return list(result.all())
