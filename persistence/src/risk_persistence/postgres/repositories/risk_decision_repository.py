from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from risk_persistence.postgres.models.risk_decision import RiskDecision


class RiskDecisionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        decision: RiskDecision,
    ) -> RiskDecision:
        self.session.add(decision)
        await self.session.flush()
        return decision

    async def get_by_transaction_id(
        self,
        transaction_id: str,
    ) -> RiskDecision | None:
        result = await self.session.execute(
            select(RiskDecision).where(
                RiskDecision.transaction_id == transaction_id
            )
        )

        return result.scalar_one_or_none()
