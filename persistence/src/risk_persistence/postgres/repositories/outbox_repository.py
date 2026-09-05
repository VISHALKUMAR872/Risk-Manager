from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from risk_persistence.postgres.models.outbox_event import (
    OutboxEvent,
)


class OutboxRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        event: OutboxEvent,
    ) -> OutboxEvent:
        self.session.add(event)
        await self.session.flush()

        return event

    async def get_pending(
        self,
        limit: int = 100,
    ) -> list[OutboxEvent]:
        result = await self.session.execute(
            select(OutboxEvent)
            .where(
                OutboxEvent.status == "PENDING",
                OutboxEvent.next_attempt_at
                <= datetime.now(timezone.utc),
            )
            .order_by(
                OutboxEvent.created_at
            )
            .limit(limit)
            .with_for_update(
                skip_locked=True
            )
        )

        return list(
            result.scalars().all()
        )

    async def mark_published(
        self,
        event_id: UUID,
    ) -> None:
        event = await self.session.get(
            OutboxEvent,
            event_id,
        )

        if event is None:
            return

        event.status = "PUBLISHED"
        event.published_at = (
            datetime.now(timezone.utc)
        )
        event.last_error = None

        await self.session.flush()

    async def mark_failed(
        self,
        event_id: UUID,
        error: str,
        next_attempt_at: datetime,
    ) -> None:
        event = await self.session.get(
            OutboxEvent,
            event_id,
        )

        if event is None:
            return

        event.status = "PENDING"
        event.attempts += 1
        event.last_error = error
        event.next_attempt_at = next_attempt_at

        await self.session.flush()