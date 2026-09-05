import asyncio

from sqlalchemy import select

from risk_persistence.postgres.models.outbox_event import OutboxEvent
from risk_persistence.postgres.session import AsyncSessionLocal


async def main():
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(OutboxEvent).where(
                OutboxEvent.aggregate_id
                == "r6-failure-recovery-test-b4610c6ba550"
            )
        )

        for event in result.scalars().all():
            print(f"ID         = {event.id}")
            print(f"STATUS     = {event.status}")
            print(f"ATTEMPTS   = {event.attempts}")
            print(f"CREATED    = {event.created_at}")
            print(f"NEXT       = {event.next_attempt_at}")
            print(f"PUBLISHED  = {event.published_at}")
            print(f"LAST_ERROR = {event.last_error}")
            print(f"TOPIC      = {event.topic}")


if __name__ == "__main__":
    asyncio.run(main())
