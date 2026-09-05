import asyncio

from risk_persistence.postgres.repositories import OutboxRepository
from risk_persistence.postgres.session import AsyncSessionLocal


async def main():
    async with AsyncSessionLocal() as session:
        repository = OutboxRepository(session)
        events = await repository.get_pending(limit=100)

        print(f"Pending eligible events: {len(events)}")

        for event in events:
            print(
                event.id,
                event.aggregate_id,
                event.status,
                event.attempts,
                event.next_attempt_at,
            )


if __name__ == "__main__":
    asyncio.run(main())
