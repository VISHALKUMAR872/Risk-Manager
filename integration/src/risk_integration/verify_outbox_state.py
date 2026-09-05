import asyncio
import sys

from sqlalchemy import select

from risk_persistence.postgres.models.outbox_event import OutboxEvent
from risk_persistence.postgres.session import AsyncSessionLocal


if len(sys.argv) != 2:
    raise SystemExit(
        "Usage: python verify_outbox_state.py <transaction_id>"
    )

TRANSACTION_ID = sys.argv[1]


async def main() -> None:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(OutboxEvent)
            .where(
                OutboxEvent.aggregate_id == TRANSACTION_ID
            )
            .order_by(
                OutboxEvent.created_at.desc()
            )
        )

        events = list(result.scalars().all())

        if not events:
            raise AssertionError(
                f"No outbox event found for {TRANSACTION_ID}"
            )

        event = events[0]

        print("=" * 88)
        print("RISK SENTINEL — R6 OUTBOX STATE VERIFICATION")
        print("=" * 88)

        print(f"Transaction: {TRANSACTION_ID}")
        print(f"Outbox ID:   {event.id}")
        print(f"Status:      {event.status}")
        print(f"Attempts:    {event.attempts}")
        print(f"Published:   {event.published_at}")
        print(f"Last error:  {event.last_error}")

        assert event.status == "PUBLISHED", (
            f"Expected PUBLISHED, got {event.status}"
        )
        assert event.published_at is not None, (
            "published_at is NULL"
        )
        assert event.last_error is None, (
            f"last_error is not NULL: {event.last_error}"
        )

        print("\n[PASS] Outbox status = PUBLISHED")
        print("[PASS] published_at is populated")
        print("[PASS] last_error is NULL")

        print("\n" + "=" * 88)
        print("R6 OUTBOX STATE VERIFICATION: PASS")
        print("=" * 88)


if __name__ == "__main__":
    asyncio.run(main())
