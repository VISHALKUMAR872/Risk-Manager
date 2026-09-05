import asyncio
from datetime import datetime, timezone
from uuid import uuid4

from risk_persistence.postgres.models.outbox_event import OutboxEvent
from risk_persistence.postgres.session import AsyncSessionLocal


TEST_SUFFIX = uuid4().hex[:12]

TRANSACTION_ID = f"r6-failure-recovery-test-{TEST_SUFFIX}"


async def main() -> None:
    async with AsyncSessionLocal() as session:
        event = OutboxEvent(
            event_type="RiskDecisionCreated",
            aggregate_type="Transaction",
            aggregate_id=TRANSACTION_ID,
            topic="risk.decisions",
            message_key=TRANSACTION_ID,
            payload={
                "event_id": f"r6-failure-recovery-event-{TEST_SUFFIX}",
                "transaction_id": TRANSACTION_ID,
                "decision": "VERIFY",
                "risk_level": "MEDIUM",
                "fraud_probability": 0.25,
                "expected_loss": 100.0,
                "created_at": datetime.now(
                    timezone.utc
                ).isoformat(),
            },
            status="PENDING",
            attempts=0,
            next_attempt_at=datetime.now(timezone.utc),
        )

        session.add(event)
        await session.commit()

        print("=" * 88)
        print("RISK SENTINEL — R6 OUTBOX DELIVERY TEST")
        print("=" * 88)
        print(f"Transaction ID: {TRANSACTION_ID}")
        print(f"Outbox ID:      {event.id}")
        print("Status:         PENDING")
        print("Attempts:       0")
        print()
        print("[PASS] Fresh PENDING outbox event created")
        print()
        print("Use this transaction ID for verification:")
        print(TRANSACTION_ID)


if __name__ == "__main__":
    asyncio.run(main())
