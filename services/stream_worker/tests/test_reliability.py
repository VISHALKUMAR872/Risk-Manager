import asyncio
from datetime import datetime, timezone
from decimal import Decimal

from risk_engine.contracts import TransactionEvent
from risk_stream.processors.transaction_processor import TransactionProcessor


def make_event(transaction_id: str) -> TransactionEvent:
    return TransactionEvent(
        event_id=f"evt-{transaction_id}",
        transaction_id=transaction_id,
        event_time=datetime.now(timezone.utc),
        customer_id="cust-reliability-001",
        merchant_id="merchant-reliability-001",
        amount=Decimal("1499.00"),
        currency="INR",
        device_id="device-reliability-001",
        ip_address="10.20.30.40",
        payment_method_id="payment-reliability-001",
        merchant_category="electronics",
        country="IN",
        channel="web",
    )


async def main() -> None:
    processor = TransactionProcessor()

    try:
        transaction_id = "txn-reliability-001"
        event = make_event(transaction_id)

        print("=" * 70)
        print("TEST 1: FIRST PROCESSING")
        print("=" * 70)

        await processor.process(event)

        print()
        print("=" * 70)
        print("TEST 2: DUPLICATE DELIVERY")
        print("=" * 70)

        await processor.process(event)

        print()
        print("=" * 70)
        print("RELIABILITY TEST COMPLETE")
        print("=" * 70)

    finally:
        await processor.close()


if __name__ == "__main__":
    asyncio.run(main())