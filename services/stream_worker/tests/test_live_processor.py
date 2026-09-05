import asyncio
from datetime import datetime, timezone
from decimal import Decimal

from risk_engine.contracts import TransactionEvent
from risk_stream.processors.transaction_processor import TransactionProcessor


async def main() -> None:
    event = TransactionEvent(
        event_id="evt-retry-002",
        transaction_id="txn-retry-002",
        event_time=datetime.now(timezone.utc),
        customer_id="cust-live-001",
        merchant_id="merchant-live-001",
        amount=Decimal("1499.00"),
        currency="INR",
        device_id="device-live-001",
        ip_address="10.20.30.40",
        payment_method_id="payment-live-001",
        merchant_category="electronics",
        country="IN",
        channel="web",
    )

    processor = TransactionProcessor()

    try:
        await processor.process(event)
        print()
        print("=" * 60)
        print("LIVE PROCESSOR TEST PASSED")
        print("=" * 60)
    finally:
        await processor.close()


if __name__ == "__main__":
    asyncio.run(main())