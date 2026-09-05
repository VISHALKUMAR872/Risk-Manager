from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4

from risk_engine.contracts.transaction import TransactionEvent
from risk_stream.producers.transaction_producer import TransactionProducer


def main() -> None:
    transaction_id = f"e2e-v5-{uuid4().hex[:12]}"
    event_id = f"e2e-event-{uuid4().hex[:12]}"

    event = TransactionEvent(
        event_id=event_id,
        transaction_id=transaction_id,
        event_time=datetime.now(timezone.utc),
        customer_id=f"e2e-customer-{uuid4().hex[:8]}",
        merchant_id=f"e2e-merchant-{uuid4().hex[:8]}",
        amount=Decimal("1499.00"),
        currency="INR",
        device_id=f"e2e-device-{uuid4().hex[:8]}",
        ip_address="10.250.250.10",
        payment_method_id=f"e2e-payment-{uuid4().hex[:8]}",
        merchant_category="electronics",
        country="IN",
        channel="web",
    )

    producer = TransactionProducer()

    print("=" * 80)
    print("V5 LIVE END-TO-END TEST TRANSACTION")
    print("=" * 80)
    print(event.model_dump_json(indent=2))

    producer.publish(event)
    producer.flush()

    print()
    print("Published successfully.")
    print(f"EVENT_ID:       {event.event_id}")
    print(f"TRANSACTION_ID: {event.transaction_id}")
    print(f"CUSTOMER_ID:    {event.customer_id}")
    print(f"DEVICE_ID:      {event.device_id}")
    print(f"IP_ADDRESS:     {event.ip_address}")


if __name__ == "__main__":
    main()
