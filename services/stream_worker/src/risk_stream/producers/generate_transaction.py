from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4

from risk_engine.contracts.transaction import TransactionEvent

from risk_stream.producers.transaction_producer import TransactionProducer


def build_transaction() -> TransactionEvent:
    transaction_id = f"txn-{uuid4().hex[:12]}"
    event_id = f"evt-{uuid4().hex[:12]}"

    return TransactionEvent(
        event_id=event_id,
        transaction_id=transaction_id,
        event_time=datetime.now(timezone.utc),
        customer_id="cust-001",
        merchant_id="merchant-001",
        amount=Decimal("1499.00"),
        currency="INR",
        device_id="device-001",
        ip_address="192.168.1.10",
        payment_method_id="payment-001",
        merchant_category="electronics",
        country="IN",
        channel="web",
    )


def main() -> None:
    producer = TransactionProducer()

    event = build_transaction()

    print("Publishing:")
    print(event.model_dump_json(indent=2))

    producer.publish(event)
    producer.flush()


if __name__ == "__main__":
    main()