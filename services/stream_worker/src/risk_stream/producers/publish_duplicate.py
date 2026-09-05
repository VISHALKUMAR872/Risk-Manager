from datetime import datetime, timezone
from decimal import Decimal

from risk_engine.contracts.transaction import TransactionEvent
from risk_stream.producers.transaction_producer import TransactionProducer


event = TransactionEvent(
    event_id="evt-duplicate-fd68d3ca697a",
    transaction_id="txn-fd68d3ca697a",
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

producer = TransactionProducer()

print("Publishing duplicate:")
print(event.model_dump_json(indent=2))

producer.publish(event)
producer.flush()

print("Duplicate event published.")
