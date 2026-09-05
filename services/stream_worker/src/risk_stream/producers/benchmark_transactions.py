from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4
import time

from risk_engine.contracts.transaction import TransactionEvent
from risk_stream.producers.transaction_producer import TransactionProducer


def build_transaction(i: int) -> TransactionEvent:
    return TransactionEvent(
        event_id=f"evt-bench-{uuid4().hex[:12]}",
        transaction_id=f"txn-bench-{uuid4().hex[:12]}",
        event_time=datetime.now(timezone.utc),
        customer_id=f"cust-bench-{i % 10:03d}",
        merchant_id=f"merchant-{i % 5:03d}",
        amount=Decimal(str(500 + (i % 20) * 100)),
        currency="INR",
        device_id=f"device-bench-{i % 10:03d}",
        ip_address=f"10.20.30.{i % 10 + 1}",
        payment_method_id=f"payment-bench-{i % 20:03d}",
        merchant_category="electronics",
        country="IN",
        channel="web",
    )


def main() -> None:
    producer = TransactionProducer()
    count = 200

    started = time.perf_counter()

    for i in range(count):
        event = build_transaction(i)
        producer.publish(event)

    producer.flush()

    elapsed = time.perf_counter() - started

    print(f"Published {count} valid transactions.")
    print(f"Producer elapsed time: {elapsed:.3f}s")
    print(f"Producer rate: {count / elapsed:.2f} events/s")


if __name__ == "__main__":
    main()
