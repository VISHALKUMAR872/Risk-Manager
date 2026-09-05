from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4
import time

from risk_engine.contracts.transaction import TransactionEvent
from risk_stream.producers.transaction_producer import TransactionProducer


def build_transaction(i: int) -> TransactionEvent:
    return TransactionEvent(
        event_id=f"evt-throughput-{uuid4().hex[:12]}",
        transaction_id=f"txn-throughput-{uuid4().hex[:12]}",
        event_time=datetime.now(timezone.utc),
        customer_id=f"cust-throughput-{i % 20:03d}",
        merchant_id=f"merchant-{i % 10:03d}",
        amount=Decimal(str(500 + (i % 20) * 100)),
        currency="INR",
        device_id=f"device-throughput-{i % 20:03d}",
        ip_address=f"10.30.40.{i % 20 + 1}",
        payment_method_id=f"payment-throughput-{i % 30:03d}",
        merchant_category="electronics",
        country="IN",
        channel="web",
    )


def main() -> None:
    producer = TransactionProducer()
    count = 100

    start = datetime.now(timezone.utc)
    perf_start = time.perf_counter()

    for i in range(count):
        producer.publish(build_transaction(i))

    producer.flush()

    perf_elapsed = time.perf_counter() - perf_start
    end = datetime.now(timezone.utc)

    print(f"THROUGHPUT_TEST_START={start.isoformat()}")
    print(f"THROUGHPUT_TEST_END={end.isoformat()}")
    print(f"PUBLISHED={count}")
    print(f"PRODUCER_ELAPSED_SECONDS={perf_elapsed:.6f}")
    print(f"PRODUCER_RATE={count / perf_elapsed:.2f}")


if __name__ == "__main__":
    main()
