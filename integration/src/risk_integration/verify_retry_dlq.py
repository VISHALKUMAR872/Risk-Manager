import asyncio
import json
import sys
import time
from uuid import uuid4

from confluent_kafka import Consumer, KafkaException, Producer

from risk_persistence.postgres.session import AsyncSessionLocal
from sqlalchemy import text


DLQ_TOPIC = "transactions.raw.dlq"
SOURCE_TOPIC = "transactions.raw"


def create_poison_payload() -> tuple[str, str]:
    suffix = uuid4().hex[:12]

    event_id = f"r7-poison-event-{suffix}"
    transaction_id = f"r7-poison-test-{suffix}"

    payload = {
        "event_id": event_id,
        "transaction_id": transaction_id,
        "event_time": "2026-09-04T00:00:00Z",
        "customer_id": f"r7-customer-{suffix}",
        "merchant_id": f"r7-merchant-{suffix}",
        "amount": "NOT_A_NUMBER",
        "currency": "INR",
        "device_id": f"r7-device-{suffix}",
        "ip_address": "10.250.250.77",
        "payment_method_id": f"r7-payment-{suffix}",
        "merchant_category": "retail",
        "country": "IN",
        "channel": "web",
    }

    return transaction_id, json.dumps(payload)


def publish_poison_message(
    transaction_id: str,
    raw_payload: str,
) -> None:
    producer = Producer(
        {
            "bootstrap.servers": "localhost:9092",
            "client.id": "risk-sentinel-r7-poison-test",
            "acks": "all",
            "enable.idempotence": True,
        }
    )

    delivery_error = None

    def callback(err, message):
        nonlocal delivery_error
        delivery_error = err

    producer.produce(
        topic=SOURCE_TOPIC,
        key=transaction_id,
        value=raw_payload,
        callback=callback,
    )

    remaining = producer.flush(timeout=10)

    if delivery_error is not None:
        raise RuntimeError(
            f"Poison message delivery failed: {delivery_error}"
        )

    if remaining > 0:
        raise RuntimeError(
            f"Poison message delivery timed out: {remaining}"
        )

    print(
        f"[PASS] Poison message published: {transaction_id}"
    )


def verify_dlq(
    transaction_id: str,
) -> None:
    consumer = Consumer(
        {
            "bootstrap.servers": "localhost:9092",
            "group.id": (
                f"risk-sentinel-r7-dlq-verification-{uuid4().hex}"
            ),
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )

    consumer.subscribe([DLQ_TOPIC])

    found = False

    try:
        deadline = time.time() + 30

        while time.time() < deadline:
            message = consumer.poll(1.0)

            if message is None:
                continue

            if message.error():
                raise KafkaException(message.error())

            payload = json.loads(
                message.value().decode("utf-8")
            )

            original_payload = json.loads(
                payload["original_payload"]
            )

            if (
                original_payload.get("transaction_id")
                != transaction_id
            ):
                continue

            found = True

            print()
            print("DLQ verification:")

            assert payload["original_topic"] == SOURCE_TOPIC
            print(
                f"[PASS] Original topic: "
                f"{payload['original_topic']}"
            )

            assert isinstance(
                payload["original_partition"],
                int,
            )
            print(
                f"[PASS] Original partition: "
                f"{payload['original_partition']}"
            )

            assert isinstance(
                payload["original_offset"],
                int,
            )
            print(
                f"[PASS] Original offset: "
                f"{payload['original_offset']}"
            )

            assert payload["attempt_count"] == 1
            print(
                f"[PASS] Attempt count: "
                f"{payload['attempt_count']}"
            )

            assert payload["error_type"] == "ValidationError"
            print(
                f"[PASS] Error type: "
                f"{payload['error_type']}"
            )

            assert "amount" in payload["error_message"]
            print("[PASS] Validation error preserved")

            assert (
                original_payload["transaction_id"]
                == transaction_id
            )
            print(
                f"[PASS] Original transaction ID: "
                f"{transaction_id}"
            )

            assert (
                original_payload["amount"]
                == "NOT_A_NUMBER"
            )
            print(
                "[PASS] Original malformed payload preserved"
            )

            break

    finally:
        consumer.close()

    if not found:
        raise AssertionError(
            "Poison message was not found in DLQ"
        )


async def verify_postgres(transaction_id: str) -> None:
    print()
    print("PostgreSQL verification:")

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text(
                """
                SELECT COUNT(*) 
                FROM transactions
                WHERE transaction_id = :transaction_id
                """
            ),
            {"transaction_id": transaction_id},
        )

        count = int(result.scalar_one())

        assert count == 0
        print(
            "[PASS] Poison message created no transaction record"
        )


async def main() -> None:
    print("=" * 88)
    print("RISK SENTINEL — R7 POISON MESSAGE / DLQ TEST")
    print("=" * 88)

    transaction_id, raw_payload = create_poison_payload()

    print(f"Transaction: {transaction_id}")

    publish_poison_message(
        transaction_id,
        raw_payload,
    )

    verify_dlq(transaction_id)

    await verify_postgres(transaction_id)

    print()
    print("=" * 88)
    print("R7 POISON MESSAGE / DLQ VERIFICATION: PASS")
    print("=" * 88)
    print("DLQ         : poison message preserved")
    print("Validation  : ValidationError preserved")
    print("Source      : transactions.raw")
    print("Attempts    : 1")
    print("PostgreSQL  : no transaction created")


if __name__ == "__main__":
    asyncio.run(main())
