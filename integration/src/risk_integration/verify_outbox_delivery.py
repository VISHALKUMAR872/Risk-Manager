import json
import uuid

from confluent_kafka import Consumer, KafkaException


BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC = "risk.decisions"

TRANSACTION_ID = "e2e-v5-eb040ea7b18f"

def main() -> None:
    consumer = Consumer(
        {
            "bootstrap.servers": BOOTSTRAP_SERVERS,
            "group.id": (
                "risk-sentinel-outbox-verification-"
                + str(uuid.uuid4())
            ),
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )

    consumer.subscribe([TOPIC])

    print("=" * 88)
    print("RISK SENTINEL — R6 OUTBOX DELIVERY VERIFICATION")
    print("=" * 88)
    print(f"Transaction: {TRANSACTION_ID}")

    found = False

    try:
        for _ in range(30):
            message = consumer.poll(1.0)

            if message is None:
                continue

            if message.error():
                raise KafkaException(message.error())

            payload = json.loads(
                message.value().decode("utf-8")
            )

            if payload.get("transaction_id") != TRANSACTION_ID:
                continue

            found = True

            print(
                f"[PASS] Kafka risk decision found: "
                f"{TRANSACTION_ID}"
            )

            print(
                f"[PASS] Topic: {message.topic()}"
            )

            print(
                f"[PASS] Partition: {message.partition()}"
            )

            print(
                f"[PASS] Offset: {message.offset()}"
            )

            print(
                f"[PASS] Decision: "
                f"{payload.get('decision')}"
            )

            print(
                f"[PASS] Risk level: "
                f"{payload.get('risk_level')}"
            )

            print(
                f"[PASS] Fraud probability: "
                f"{payload.get('fraud_probability')}"
            )

            print(
                f"[PASS] Expected loss: "
                f"{payload.get('expected_loss')}"
            )

            break

    finally:
        consumer.close()

    if not found:
        raise AssertionError(
            "Risk decision was not found in risk.decisions"
        )

    print("\n" + "=" * 88)
    print("R6 OUTBOX DELIVERY VERIFICATION: PASS")
    print("=" * 88)


if __name__ == "__main__":
    main()