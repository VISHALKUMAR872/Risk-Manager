import asyncio
import json
import time
from decimal import Decimal

from confluent_kafka import Producer
from sqlalchemy import text

from risk_persistence.neo4j import Neo4jClient
from risk_persistence.postgres.session import AsyncSessionLocal
from risk_persistence.redis import RedisClient


BOOTSTRAP_SERVERS = "localhost:9092"
TRANSACTION_TOPIC = "transactions.raw"

from uuid import uuid4

TEST_SUFFIX = uuid4().hex[:12]

TRANSACTION_ID = f"r5-duplicate-test-{TEST_SUFFIX}"
EVENT_ID = f"r5-event-duplicate-test-{TEST_SUFFIX}"

CUSTOMER_ID = f"r5-customer-{TEST_SUFFIX}"
MERCHANT_ID = f"r5-merchant-{TEST_SUFFIX}"
DEVICE_ID = f"r5-device-{TEST_SUFFIX}"
IP_ADDRESS = "10.250.250.99"
PAYMENT_METHOD_ID = f"r5-payment-{TEST_SUFFIX}"

EVENT_TIME = "2026-09-03T17:00:00Z"


EVENT = {
    "event_id": EVENT_ID,
    "transaction_id": TRANSACTION_ID,
    "event_time": EVENT_TIME,
    "customer_id": CUSTOMER_ID,
    "merchant_id": MERCHANT_ID,
    "amount": "1499.00",
    "currency": "INR",
    "device_id": DEVICE_ID,
    "ip_address": IP_ADDRESS,
    "payment_method_id": PAYMENT_METHOD_ID,
    "merchant_category": "electronics",
    "country": "IN",
    "channel": "web",
}


def publish_duplicate_events() -> None:
    producer = Producer(
        {
            "bootstrap.servers": BOOTSTRAP_SERVERS,
            "client.id": "risk-sentinel-r5-duplicate-test",
            "acks": "all",
            "enable.idempotence": True,
        }
    )

    payload = json.dumps(EVENT)

    print("\nPublishing exact same event twice:")

    for attempt in (1, 2):
        producer.produce(
            topic=TRANSACTION_TOPIC,
            key=TRANSACTION_ID,
            value=payload,
        )

        producer.flush()

        print(
            f"[PASS] Duplicate delivery #{attempt} published"
        )

    producer.flush()


async def wait_for_decided() -> None:
    deadline = time.monotonic() + 20

    while time.monotonic() < deadline:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                text(
                    """
                    SELECT status
                    FROM transactions
                    WHERE transaction_id = :transaction_id
                    """
                ),
                {"transaction_id": TRANSACTION_ID},
            )

            status = result.scalar_one_or_none()

            if status == "DECIDED":
                return

        await asyncio.sleep(0.5)

    raise AssertionError(
        "Transaction did not reach DECIDED within 20 seconds"
    )


async def verify_postgres() -> None:
    print("\nPostgreSQL verification:")

    async with AsyncSessionLocal() as session:

        result = await session.execute(
            text(
                """
                SELECT
                    COUNT(*) AS count
                FROM transactions
                WHERE transaction_id = :transaction_id
                """
            ),
            {"transaction_id": TRANSACTION_ID},
        )

        transaction_count = int(result.scalar_one())

        assert transaction_count == 1

        print(
            "[PASS] Transaction count: 1"
        )

        result = await session.execute(
            text(
                """
                SELECT
                    status,
                    event_id,
                    amount
                FROM transactions
                WHERE transaction_id = :transaction_id
                """
            ),
            {"transaction_id": TRANSACTION_ID},
        )

        transaction = result.mappings().one()

        assert transaction["status"] == "DECIDED"
        assert transaction["event_id"] == EVENT_ID
        assert Decimal(str(transaction["amount"])) == Decimal(
            "1499.00"
        )

        print(
            "[PASS] Transaction status: DECIDED"
        )
        print(
            f"[PASS] Event ID: {transaction['event_id']}"
        )
        print(
            f"[PASS] Amount: {transaction['amount']}"
        )

        result = await session.execute(
            text(
                """
                SELECT COUNT(*) AS count
                FROM risk_decisions
                WHERE transaction_id = :transaction_id
                """
            ),
            {"transaction_id": TRANSACTION_ID},
        )

        decision_count = int(result.scalar_one())

        assert decision_count == 1

        print(
            "[PASS] Risk decision count: 1"
        )

        result = await session.execute(
            text(
                """
                SELECT COUNT(*) AS count
                FROM outbox_events
                WHERE aggregate_id = :transaction_id
                """
            ),
            {"transaction_id": TRANSACTION_ID},
        )

        outbox_count = int(result.scalar_one())

        assert outbox_count == 1

        print(
            "[PASS] Outbox event count: 1"
        )


async def verify_redis() -> None:
    print("\nRedis verification:")

    redis = RedisClient()

    try:
        customer_key = (
            f"customer:{CUSTOMER_ID}:tx_events"
        )
        device_key = (
            f"device:{DEVICE_ID}:tx_events"
        )
        ip_key = (
            f"ip:{IP_ADDRESS}:tx_events"
        )

        customer_score = await redis.client.zscore(
            customer_key,
            TRANSACTION_ID,
        )

        device_score = await redis.client.zscore(
            device_key,
            TRANSACTION_ID,
        )

        ip_score = await redis.client.zscore(
            ip_key,
            TRANSACTION_ID,
        )

        assert customer_score is not None
        assert device_score is not None
        assert ip_score is not None

        print(
            "[PASS] Customer velocity contains transaction once"
        )
        print(
            "[PASS] Device velocity contains transaction once"
        )
        print(
            "[PASS] IP velocity contains transaction once"
        )

        marker = await redis.get(
            f"velocity:processed:{TRANSACTION_ID}"
        )

        assert marker == "1"

        print(
            "[PASS] Redis idempotency marker = 1"
        )

    finally:
        await redis.close()


async def verify_neo4j() -> None:
    print("\nNeo4j verification:")

    neo4j = Neo4jClient()

    try:
        result = await neo4j.execute(
            """
            MATCH (t:Transaction {id: $transaction_id})
            RETURN count(t) AS transaction_count
            """,
            {
                "transaction_id": TRANSACTION_ID,
            },
        )

        transaction_count = int(
            result[0]["transaction_count"]
        )

        assert transaction_count == 1

        print(
            "[PASS] Neo4j Transaction node count: 1"
        )

        result = await neo4j.execute(
            """
            MATCH (c:Customer {id: $customer_id})
                  -[:MADE]->
                  (t:Transaction {id: $transaction_id})
            RETURN count(*) AS count
            """,
            {
                "customer_id": CUSTOMER_ID,
                "transaction_id": TRANSACTION_ID,
            },
        )

        made_count = int(result[0]["count"])

        assert made_count == 1

        print(
            "[PASS] Customer → Transaction relationship count: 1"
        )

        result = await neo4j.execute(
            """
            MATCH (t:Transaction {id: $transaction_id})
                  -[:AT_MERCHANT]->
                  (m:Merchant {id: $merchant_id})
            RETURN count(*) AS count
            """,
            {
                "transaction_id": TRANSACTION_ID,
                "merchant_id": MERCHANT_ID,
            },
        )

        merchant_count = int(result[0]["count"])

        assert merchant_count == 1

        print(
            "[PASS] Transaction → Merchant relationship count: 1"
        )

        result = await neo4j.execute(
            """
            MATCH (c:Customer {id: $customer_id})
                  -[:USED_DEVICE]->
                  (d:Device {id: $device_id})
            RETURN count(*) AS count
            """,
            {
                "customer_id": CUSTOMER_ID,
                "device_id": DEVICE_ID,
            },
        )

        device_count = int(result[0]["count"])

        assert device_count == 1

        print(
            "[PASS] Customer → Device relationship count: 1"
        )

        result = await neo4j.execute(
            """
            MATCH (c:Customer {id: $customer_id})
                  -[:USED_IP]->
                  (ip:IP {address: $ip_address})
            RETURN count(*) AS count
            """,
            {
                "customer_id": CUSTOMER_ID,
                "ip_address": IP_ADDRESS,
            },
        )

        ip_count = int(result[0]["count"])

        assert ip_count == 1

        print(
            "[PASS] Customer → IP relationship count: 1"
        )

    finally:
        await neo4j.close()


async def main() -> None:
    print("=" * 88)
    print(
        "RISK SENTINEL — R5 DUPLICATE DELIVERY / IDEMPOTENCY TEST"
    )
    print("=" * 88)

    print(f"Transaction: {TRANSACTION_ID}")
    print(f"Event:       {EVENT_ID}")

    publish_duplicate_events()

    await wait_for_decided()

    await verify_postgres()
    await verify_redis()
    await verify_neo4j()

    print("\n" + "=" * 88)
    print(
        "R5 DUPLICATE DELIVERY / IDEMPOTENCY TEST: PASS"
    )
    print("=" * 88)

    print(
        "Duplicate Kafka deliveries : 2"
    )
    print(
        "PostgreSQL transaction     : 1"
    )
    print(
        "Risk decision              : 1"
    )
    print(
        "Outbox event               : 1"
    )
    print(
        "Redis velocity             : 1"
    )
    print(
        "Neo4j transaction          : 1"
    )


if __name__ == "__main__":
    asyncio.run(main())