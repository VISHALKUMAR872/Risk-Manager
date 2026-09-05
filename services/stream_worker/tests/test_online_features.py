import asyncio

from risk_engine.contracts import TransactionEvent
from risk_features import OnlineFeatureService
from risk_persistence.neo4j import Neo4jClient
from risk_persistence.redis import RedisClient

from datetime import datetime, timezone
from decimal import Decimal


async def main() -> None:
    event = TransactionEvent(
        event_id="evt-feature-test",
        transaction_id="txn-feature-test",
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

    redis = RedisClient()
    neo4j = Neo4jClient()

    service = OnlineFeatureService(
        redis=redis,
        neo4j=neo4j,
    )

    features = await service.build(event)

    print(features.model_dump_json(indent=2))

    await redis.close()
    await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())