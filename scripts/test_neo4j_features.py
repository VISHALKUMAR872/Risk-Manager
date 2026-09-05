import asyncio
from datetime import datetime

from risk_persistence.neo4j import Neo4jClient
from risk_features.online.neo4j_provider import Neo4jFeatureProvider


async def main() -> None:
    neo4j = Neo4jClient()

    try:
        await neo4j.verify_connectivity()

        provider = Neo4jFeatureProvider(neo4j)

        t1 = datetime.fromisoformat(
            "2026-08-23T04:53:48.574138+00:00"
        )

        t2 = datetime.fromisoformat(
            "2026-08-23T04:59:15.535611+00:00"
        )

        features_t1 = await provider.get_graph_features(
            customer_id="cust-001",
            device_id="device-001",
            ip_address="192.168.1.10",
            payment_method_id="payment-001",
            merchant_id="merchant-001",
            event_time=t1,
        )

        features_t2 = await provider.get_graph_features(
            customer_id="cust-001",
            device_id="device-001",
            ip_address="192.168.1.10",
            payment_method_id="payment-001",
            merchant_id="merchant-001",
            event_time=t2,
        )

        print("Features at T1:")
        print(features_t1)

        print()
        print("Features at T2:")
        print(features_t2)

    finally:
        await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())