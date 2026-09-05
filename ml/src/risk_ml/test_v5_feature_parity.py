from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal

from risk_engine.contracts import TransactionEvent

from risk_features.online import OnlineFeatureService
from risk_persistence.neo4j.client import Neo4jClient
from risk_persistence.neo4j.features import Neo4jFeatureProvider
from risk_persistence.redis.client import RedisClient
from risk_persistence.redis.velocity import VelocityStore


FEATURES = [
    "amount",
    "customer_transactions_1m",
    "customer_transactions_1h",
    "device_transactions_1h",
    "ip_transactions_1h",
    "customer_degree",
    "device_customer_count",
    "ip_customer_count",
]


def assert_equal_feature(name: str, offline, live) -> None:
    if name == "amount":
        assert Decimal(str(offline)) == Decimal(str(live)), (
            f"{name}: offline={offline!r}, live={live!r}"
        )
        return

    assert int(offline) == int(live), (
        f"{name}: offline={offline!r}, live={live!r}"
    )


async def build_live_service():
    redis_client = RedisClient()
    neo4j_client = Neo4jClient()

    velocity = VelocityStore(redis_client)
    graph = Neo4jFeatureProvider(neo4j_client)

    service = OnlineFeatureService(
        velocity_store=velocity,
        graph_provider=graph,
    )

    return service, redis_client, neo4j_client


async def test_v5_feature_parity():
    """
    Hard parity gate.

    The test deliberately scores transactions at their event_time and verifies
    that live Redis + Neo4j features obey the same point-in-time semantics
    expected by the V5 offline replay.
    """

    service, redis_client, neo4j_client = await build_live_service()

    try:
        assert await redis_client.ping()

        transactions = [
            TransactionEvent(
                event_id="parity-event-001",
                transaction_id="parity-txn-001",
                event_time=datetime(
                    2026, 9, 3, 10, 0, 0, tzinfo=timezone.utc
                ),
                customer_id="parity-customer-001",
                merchant_id="parity-merchant-001",
                amount=Decimal("100.00"),
                currency="INR",
                device_id="parity-device-001",
                ip_address="10.20.30.40",
                payment_method_id="parity-payment-001",
                merchant_category="electronics",
                country="IN",
                channel="web",
            ),
            TransactionEvent(
                event_id="parity-event-002",
                transaction_id="parity-txn-002",
                event_time=datetime(
                    2026, 9, 3, 10, 1, 0, tzinfo=timezone.utc
                ),
                customer_id="parity-customer-001",
                merchant_id="parity-merchant-001",
                amount=Decimal("200.00"),
                currency="INR",
                device_id="parity-device-001",
                ip_address="10.20.30.40",
                payment_method_id="parity-payment-001",
                merchant_category="electronics",
                country="IN",
                channel="web",
            ),
        ]

        # ------------------------------------------------------------
        # T1: nothing before the first transaction
        # ------------------------------------------------------------

        t1 = transactions[0]

        features_t1 = await service.build(t1)

        expected_t1 = {
            "amount": Decimal("100.00"),
            "customer_transactions_1m": 0,
            "customer_transactions_1h": 0,
            "device_transactions_1h": 0,
            "ip_transactions_1h": 0,
            "customer_degree": 0,
            "device_customer_count": 0,
            "ip_customer_count": 0,
        }

        print("\nT1 LIVE FEATURES")
        print(features_t1)

        for name in FEATURES:
            assert_equal_feature(
                name,
                expected_t1[name],
                getattr(features_t1, name),
            )

        # ------------------------------------------------------------
        # Update state after T1
        # ------------------------------------------------------------

        await redis_client.record_velocity_event(
            idempotency_key="velocity:processed:parity-txn-001",
            customer_key="customer:parity-customer-001:tx_events",
            device_key="device:parity-device-001:tx_events",
            ip_key="ip:10.20.30.40:tx_events",
            score=t1.event_time.timestamp(),
            member=t1.transaction_id,
            ttl_seconds=3600,
        )

        await neo4j_client.write_transaction(
            t1
        )

        # ------------------------------------------------------------
        # T2: T1 must now be visible
        # ------------------------------------------------------------

        t2 = transactions[1]

        features_t2 = await service.build(t2)

        expected_t2 = {
            "amount": Decimal("200.00"),
            "customer_transactions_1m": 1,
            "customer_transactions_1h": 1,
            "device_transactions_1h": 1,
            "ip_transactions_1h": 1,

            # MADE + USED_DEVICE + USED_IP + USED_PAYMENT
            "customer_degree": 4,

            "device_customer_count": 1,
            "ip_customer_count": 1,
        }

        print("\nT2 LIVE FEATURES")
        print(features_t2)

        for name in FEATURES:
            assert_equal_feature(
                name,
                expected_t2[name],
                getattr(features_t2, name),
            )

        print("\n[PASS] T1 point-in-time parity")
        print("[PASS] T2 historical-state parity")
        print("[PASS] Redis velocity parity")
        print("[PASS] Neo4j graph parity")

    finally:
        await redis_client.close()
        await neo4j_client.close()


if __name__ == "__main__":
    asyncio.run(test_v5_feature_parity())