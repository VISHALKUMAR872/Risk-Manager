from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from risk_engine.contracts import TransactionEvent
from risk_features.online import OnlineFeatureService
from risk_persistence.neo4j import Neo4jClient
from risk_persistence.neo4j.transaction_graph import TransactionGraph
from risk_persistence.redis import RedisClient, VelocityStore


def make_event(
    *,
    transaction_id: str,
    event_id: str,
    event_time: datetime,
    customer_id: str = "parity-customer-001",
    device_id: str = "parity-device-001",
    ip_address: str = "10.20.30.40",
    payment_method_id: str = "parity-payment-001",
    merchant_id: str = "parity-merchant-001",
    amount: str = "100.00",
) -> TransactionEvent:
    return TransactionEvent(
        event_id=event_id,
        transaction_id=transaction_id,
        event_time=event_time,
        customer_id=customer_id,
        merchant_id=merchant_id,
        amount=Decimal(amount),
        currency="INR",
        device_id=device_id,
        ip_address=ip_address,
        payment_method_id=payment_method_id,
        merchant_category="electronics",
        country="IN",
        channel="web",
    )


def assert_feature(
    name: str,
    actual,
    expected,
) -> None:
    if name == "amount":
        assert Decimal(str(actual)) == Decimal(str(expected)), (
            f"{name}: actual={actual}, expected={expected}"
        )
    else:
        assert int(actual) == int(expected), (
            f"{name}: actual={actual}, expected={expected}"
        )


async def cleanup(
    redis: RedisClient,
    neo4j: Neo4jClient,
) -> None:
    """
    Remove only the entities used by this test.
    """

    # Redis velocity state.
    await redis.client.delete(
        "customer:parity-customer-001:tx_events",
        "customer:parity-customer-002:tx_events",
        "device:parity-device-001:tx_events",
        "device:parity-device-002:tx_events",
        "ip:10.20.30.40:tx_events",
        "ip:10.20.30.41:tx_events",
    )

    # Redis idempotency markers.
    await redis.client.delete(
        "velocity:processed:parity-txn-001",
        "velocity:processed:parity-txn-002",
        "velocity:processed:parity-txn-003",
        "velocity:processed:parity-txn-004",
        "velocity:processed:parity-txn-005",
        "velocity:processed:parity-txn-006",
    )

    # Neo4j test transactions.
    await neo4j.execute(
        """
        MATCH (t:Transaction)
        WHERE t.id STARTS WITH 'parity-txn-'
        DETACH DELETE t
        """
    )

    # Neo4j test customers.
    await neo4j.execute(
        """
        MATCH (c:Customer)
        WHERE c.id STARTS WITH 'parity-customer-'
        DETACH DELETE c
        """
    )

    # Neo4j test devices.
    await neo4j.execute(
        """
        MATCH (d:Device)
        WHERE d.id STARTS WITH 'parity-device-'
        DETACH DELETE d
        """
    )

    # Neo4j test IPs.
    await neo4j.execute(
        """
        MATCH (ip:IP)
        WHERE ip.address STARTS WITH '10.20.30.'
        DETACH DELETE ip
        """
    )

    # Neo4j test payment methods.
    await neo4j.execute(
        """
        MATCH (p:PaymentMethod)
        WHERE p.id STARTS WITH 'parity-payment-'
        DETACH DELETE p
        """
    )

    # Neo4j test merchants.
    await neo4j.execute(
        """
        MATCH (m:Merchant)
        WHERE m.id STARTS WITH 'parity-merchant-'
        DETACH DELETE m
        """
    )


async def record_state(
    velocity: VelocityStore,
    graph: TransactionGraph,
    event: TransactionEvent,
) -> None:
    """
    Add an event to both online feature state stores.

    The event is recorded only AFTER its own feature vector has
    been built, preserving point-in-time semantics.
    """

    await velocity.record_transaction(
        transaction_id=event.transaction_id,
        customer_id=event.customer_id,
        device_id=event.device_id,
        ip_address=event.ip_address,
        event_time=event.event_time,
    )

    await graph.project_transaction(event)


async def main() -> None:
    print("=" * 88)
    print("RISK SENTINEL — V5 LIVE FEATURE SEMANTICS / PARITY GATE")
    print("=" * 88)

    redis = RedisClient()
    neo4j = Neo4jClient()

    velocity = VelocityStore(redis)
    graph = TransactionGraph(neo4j)

    service = OnlineFeatureService(
        redis=redis,
        neo4j=neo4j,
    )

    redis_ready = False
    neo4j_ready = False

    try:
        # ============================================================
        # Infrastructure readiness
        # ============================================================

        await redis.ping()
        redis_ready = True

        await neo4j.verify_connectivity()
        neo4j_ready = True

        await cleanup(redis, neo4j)

        base = datetime(
            2026,
            9,
            3,
            10,
            0,
            0,
            tzinfo=timezone.utc,
        )

        # ============================================================
        # T1 — first transaction
        #
        # T1 = 10:00:30.
        #
        # There must be no historical state for this customer,
        # device, IP, or payment method.
        # ============================================================

        t1 = make_event(
            transaction_id="parity-txn-001",
            event_id="parity-event-001",
            event_time=base + timedelta(seconds=30),
            amount="100.00",
        )

        f1 = await service.build(t1)

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

        for name, expected in expected_t1.items():
            assert_feature(
                name,
                getattr(f1, name),
                expected,
            )

        print("[PASS] T1 first-event point-in-time semantics")

        # Project T1 only AFTER scoring it.
        await record_state(velocity, graph, t1)

        # ============================================================
        # T2 — 31 seconds after T1
        #
        # T1 = 10:00:30
        # T2 = 10:01:01
        #
        # T1 is safely inside the one-minute window.
        # ============================================================

        t2 = make_event(
            transaction_id="parity-txn-002",
            event_id="parity-event-002",
            event_time=base + timedelta(
                minutes=1,
                seconds=1,
            ),
            amount="200.00",
        )

        f2 = await service.build(t2)

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

        for name, expected in expected_t2.items():
            assert_feature(
                name,
                getattr(f2, name),
                expected,
            )

        print("[PASS] T2 historical state accumulation")

        # Project T2 only AFTER scoring it.
        await record_state(velocity, graph, t2)

        # ============================================================
        # T3 — rolling one-minute window
        #
        # T2 = 10:01:01
        # T3 = 10:02:00
        #
        # T2 is 59 seconds old -> included.
        # T1 is 90 seconds old -> excluded.
        # ============================================================

        t3 = make_event(
            transaction_id="parity-txn-003",
            event_id="parity-event-003",
            event_time=base + timedelta(minutes=2),
            amount="300.00",
        )

        f3 = await service.build(t3)

        assert_feature(
            "customer_transactions_1m",
            f3.customer_transactions_1m,
            1,
        )

        assert_feature(
            "customer_transactions_1h",
            f3.customer_transactions_1h,
            2,
        )

        print("[PASS] 60-second rolling-window semantics")

        # ============================================================
        # T4 — exactly one hour after T2
        #
        # T2 = 10:01:01
        # T4 = 11:01:01
        #
        # T2 is exactly on the exclusive lower boundary and therefore
        # must not be included.
        #
        # T1 is also outside the one-hour window.
        # ============================================================

        t4 = make_event(
            transaction_id="parity-txn-004",
            event_id="parity-event-004",
            event_time=base + timedelta(
                hours=1,
                minutes=1,
                seconds=1,
            ),
            amount="400.00",
        )

        f4 = await service.build(t4)

        assert_feature(
            "customer_transactions_1h",
            f4.customer_transactions_1h,
            0,
        )

        print("[PASS] 1-hour exclusive lower boundary")

        # ============================================================
        # Same-timestamp isolation
        #
        # IMPORTANT:
        # T5 uses a completely fresh customer/device/IP/payment/
        # merchant entity set.
        #
        # This prevents T1-T4 graph history from contaminating the
        # test.
        #
        # T5A and T5B have exactly the same timestamp.
        # Neither is projected before scoring.
        # Therefore neither may see the other.
        # ============================================================

        same_time = base + timedelta(hours=2)

        t5a = make_event(
            transaction_id="parity-txn-005",
            event_id="parity-event-005",
            event_time=same_time,
            customer_id="parity-customer-002",
            device_id="parity-device-002",
            ip_address="10.20.30.41",
            payment_method_id="parity-payment-002",
            merchant_id="parity-merchant-002",
            amount="500.00",
        )

        t5b = make_event(
            transaction_id="parity-txn-006",
            event_id="parity-event-006",
            event_time=same_time,
            customer_id="parity-customer-002",
            device_id="parity-device-002",
            ip_address="10.20.30.41",
            payment_method_id="parity-payment-002",
            merchant_id="parity-merchant-002",
            amount="600.00",
        )

        # Score T5A before projecting it.
        f5a = await service.build(t5a)

        assert_feature(
            "customer_transactions_1m",
            f5a.customer_transactions_1m,
            0,
        )

        assert_feature(
            "customer_transactions_1h",
            f5a.customer_transactions_1h,
            0,
        )

        assert_feature(
            "customer_degree",
            f5a.customer_degree,
            0,
        )

        # IMPORTANT:
        # T5A is deliberately NOT projected here.
        #
        # Therefore T5B must see exactly the same historical state.
        f5b = await service.build(t5b)

        assert_feature(
            "customer_transactions_1m",
            f5b.customer_transactions_1m,
            0,
        )

        assert_feature(
            "customer_transactions_1h",
            f5b.customer_transactions_1h,
            0,
        )

        assert_feature(
            "customer_degree",
            f5b.customer_degree,
            0,
        )

        print("[PASS] Same-timestamp scoring isolation")

        # ============================================================
        # Duplicate idempotency
        #
        # T3 is now inserted into Redis/Neo4j.
        #
        # Recording T3 a second time must not create another Redis
        # velocity event.
        # ============================================================

        await record_state(
            velocity,
            graph,
            t3,
        )

        first = await velocity.get_velocity_features(
            customer_id=t3.customer_id,
            device_id=t3.device_id,
            ip_address=t3.ip_address,
            event_time=t3.event_time + timedelta(seconds=1),
        )

        await record_state(
            velocity,
            graph,
            t3,
        )

        second = await velocity.get_velocity_features(
            customer_id=t3.customer_id,
            device_id=t3.device_id,
            ip_address=t3.ip_address,
            event_time=t3.event_time + timedelta(seconds=1),
        )

        assert first == second, (
            f"Duplicate changed velocity state: "
            f"before={first}, after={second}"
        )

        print("[PASS] Duplicate velocity idempotency")

        # ============================================================
        # Final feature vector visibility
        # ============================================================

        print("\nRepresentative T2 feature vector:")
        print(f2)

        print("\n" + "=" * 88)
        print("V5 FEATURE SEMANTICS / PARITY GATE: PASS")
        print("=" * 88)

    finally:
        # Cleanup should never hide the original test failure.
        if redis_ready and neo4j_ready:
            try:
                await cleanup(
                    redis,
                    neo4j,
                )
            except Exception as exc:
                print(f"[WARN] Cleanup failed: {exc}")

        await redis.close()
        await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())