import asyncio
from datetime import datetime

from risk_persistence.redis import RedisClient, VelocityStore


CUSTOMER_ID = "redis-test-customer"
DEVICE_ID = "redis-test-device"
IP_ADDRESS = "10.0.0.99"

T1 = datetime.fromisoformat(
    "2026-09-01T10:00:00+00:00"
)

T2 = datetime.fromisoformat(
    "2026-09-01T10:00:30+00:00"
)

T3 = datetime.fromisoformat(
    "2026-09-01T10:02:00+00:00"
)


async def main() -> None:
    redis = RedisClient()
    velocity = VelocityStore(redis)

    try:
        if not await redis.ping():
            raise RuntimeError("Redis connectivity check failed.")

        # Clean test state.
        await redis.client.delete(
            f"customer:{CUSTOMER_ID}:tx_events",
            f"device:{DEVICE_ID}:tx_events",
            f"ip:{IP_ADDRESS}:tx_events",
            "velocity:processed:redis-test-t1",
            "velocity:processed:redis-test-t2",
        )

        # Before T1: nothing should exist.
        before_t1 = await velocity.get_velocity_features(
            customer_id=CUSTOMER_ID,
            device_id=DEVICE_ID,
            ip_address=IP_ADDRESS,
            event_time=T1,
        )

        print("Before T1:")
        print(before_t1)

        # Record T1 AFTER its features have been evaluated.
        await velocity.record_transaction(
            transaction_id="redis-test-t1",
            customer_id=CUSTOMER_ID,
            device_id=DEVICE_ID,
            ip_address=IP_ADDRESS,
            event_time=T1,
        )

        # T1 should now be visible to T2.
        at_t2 = await velocity.get_velocity_features(
            customer_id=CUSTOMER_ID,
            device_id=DEVICE_ID,
            ip_address=IP_ADDRESS,
            event_time=T2,
        )

        print()
        print("At T2:")
        print(at_t2)

        # T3 is 2 minutes after T1.
        # Therefore T1 must NOT appear in the 1-minute customer window,
        # but must remain in the 1-hour window.
        at_t3 = await velocity.get_velocity_features(
            customer_id=CUSTOMER_ID,
            device_id=DEVICE_ID,
            ip_address=IP_ADDRESS,
            event_time=T3,
        )

        print()
        print("At T3:")
        print(at_t3)

        assert before_t1 == {
            "customer_transactions_1m": 0,
            "customer_transactions_1h": 0,
            "device_transactions_1h": 0,
            "ip_transactions_1h": 0,
        }

        assert at_t2 == {
            "customer_transactions_1m": 1,
            "customer_transactions_1h": 1,
            "device_transactions_1h": 1,
            "ip_transactions_1h": 1,
        }

        assert at_t3 == {
            "customer_transactions_1m": 0,
            "customer_transactions_1h": 1,
            "device_transactions_1h": 1,
            "ip_transactions_1h": 1,
        }

        # Verify idempotency: recording T1 again must not double-count it.
        await velocity.record_transaction(
            transaction_id="redis-test-t1",
            customer_id=CUSTOMER_ID,
            device_id=DEVICE_ID,
            ip_address=IP_ADDRESS,
            event_time=T1,
        )

        after_duplicate = await velocity.get_velocity_features(
            customer_id=CUSTOMER_ID,
            device_id=DEVICE_ID,
            ip_address=IP_ADDRESS,
            event_time=T2,
        )

        print()
        print("After duplicate T1:")
        print(after_duplicate)

        assert after_duplicate == at_t2

        # Record T2.
        await velocity.record_transaction(
            transaction_id="redis-test-t2",
            customer_id=CUSTOMER_ID,
            device_id=DEVICE_ID,
            ip_address=IP_ADDRESS,
            event_time=T2,
        )

        at_t3_after_t2 = await velocity.get_velocity_features(
            customer_id=CUSTOMER_ID,
            device_id=DEVICE_ID,
            ip_address=IP_ADDRESS,
            event_time=T3,
        )

        print()
        print("At T3 after T2:")
        print(at_t3_after_t2)

        assert at_t3_after_t2 == {
            "customer_transactions_1m": 0,
            "customer_transactions_1h": 2,
            "device_transactions_1h": 2,
            "ip_transactions_1h": 2,
        }

        print()
        print("Redis event-time velocity test PASSED.")

    finally:
        await redis.client.delete(
            f"customer:{CUSTOMER_ID}:tx_events",
            f"device:{DEVICE_ID}:tx_events",
            f"ip:{IP_ADDRESS}:tx_events",
            "velocity:processed:redis-test-t1",
            "velocity:processed:redis-test-t2",
        )

        await redis.close()


if __name__ == "__main__":
    asyncio.run(main())