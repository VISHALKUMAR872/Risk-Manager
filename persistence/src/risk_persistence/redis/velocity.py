from datetime import datetime, timezone

from risk_persistence.redis.client import RedisClient


class VelocityStore:
    IDEMPOTENCY_TTL = 3600

    CUSTOMER_1M = 60
    CUSTOMER_1H = 3600
    DEVICE_1H = 3600
    IP_1H = 3600

    def __init__(self, redis: RedisClient):
        self.redis = redis

    @staticmethod
    def _timestamp(event_time: datetime) -> float:
        if event_time.tzinfo is None:
            event_time = event_time.replace(tzinfo=timezone.utc)

        return event_time.timestamp()

    @staticmethod
    def _member(transaction_id: str) -> str:
        return transaction_id

    async def get_velocity_features(
        self,
        customer_id: str,
        device_id: str,
        ip_address: str,
        event_time: datetime,
    ) -> dict[str, int]:
        timestamp = self._timestamp(event_time)

        customer_1m = await self.redis.count_events_in_window(
            key=f"customer:{customer_id}:tx_events",
            start=timestamp - self.CUSTOMER_1M,
            end=timestamp,
        )

        customer_1h = await self.redis.count_events_in_window(
            key=f"customer:{customer_id}:tx_events",
            start=timestamp - self.CUSTOMER_1H,
            end=timestamp,
        )

        device_1h = await self.redis.count_events_in_window(
            key=f"device:{device_id}:tx_events",
            start=timestamp - self.DEVICE_1H,
            end=timestamp,
        )

        ip_1h = await self.redis.count_events_in_window(
            key=f"ip:{ip_address}:tx_events",
            start=timestamp - self.IP_1H,
            end=timestamp,
        )

        return {
            "customer_transactions_1m": customer_1m,
            "customer_transactions_1h": customer_1h,
            "device_transactions_1h": device_1h,
            "ip_transactions_1h": ip_1h,
        }

    async def record_transaction(
        self,
        transaction_id: str,
        customer_id: str,
        device_id: str,
        ip_address: str,
        event_time: datetime,
    ) -> None:
        timestamp = self._timestamp(event_time)
        member = self._member(transaction_id)

        await self.redis.record_velocity_event(
            idempotency_key=f"velocity:processed:{transaction_id}",
            customer_key=f"customer:{customer_id}:tx_events",
            device_key=f"device:{device_id}:tx_events",
            ip_key=f"ip:{ip_address}:tx_events",
            score=timestamp,
            member=member,
            ttl_seconds=self.IDEMPOTENCY_TTL,
        )