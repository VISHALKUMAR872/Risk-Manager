from datetime import datetime

from risk_persistence.redis import RedisClient, VelocityStore


class RedisFeatureProvider:
    def __init__(self, redis: RedisClient):
        self.redis = redis
        self.velocity = VelocityStore(redis)

    async def get_velocity_features(
        self,
        customer_id: str,
        device_id: str,
        ip_address: str,
        event_time: datetime,
    ) -> dict[str, int]:
        return await self.velocity.get_velocity_features(
            customer_id=customer_id,
            device_id=device_id,
            ip_address=ip_address,
            event_time=event_time,
        )