from redis.asyncio import Redis

from risk_persistence.config import get_settings


class RedisClient:
    def __init__(self) -> None:
        settings = get_settings()

        self.client = Redis.from_url(
            settings.redis_url,
            decode_responses=True,
        )

    async def ping(self) -> bool:
        return bool(await self.client.ping())

    async def increment_with_expiry(
        self,
        key: str,
        ttl_seconds: int,
    ) -> int:
        value = await self.client.incr(key)

        if value == 1:
            await self.client.expire(
                key,
                ttl_seconds,
            )

        return value

    async def mark_once(
        self,
        key: str,
        ttl_seconds: int,
    ) -> bool:
        created = await self.client.set(
            key,
            "1",
            ex=ttl_seconds,
            nx=True,
        )

        return created is True

    async def get(
        self,
        key: str,
    ) -> str | None:
        return await self.client.get(key)

    async def add_event(
        self,
        key: str,
        score: float,
        member: str,
    ) -> None:
        await self.client.zadd(
            key,
            {member: score},
        )

    async def count_events_in_window(
        self,
        key: str,
        start: float,
        end: float,
    ) -> int:
        return int(
            await self.client.zcount(
                key,
                f"({start}",
                f"({end}",
            )
        )

    async def record_velocity_event(
        self,
        *,
        idempotency_key: str,
        customer_key: str,
        device_key: str,
        ip_key: str,
        score: float,
        member: str,
        ttl_seconds: int,
    ) -> bool:
        script = """
        if redis.call("EXISTS", KEYS[4]) == 1 then
            return 0
        end

        redis.call(
            "ZADD",
            KEYS[1],
            ARGV[1],
            ARGV[2]
        )

        redis.call(
            "ZADD",
            KEYS[2],
            ARGV[1],
            ARGV[2]
        )

        redis.call(
            "ZADD",
            KEYS[3],
            ARGV[1],
            ARGV[2]
        )

        redis.call(
            "SET",
            KEYS[4],
            "1",
            "EX",
            ARGV[3]
        )

        return 1
        """

        result = await self.client.eval(
            script,
            4,
            customer_key,
            device_key,
            ip_key,
            idempotency_key,
            str(score),
            member,
            str(ttl_seconds),
        )

        return int(result) == 1

    async def close(self) -> None:
        await self.client.aclose()