import asyncio

from risk_persistence.redis import RedisClient, VelocityStore


async def main() -> None:
    redis = RedisClient()
    velocity = VelocityStore(redis)

    result = await velocity.record_transaction(
        transaction_id="txn-velocity-test",
        customer_id="cust-test",
        device_id="device-test",
        ip_address="10.0.0.1",
    )

    print("Velocity result:")
    print(result)

    await redis.close()


if __name__ == "__main__":
    asyncio.run(main())