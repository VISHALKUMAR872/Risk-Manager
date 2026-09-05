import asyncio
import json
import os
from datetime import datetime, timedelta, timezone

from confluent_kafka import Producer

from risk_persistence.postgres.repositories import OutboxRepository
from risk_persistence.postgres.session import AsyncSessionLocal
from risk_stream.config import get_settings


class OutboxDeliveryError(Exception):
    """Raised when Kafka does not confirm successful delivery."""


class OutboxPublisher:
    def __init__(self):
        settings = get_settings()

        self.poll_interval_seconds = 1.0
        self.batch_size = 100

        # Test-only failure injection.
        # If set, the first publication attempt for this aggregate
        # fails before Kafka publish. The retry is allowed to succeed.
        self.test_fail_once_aggregate = os.getenv(
            "RISK_SENTINEL_OUTBOX_FAIL_ONCE_AGGREGATE"
        )
        self.test_failed_aggregates: set[str] = set()

        self.producer = Producer(
            {
                "bootstrap.servers": settings.redpanda_bootstrap_servers,
                "client.id": "risk-sentinel-outbox-publisher",
                "acks": "all",
                "enable.idempotence": True,
                "message.timeout.ms": 5000,
            }
        )

    def publish_event(self, event) -> None:
        aggregate_id = str(event.aggregate_id)

        if (
            self.test_fail_once_aggregate
            and aggregate_id == self.test_fail_once_aggregate
            and aggregate_id not in self.test_failed_aggregates
        ):
            self.test_failed_aggregates.add(aggregate_id)

            raise OutboxDeliveryError(
                "CONTROLLED TEST FAILURE: "
                "RISK_SENTINEL_OUTBOX_FAIL_ONCE_AGGREGATE"
            )

        delivery_error = None

        def delivery_callback(err, message):
            nonlocal delivery_error
            delivery_error = err

        self.producer.produce(
            topic=event.topic,
            key=event.message_key,
            value=json.dumps(event.payload),
            callback=delivery_callback,
        )

        remaining = self.producer.flush(timeout=10)

        if delivery_error is not None:
            raise OutboxDeliveryError(str(delivery_error))

        if remaining > 0:
            raise OutboxDeliveryError(
                f"Kafka delivery timed out with {remaining} message(s) outstanding"
            )

    async def publish_batch(self):
        async with AsyncSessionLocal() as session:
            repository = OutboxRepository(session)

            events = await repository.get_pending(limit=self.batch_size)

            if not events:
                return 0

            for event in events:
                try:
                    self.publish_event(event)

                    await repository.mark_published(event.id)

                    print(
                        "Published outbox event: "
                        f"{event.id} aggregate={event.aggregate_id}"
                    )

                except Exception as exc:
                    delay_seconds = min(
                        60,
                        2 ** min(event.attempts, 5),
                    )

                    next_attempt = (
                        datetime.now(timezone.utc)
                        + timedelta(seconds=delay_seconds)
                    )

                    await repository.mark_failed(
                        event.id,
                        str(exc),
                        next_attempt,
                    )

                    print(
                        "Outbox event scheduled for retry: "
                        f"{event.id} "
                        f"retry_in={delay_seconds}s "
                        f"error={exc}"
                    )

            await session.commit()

            return len(events)

    async def run(self):
        print("Risk Sentinel Outbox Publisher is running.")

        if self.test_fail_once_aggregate:
            print(
                "WARNING: OUTBOX FAILURE INJECTION ENABLED for aggregate="
                f"{self.test_fail_once_aggregate}"
            )

        while True:
            try:
                count = await self.publish_batch()

                if count == 0:
                    await asyncio.sleep(self.poll_interval_seconds)

            except asyncio.CancelledError:
                raise

            except Exception as exc:
                print(f"Outbox publisher error: {exc}")
                await asyncio.sleep(2.0)

    def close(self):
        self.producer.flush(timeout=10)
