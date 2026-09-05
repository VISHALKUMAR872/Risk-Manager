import json
from datetime import datetime, timezone
from typing import Any

from confluent_kafka import Producer

from risk_stream.config import get_settings


class DlqProducer:
    def __init__(self) -> None:
        settings = get_settings()

        self.topic = settings.transactions_dlq_topic

        self.producer = Producer(
            {
                "bootstrap.servers": settings.redpanda_bootstrap_servers,
                "client.id": "risk-sentinel-dlq-producer",
                "acks": "all",
                "enable.idempotence": True,
                "message.timeout.ms": 5000,
            }
        )

    def publish(
            self,
            *,
            original_payload: str,
            error_type: str,
            error_message: str,
            topic: str,
            partition: int,
            offset: int,
            attempt_count: int = 1,
    ) -> None:
        dlq_event: dict[str, Any] = {
            "original_payload": original_payload,
            "error_type": error_type,
            "error_message": error_message,
            "original_topic": topic,
            "original_partition": partition,
            "original_offset": offset,
            "attempt_count": attempt_count,
            "failed_at": datetime.now(timezone.utc).isoformat(),
        }

        delivery_error = None

        def delivery_callback(err, message) -> None:
            nonlocal delivery_error
            delivery_error = err

        self.producer.produce(
            topic=self.topic,
            value=json.dumps(dlq_event),
            callback=delivery_callback,
        )

        remaining = self.producer.flush(timeout=10)

        if delivery_error is not None:
            raise RuntimeError(
                f"DLQ delivery failed: {delivery_error}"
            )

        if remaining > 0:
            raise RuntimeError(
                f"DLQ delivery timed out with {remaining} message(s) outstanding"
            )

        print(
            "Published poison message to DLQ: "
            f"topic={self.topic} "
            f"original_topic={topic} "
            f"partition={partition} "
            f"offset={offset}"
        )

    def close(self) -> None:
        self.producer.flush(timeout=10)