import json
from dataclasses import dataclass
from typing import Any

from confluent_kafka import Consumer, KafkaError, KafkaException
from pydantic import ValidationError

from risk_engine.contracts.transaction import TransactionEvent
from risk_stream.config import get_settings
from risk_stream.producers.dlq_producer import DlqProducer
from risk_stream.metrics import (
    consumer_commit_errors_total,
    consumer_poll_errors_total,
    dlq_messages_total,
)


@dataclass(frozen=True)
class ReceivedTransaction:
    event: TransactionEvent
    message: Any


class TransactionConsumer:
    def __init__(self) -> None:
        settings = get_settings()

        self.consumer = Consumer(
            {
                "bootstrap.servers": settings.redpanda_bootstrap_servers,
                "group.id": settings.consumer_group,
                "auto.offset.reset": "earliest",
                "enable.auto.commit": False,
            }
        )

        self.topic = settings.transaction_topic
        self.dlq_producer = DlqProducer()

    def start(self) -> None:
        self.consumer.subscribe([self.topic])

        settings = get_settings()

        print(
            "Consumer started. "
            f"topic={self.topic}, "
            f"group={settings.consumer_group}"
        )

    def consume_one(
        self,
        timeout: float = 1.0,
    ) -> ReceivedTransaction | None:
        message = self.consumer.poll(timeout)

        if message is None:
            return None

        if message.error():
            if message.error().code() == KafkaError._PARTITION_EOF:
                return None

            consumer_poll_errors_total.inc()

            raise KafkaException(message.error())

        raw_payload = message.value().decode("utf-8")

        try:
            payload: dict[str, Any] = json.loads(raw_payload)

            event = TransactionEvent.model_validate(payload)

            print(
                "Validated transaction: "
                f"{event.transaction_id}"
            )

            return ReceivedTransaction(
                event=event,
                message=message,
            )

        except (
            json.JSONDecodeError,
            UnicodeDecodeError,
            ValidationError,
        ) as exc:
            print(
                "Poison message detected: "
                f"topic={message.topic()} "
                f"partition={message.partition()} "
                f"offset={message.offset()} "
                f"error={exc}"
            )

            try:
                self.dlq_producer.publish(
                    original_payload=raw_payload,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    topic=message.topic(),
                    partition=message.partition(),
                    offset=message.offset(),
                )

                dlq_messages_total.inc()

                self.commit(message)

                print(
                    "Committed poison-message offset after DLQ publication: "
                    f"partition={message.partition()} "
                    f"offset={message.offset()}"
                )

            except Exception as dlq_exc:
                print(
                    "CRITICAL: failed to publish poison message to DLQ: "
                    f"{dlq_exc}"
                )

                # Do not commit the original message if
                # publishing to the DLQ failed.

            return None

        except Exception as exc:
            print(
                "Unexpected transaction consumer error: "
                f"{exc}"
            )

            # Unexpected errors remain retryable.
            return None

    def publish_to_dlq(
        self,
        *,
        message: Any,
        error_type: str,
        error_message: str,
        attempt_count: int,
    ) -> None:
        raw_payload = message.value().decode("utf-8")

        self.dlq_producer.publish(
            original_payload=raw_payload,
            error_type=error_type,
            error_message=error_message,
            topic=message.topic(),
            partition=message.partition(),
            offset=message.offset(),
            attempt_count=attempt_count,
        )

        dlq_messages_total.inc()

    def commit(self, message: Any) -> None:
        try:
            self.consumer.commit(
                message=message,
                asynchronous=False,
            )

        except Exception:
            consumer_commit_errors_total.inc()
            raise

        print(
            "Committed Redpanda offset: "
            f"partition={message.partition()} "
            f"offset={message.offset()}"
        )

    def close(self) -> None:
        self.consumer.close()
        self.dlq_producer.close()