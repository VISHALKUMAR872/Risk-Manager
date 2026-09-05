import json
from typing import Any

from confluent_kafka import Producer

from risk_engine.contracts.transaction import TransactionEvent

from risk_stream.config import get_settings


class TransactionProducer:
    def __init__(self) -> None:
        settings = get_settings()

        self.topic = settings.transaction_topic

        self.producer = Producer(
            {
                "bootstrap.servers": settings.redpanda_bootstrap_servers,
                "client.id": "risk-sentinel-transaction-producer",
            }
        )

    @staticmethod
    def _delivery_callback(err: Any, message: Any) -> None:
        if err is not None:
            print(f"Delivery failed: {err}")
            return

        print(
            "Delivered transaction event "
            f"to {message.topic()} "
            f"[partition={message.partition()} "
            f"offset={message.offset()}]"
        )

    def publish(self, event: TransactionEvent) -> None:
        payload = event.model_dump(mode="json")

        self.producer.produce(
            topic=self.topic,
            key=event.transaction_id,
            value=json.dumps(payload),
            callback=self._delivery_callback,
        )

        self.producer.poll(0)

    def flush(self) -> None:
        self.producer.flush()