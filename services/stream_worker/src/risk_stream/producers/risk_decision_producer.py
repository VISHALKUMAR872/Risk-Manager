import json
from typing import Any

from confluent_kafka import Producer

from risk_engine.contracts import RiskDecision
from risk_stream.config import get_settings


class RiskDecisionProducer:
    def __init__(self) -> None:
        settings = get_settings()

        self.topic = settings.risk_decision_topic

        self.producer = Producer(
            {
                "bootstrap.servers": settings.redpanda_bootstrap_servers,
                "client.id": "risk-sentinel-risk-decision-producer",
                "acks": "all",
            }
        )

    @staticmethod
    def _delivery_callback(
        err: Any,
        message: Any,
    ) -> None:
        if err is not None:
            print(f"Risk decision delivery failed: {err}")
            return

        print(
            "Delivered risk decision "
            f"to {message.topic()} "
            f"[partition={message.partition()} "
            f"offset={message.offset()}]"
        )

    def publish(self, decision: RiskDecision) -> None:
        payload = decision.model_dump(mode="json")

        self.producer.produce(
            topic=self.topic,
            key=decision.transaction_id,
            value=json.dumps(payload),
            callback=self._delivery_callback,
        )

        self.producer.poll(0)

    def flush(self) -> None:
        self.producer.flush()

    def close(self) -> None:
        self.producer.flush()