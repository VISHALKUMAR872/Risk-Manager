import time

from risk_engine.contracts import FeatureVector, TransactionEvent
from risk_engine.observability import (
    NoOpRiskPipelineObserver,
    PipelineStage,
    RiskPipelineObserver,
)

from risk_persistence.neo4j import Neo4jClient
from risk_persistence.redis import RedisClient

from risk_features.online.neo4j_provider import (
    Neo4jFeatureProvider,
)
from risk_features.online.redis_provider import (
    RedisFeatureProvider,
)


class OnlineFeatureService:
    def __init__(
        self,
        redis: RedisClient,
        neo4j: Neo4jClient,
        observer: RiskPipelineObserver | None = None,
    ) -> None:
        self.redis = redis
        self.neo4j = neo4j
        self.observer = observer or NoOpRiskPipelineObserver()

        self.redis_provider = RedisFeatureProvider(redis)
        self.neo4j_provider = Neo4jFeatureProvider(neo4j)

    async def build(
        self,
        event: TransactionEvent,
    ) -> FeatureVector:

        # =========================================================
        # Redis velocity features
        # =========================================================

        started_at = time.perf_counter()

        try:
            velocity = (
                await self.redis_provider.get_velocity_features(
                    customer_id=event.customer_id,
                    device_id=event.device_id,
                    ip_address=event.ip_address,
                    event_time=event.event_time,
                )
            )

        except Exception:
            self.observer.observe_error(
                PipelineStage.REDIS_FEATURES
            )
            raise

        finally:
            self.observer.observe_latency(
                PipelineStage.REDIS_FEATURES,
                time.perf_counter() - started_at,
            )

        # =========================================================
        # Neo4j graph features
        # =========================================================

        started_at = time.perf_counter()

        try:
            graph = (
                await self.neo4j_provider.get_graph_features(
                    customer_id=event.customer_id,
                    device_id=event.device_id,
                    ip_address=event.ip_address,
                    payment_method_id=event.payment_method_id,
                    merchant_id=event.merchant_id,
                    event_time=event.event_time,
                )
            )

        except Exception:
            self.observer.observe_error(
                PipelineStage.NEO4J_FEATURES
            )
            raise

        finally:
            self.observer.observe_latency(
                PipelineStage.NEO4J_FEATURES,
                time.perf_counter() - started_at,
            )

        # =========================================================
        # Construct point-in-time feature vector
        # =========================================================

        return FeatureVector(
            transaction_id=event.transaction_id,
            as_of_time=event.event_time,
            amount=float(event.amount),
            currency=event.currency,
            **velocity,
            **graph,
        )