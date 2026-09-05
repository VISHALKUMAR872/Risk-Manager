import os
import time

from risk_stream.observability import (
    PrometheusRiskPipelineObserver,
)

from risk_engine.calibration import IsotonicCalibrator
from risk_engine.contracts import TransactionEvent
from risk_engine.decision import DecisionService
from risk_engine.expected_loss import ExpectedLossCalculator
from risk_engine.inference import (
    CatBoostRiskModel,
    InferenceService,
)
from risk_engine.policy import PolicyEngine

from risk_features.online.service import OnlineFeatureService

from risk_persistence.neo4j import Neo4jClient, TransactionGraph
from risk_persistence.postgres.mappers import (
    risk_decision_from_contract,
    transaction_from_event,
)
from risk_persistence.postgres.models.outbox_event import (
    OutboxEvent,
)
from risk_persistence.postgres.repositories import (
    OutboxRepository,
    RiskDecisionRepository,
    TransactionRepository,
)
from risk_persistence.postgres.session import AsyncSessionLocal
from risk_persistence.redis import RedisClient, VelocityStore

from risk_stream.config import get_settings
from risk_stream.decision.feature_provider import OnlineFeatureProvider
from risk_stream.metrics import (
    risk_decision_latency_seconds,
    risk_evaluation_errors_total,
    risk_evaluation_latency_seconds,
    risk_neo4j_errors_total,
    risk_neo4j_latency_seconds,
    risk_postgres_errors_total,
    risk_postgres_latency_seconds,
    risk_redis_errors_total,
    risk_redis_latency_seconds,
)


class TransactionProcessor:
    def __init__(self) -> None:
        # ---------------------------------------------------------
        # 1. Settings + observability
        # ---------------------------------------------------------
        settings = get_settings()

        self.settings = settings
        self.observer = PrometheusRiskPipelineObserver()

        # ---------------------------------------------------------
        # 2. Redis
        # ---------------------------------------------------------
        self.fail_once = (
                os.getenv("RISK_SENTINEL_FAIL_ONCE") == "1"
        )

        self.fail_always = (
                os.getenv("RISK_SENTINEL_FAIL_ALWAYS") == "1"
        )

        self.redis = RedisClient()
        self.velocity = VelocityStore(self.redis)

        # ---------------------------------------------------------
        # 3. Neo4j
        # ---------------------------------------------------------
        self.neo4j = Neo4jClient()
        self.graph = TransactionGraph(self.neo4j)

        # ---------------------------------------------------------
        # 4. Online feature service
        # ---------------------------------------------------------
        self.feature_service = OnlineFeatureService(
            redis=self.redis,
            neo4j=self.neo4j,
            observer=self.observer,
        )

        # ---------------------------------------------------------
        # 5. FeatureProvider adapter
        # ---------------------------------------------------------
        self.feature_provider = OnlineFeatureProvider(
            feature_service=self.feature_service,
        )

        # ---------------------------------------------------------
        # 6. Production ML model + calibration
        # ---------------------------------------------------------
        self.model = CatBoostRiskModel(
            settings.risk_model_path,
            version=settings.model_version,
        )

        self.calibrator = IsotonicCalibrator(
            settings.risk_calibrator_path,
            version=settings.calibration_version,
        )

        # ---------------------------------------------------------
        # 7. End-to-end risk decision pipeline
        # ---------------------------------------------------------
        self.decision_service = DecisionService(
            feature_provider=self.feature_provider,
            inference_service=InferenceService(
                model=self.model,
                calibrator=self.calibrator,
                observer=self.observer,
            ),
            observer=self.observer,
            expected_loss_calculator=ExpectedLossCalculator(
                loss_given_fraud=0.80,
            ),
            policy_engine=PolicyEngine(),
        )

    async def process(
        self,
        event: TransactionEvent,
    ) -> None:

        # =========================================================
        # 1. Create transaction / handle duplicate delivery
        # =========================================================

        postgres_started_at = time.perf_counter()

        try:
            async with AsyncSessionLocal() as session:
                transaction_repository = TransactionRepository(
                    session
                )

                existing = (
                    await transaction_repository
                    .get_by_transaction_id(
                        event.transaction_id
                    )
                )

                if existing is not None:
                    if existing.status == "DECIDED":
                        print(
                            "Transaction already decided: "
                            f"{event.transaction_id}"
                        )
                        return

                    if existing.status == "PROCESSING":
                        print(
                            "Transaction already processing: "
                            f"{event.transaction_id}"
                        )

                    if existing.status == "FAILED":
                        print(
                            "Retrying failed transaction: "
                            f"{event.transaction_id}"
                        )

                    existing.status = "PROCESSING"

                    await session.commit()

                else:
                    transaction = transaction_from_event(
                        event
                    )

                    transaction.status = "PROCESSING"

                    await transaction_repository.create(
                        transaction
                    )

                    await session.commit()

                    print(
                        "Persisted transaction: "
                        f"{event.transaction_id}"
                    )

        except Exception:
            risk_postgres_errors_total.inc()
            raise

        finally:
            risk_postgres_latency_seconds.observe(
                time.perf_counter() - postgres_started_at
            )

        try:
            # =====================================================
            # Controlled failure injection for retry testing
            # =====================================================

            if self.fail_always:
                print(
                    "Injecting permanent controlled failure: "
                    f"{event.transaction_id}"
                )

                raise RuntimeError(
                    "CONTROLLED_PERMANENT_FAILURE_TEST"
                )

            if self.fail_once:
                self.fail_once = False

                print(
                    "Injecting controlled failure: "
                    f"{event.transaction_id}"
                )

                raise RuntimeError(
                    "CONTROLLED_RETRY_TEST"
                )

            # =====================================================
            # 2. Build features + calculate risk decision
            # =====================================================

            evaluation_started_at = time.perf_counter()

            try:
                decision = (
                    await self.decision_service.evaluate(
                        event
                    )
                )

            except Exception:
                risk_evaluation_errors_total.inc()
                raise

            finally:
                risk_evaluation_latency_seconds.observe(
                    time.perf_counter()
                    - evaluation_started_at
                )

            print(
                "Risk score: "
                f"{decision.fraud_probability:.4f}"
            )

            print(
                "Expected loss: "
                f"{decision.expected_loss:.2f} "
                f"{event.currency}"
            )

            print(
                "Risk level: "
                f"{decision.risk_level}"
            )

            print(
                "Risk decision: "
                f"{decision.decision}"
            )

            print(
                "Reason codes: "
                f"{decision.reason_codes}"
            )

            # =====================================================
            # 3. Persist risk decision + outbox event atomically
            # =====================================================

            postgres_started_at = time.perf_counter()

            try:
                async with AsyncSessionLocal() as session:
                    decision_repository = (
                        RiskDecisionRepository(session)
                    )

                    outbox_repository = OutboxRepository(
                        session
                    )

                    existing_decision = (
                        await decision_repository
                        .get_by_transaction_id(
                            event.transaction_id
                        )
                    )

                    if existing_decision is None:
                        decision_model = (
                            risk_decision_from_contract(
                                decision
                            )
                        )

                        await decision_repository.create(
                            decision_model
                        )

                        outbox_event = OutboxEvent(
                            event_type="RiskDecisionCreated",
                            aggregate_type="RiskDecision",
                            aggregate_id=event.transaction_id,
                            topic=self.settings.risk_decision_topic,
                            message_key=event.transaction_id,
                            payload=decision.model_dump(
                                mode="json"
                            ),
                        )

                        await outbox_repository.create(
                            outbox_event
                        )

                        await session.commit()

                        print(
                            "Persisted risk decision + "
                            "outbox event: "
                            f"{event.transaction_id}"
                        )

                    else:
                        print(
                            "Risk decision already exists: "
                            f"{event.transaction_id}"
                        )

            except Exception:
                risk_postgres_errors_total.inc()
                raise

            finally:
                risk_postgres_latency_seconds.observe(
                    time.perf_counter()
                    - postgres_started_at
                )

            # =====================================================
            # 4. Update Redis velocity state
            # =====================================================

            redis_started_at = time.perf_counter()

            try:
                velocity = (
                    await self.velocity.record_transaction(
                        transaction_id=event.transaction_id,
                        customer_id=event.customer_id,
                        device_id=event.device_id,
                        ip_address=event.ip_address,
                        event_time=event.event_time,
                    )
                )

            except Exception:
                risk_redis_errors_total.inc()
                raise

            finally:
                risk_redis_latency_seconds.observe(
                    time.perf_counter()
                    - redis_started_at
                )

            print(
                f"Velocity state: {velocity}"
            )

            # =====================================================
            # 5. Project transaction into Neo4j
            # =====================================================

            neo4j_started_at = time.perf_counter()

            try:
                await self.graph.project_transaction(
                    event
                )

            except Exception:
                risk_neo4j_errors_total.inc()
                raise

            finally:
                risk_neo4j_latency_seconds.observe(
                    time.perf_counter()
                    - neo4j_started_at
                )

            print(
                "Projected transaction to Neo4j: "
                f"{event.transaction_id}"
            )

            # =====================================================
            # 6. Mark transaction as DECIDED
            # =====================================================

            postgres_started_at = time.perf_counter()

            try:
                async with AsyncSessionLocal() as session:
                    transaction_repository = (
                        TransactionRepository(session)
                    )

                    await transaction_repository.update_status(
                        transaction_id=event.transaction_id,
                        status="DECIDED",
                    )

                    await session.commit()

            except Exception:
                risk_postgres_errors_total.inc()
                raise

            finally:
                risk_postgres_latency_seconds.observe(
                    time.perf_counter()
                    - postgres_started_at
                )

            print(
                "Transaction status: DECIDED "
                f"{event.transaction_id}"
            )

        except Exception:
            # =====================================================
            # Failure state
            # =====================================================

            postgres_started_at = time.perf_counter()

            try:
                async with AsyncSessionLocal() as session:
                    transaction_repository = (
                        TransactionRepository(session)
                    )

                    await transaction_repository.update_status(
                        transaction_id=event.transaction_id,
                        status="FAILED",
                    )

                    await session.commit()

            except Exception:
                risk_postgres_errors_total.inc()
                raise

            finally:
                risk_postgres_latency_seconds.observe(
                    time.perf_counter()
                    - postgres_started_at
                )

            print(
                "Transaction status: FAILED "
                f"{event.transaction_id}"
            )

            raise

    async def close(self) -> None:
        await self.redis.close()
        await self.neo4j.close()