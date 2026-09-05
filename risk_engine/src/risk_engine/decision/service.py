import time

from risk_engine.contracts import (
    RiskDecision,
    TransactionEvent,
)
from risk_engine.expected_loss import ExpectedLossCalculator
from risk_engine.inference import InferenceService
from risk_engine.policy import PolicyEngine

from risk_engine.decision.providers import FeatureProvider
from risk_engine.observability import (
    NoOpRiskPipelineObserver,
    PipelineStage,
    RiskPipelineObserver,
)


class DecisionService:
    """
    End-to-end risk decision orchestration.

    Flow:

        TransactionEvent
            ↓
        FeatureProvider
            ↓
        FeatureVector
            ↓
        InferenceService
            ↓
        RiskScore
            ↓
        ExpectedLossCalculator
            ↓
        ExpectedLoss
            ↓
        PolicyEngine
            ↓
        RiskDecision
    """

    def __init__(
        self,
        feature_provider: FeatureProvider,
        inference_service: InferenceService,
        expected_loss_calculator: ExpectedLossCalculator,
        policy_engine: PolicyEngine,
        observer: RiskPipelineObserver | None = None,
    ) -> None:
        self.feature_provider = feature_provider
        self.inference_service = inference_service
        self.expected_loss_calculator = expected_loss_calculator
        self.policy_engine = policy_engine
        self.observer = observer or NoOpRiskPipelineObserver()

    async def evaluate(
        self,
        event: TransactionEvent,
    ) -> RiskDecision:

        # =========================================================
        # 1. Build online features
        # =========================================================

        started_at = time.perf_counter()

        try:
            features = await self.feature_provider.build(event)

        except Exception:
            self.observer.observe_error(
                PipelineStage.FEATURE_BUILD
            )
            raise

        finally:
            self.observer.observe_latency(
                PipelineStage.FEATURE_BUILD,
                time.perf_counter() - started_at,
            )

        # =========================================================
        # 2. Model inference + calibration
        # =========================================================

        risk_score = self.inference_service.predict(features)

        # =========================================================
        # 3. Expected loss
        # =========================================================

        started_at = time.perf_counter()

        try:
            expected_loss = (
                self.expected_loss_calculator.calculate(
                    risk_score=risk_score,
                    exposure_amount=float(event.amount),
                    currency=event.currency,
                )
            )

        except Exception:
            self.observer.observe_error(
                PipelineStage.EXPECTED_LOSS
            )
            raise

        finally:
            self.observer.observe_latency(
                PipelineStage.EXPECTED_LOSS,
                time.perf_counter() - started_at,
            )

        # =========================================================
        # 4. Deterministic policy
        # =========================================================

        started_at = time.perf_counter()

        try:
            decision = self.policy_engine.decide(
                risk_score=risk_score,
                expected_loss=expected_loss,
            )

        except Exception:
            self.observer.observe_error(
                PipelineStage.POLICY
            )
            raise

        finally:
            self.observer.observe_latency(
                PipelineStage.POLICY,
                time.perf_counter() - started_at,
            )

        self.observer.observe_decision(
            decision=str(decision.decision),
            risk_level=str(decision.risk_level),
        )

        return decision