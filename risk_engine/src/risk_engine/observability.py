from abc import ABC, abstractmethod
from enum import StrEnum


class PipelineStage(StrEnum):
    FEATURE_BUILD = "feature_build"
    REDIS_FEATURES = "redis_features"
    NEO4J_FEATURES = "neo4j_features"
    MODEL_INFERENCE = "model_inference"
    CALIBRATION = "calibration"
    EXPECTED_LOSS = "expected_loss"
    POLICY = "policy"


class RiskPipelineObserver(ABC):
    """
    Infrastructure-independent observation interface.

    The risk engine does not depend on Prometheus, OpenTelemetry,
    or any other telemetry implementation.
    """

    @abstractmethod
    def observe_latency(
        self,
        stage: PipelineStage,
        seconds: float,
    ) -> None:
        ...

    @abstractmethod
    def observe_error(
        self,
        stage: PipelineStage,
    ) -> None:
        ...

    @abstractmethod
    def observe_decision(
        self,
        decision: str,
        risk_level: str,
    ) -> None:
        ...


class NoOpRiskPipelineObserver(RiskPipelineObserver):
    """
    Default observer for environments without telemetry.
    """

    def observe_latency(
        self,
        stage: PipelineStage,
        seconds: float,
    ) -> None:
        pass

    def observe_error(
        self,
        stage: PipelineStage,
    ) -> None:
        pass

    def observe_decision(
        self,
        decision: str,
        risk_level: str,
    ) -> None:
        pass