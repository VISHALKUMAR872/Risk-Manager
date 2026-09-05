from risk_engine.observability import (
    PipelineStage,
    RiskPipelineObserver,
)

from risk_stream.metrics import (
    risk_calibration_errors_total,
    risk_calibration_latency_seconds,
    risk_decisions_total,
    risk_expected_loss_errors_total,
    risk_expected_loss_latency_seconds,
    risk_feature_build_errors_total,
    risk_feature_build_latency_seconds,
    risk_model_inference_errors_total,
    risk_model_inference_latency_seconds,
    risk_neo4j_errors_total,
    risk_neo4j_latency_seconds,
    risk_policy_errors_total,
    risk_policy_latency_seconds,
    risk_redis_errors_total,
    risk_redis_latency_seconds,
    risk_levels_total,
)


class PrometheusRiskPipelineObserver(
    RiskPipelineObserver
):
    """
    Prometheus-backed implementation of the risk pipeline observer.
    """

    _LATENCY_METRICS = {
        PipelineStage.FEATURE_BUILD:
            risk_feature_build_latency_seconds,

        PipelineStage.REDIS_FEATURES:
            risk_redis_latency_seconds,

        PipelineStage.NEO4J_FEATURES:
            risk_neo4j_latency_seconds,

        PipelineStage.MODEL_INFERENCE:
            risk_model_inference_latency_seconds,

        PipelineStage.CALIBRATION:
            risk_calibration_latency_seconds,

        PipelineStage.EXPECTED_LOSS:
            risk_expected_loss_latency_seconds,

        PipelineStage.POLICY:
            risk_policy_latency_seconds,
    }

    _ERROR_METRICS = {
        PipelineStage.FEATURE_BUILD:
            risk_feature_build_errors_total,

        PipelineStage.REDIS_FEATURES:
            risk_redis_errors_total,

        PipelineStage.NEO4J_FEATURES:
            risk_neo4j_errors_total,

        PipelineStage.MODEL_INFERENCE:
            risk_model_inference_errors_total,

        PipelineStage.CALIBRATION:
            risk_calibration_errors_total,

        PipelineStage.EXPECTED_LOSS:
            risk_expected_loss_errors_total,

        PipelineStage.POLICY:
            risk_policy_errors_total,
    }

    def observe_latency(
        self,
        stage: PipelineStage,
        seconds: float,
    ) -> None:
        metric = self._LATENCY_METRICS.get(stage)

        if metric is not None:
            metric.observe(seconds)

    def observe_error(
        self,
        stage: PipelineStage,
    ) -> None:
        metric = self._ERROR_METRICS.get(stage)

        if metric is not None:
            metric.inc()

    def observe_decision(
        self,
        decision: str,
        risk_level: str,
    ) -> None:
        risk_decisions_total.labels(
            decision=decision
        ).inc()

        risk_levels_total.labels(
            level=risk_level
        ).inc()