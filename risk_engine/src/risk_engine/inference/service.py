import time

from risk_engine.calibration import (
    IdentityCalibrator,
    ProbabilityCalibrator,
)
from risk_engine.contracts import FeatureVector, RiskScore
from risk_engine.inference.model import RiskModel
from risk_engine.observability import (
    NoOpRiskPipelineObserver,
    PipelineStage,
    RiskPipelineObserver,
)


class InferenceService:
    """
    Model inference followed by probability calibration.
    """

    def __init__(
        self,
        model: RiskModel,
        calibrator: ProbabilityCalibrator | None = None,
        observer: RiskPipelineObserver | None = None,
    ) -> None:
        self.model = model
        self.calibrator = calibrator or IdentityCalibrator()
        self.observer = observer or NoOpRiskPipelineObserver()

    def predict(
        self,
        features: FeatureVector,
    ) -> RiskScore:

        # =========================================================
        # 1. Model inference
        # =========================================================

        started_at = time.perf_counter()

        try:
            raw_score = self.model.predict_probability(
                features
            )

        except Exception:
            self.observer.observe_error(
                PipelineStage.MODEL_INFERENCE
            )
            raise

        finally:
            self.observer.observe_latency(
                PipelineStage.MODEL_INFERENCE,
                time.perf_counter() - started_at,
            )

        # =========================================================
        # 2. Probability calibration
        # =========================================================

        started_at = time.perf_counter()

        try:
            calibrated_probability = self.calibrator.calibrate(
                raw_score
            )

        except Exception:
            self.observer.observe_error(
                PipelineStage.CALIBRATION
            )
            raise

        finally:
            self.observer.observe_latency(
                PipelineStage.CALIBRATION,
                time.perf_counter() - started_at,
            )

        return RiskScore(
            transaction_id=features.transaction_id,
            fraud_probability=calibrated_probability,
            model_version=self.model.model_version,
            calibration_version=self.calibrator.calibration_version,
        )