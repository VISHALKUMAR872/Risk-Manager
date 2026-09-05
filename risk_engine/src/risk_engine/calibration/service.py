from risk_engine.contracts import RiskScore
from risk_engine.inference.model import RiskModel
from risk_engine.contracts import FeatureVector

from risk_engine.calibration.calibrator import ProbabilityCalibrator


class CalibrationService:
    def __init__(
        self,
        model: RiskModel,
        calibrator: ProbabilityCalibrator,
    ) -> None:
        self.model = model
        self.calibrator = calibrator

    def predict(self, features: FeatureVector) -> RiskScore:
        raw_score = self.model.predict_probability(features)

        calibrated_probability = self.calibrator.calibrate(
            raw_score
        )

        return RiskScore(
            transaction_id=features.transaction_id,
            fraud_probability=calibrated_probability,
            model_version=self.model.model_version,
            calibration_version=self.calibrator.calibration_version,
        )