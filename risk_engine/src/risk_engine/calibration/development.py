from risk_engine.calibration.calibrator import ProbabilityCalibrator


class IdentityCalibrator(ProbabilityCalibrator):
    """
    Temporary calibration implementation.

    It deliberately performs no transformation.
    This exists only to establish the production pipeline
    before a fitted calibration artifact is available.
    """

    @property
    def calibration_version(self) -> str:
        return "identity-v1"

    def calibrate(self, raw_score: float) -> float:
        return min(max(raw_score, 0.0), 1.0)