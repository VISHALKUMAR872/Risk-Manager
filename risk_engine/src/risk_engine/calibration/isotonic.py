from pathlib import Path

import joblib

from risk_engine.calibration.calibrator import (
    ProbabilityCalibrator,
)


class IsotonicCalibrator(ProbabilityCalibrator):
    """
    Production isotonic probability calibrator.
    """

    def __init__(
        self,
        calibrator_path: str | Path,
        version: str = "isotonic-online-v5",
    ) -> None:
        self._calibrator = joblib.load(
            calibrator_path
        )

        self._version = version

    @property
    def calibration_version(self) -> str:
        return self._version

    def calibrate(
        self,
        raw_score: float,
    ) -> float:
        raw_score = min(
            max(float(raw_score), 0.0),
            1.0,
        )

        calibrated = self._calibrator.predict(
            [raw_score]
        )[0]

        return min(
            max(float(calibrated), 0.0),
            1.0,
        )
