from abc import ABC, abstractmethod


class ProbabilityCalibrator(ABC):
    """
    Converts a raw model score into a calibrated probability.
    """

    @property
    @abstractmethod
    def calibration_version(self) -> str:
        ...

    @abstractmethod
    def calibrate(self, raw_score: float) -> float:
        """
        Return calibrated probability in [0, 1].
        """
        ...