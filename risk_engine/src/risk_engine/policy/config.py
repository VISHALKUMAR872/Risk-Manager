from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyConfig:
    """
    Risk Sentinel V6.1 Balanced production policy.

    Thresholds were selected exclusively on the disjoint temporal
    calibration/selection partition and locked before future-test evaluation.
    """

    critical_probability: float = 0.640
    critical_expected_loss: float = 700.0

    high_probability: float = 0.560
    high_expected_loss: float = 400.0

    medium_probability: float = 0.275
    medium_expected_loss: float = 50.0
