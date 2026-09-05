from risk_engine.contracts import (
    Decision,
    ExpectedLoss,
    RiskDecision,
    RiskLevel,
    RiskScore,
)

from risk_engine.policy.config import PolicyConfig


class PolicyEngine:
    """
    Deterministic business policy layer.

    ML estimates fraud probability.
    Expected-loss estimates financial exposure.
    Policy determines the operational action.
    """

    VERSION = "policy-v6-balanced"

    def __init__(
        self,
        config: PolicyConfig | None = None,
    ) -> None:
        self.config = config or PolicyConfig()

    def decide(
        self,
        risk_score: RiskScore,
        expected_loss: ExpectedLoss,
    ) -> RiskDecision:

        probability = risk_score.fraud_probability
        loss = expected_loss.expected_loss

        if (
            probability >= self.config.critical_probability
            or loss >= self.config.critical_expected_loss
        ):
            decision = Decision.HOLD
            risk_level = RiskLevel.CRITICAL

        elif (
            probability >= self.config.high_probability
            or loss >= self.config.high_expected_loss
        ):
            decision = Decision.REVIEW
            risk_level = RiskLevel.HIGH

        elif (
            probability >= self.config.medium_probability
            or loss >= self.config.medium_expected_loss
        ):
            decision = Decision.VERIFY
            risk_level = RiskLevel.MEDIUM

        else:
            decision = Decision.APPROVE
            risk_level = RiskLevel.LOW

        return RiskDecision(
            transaction_id=risk_score.transaction_id,
            fraud_probability=probability,
            expected_loss=loss,
            risk_level=risk_level,
            decision=decision,
            reason_codes=self._reason_codes(
                probability=probability,
                expected_loss=loss,
                decision=decision,
            ),
            policy_version=self.VERSION,
            model_version=risk_score.model_version,
            calibration_version=risk_score.calibration_version,
        )

    def _reason_codes(
        self,
        probability: float,
        expected_loss: float,
        decision: Decision,
    ) -> list[str]:

        reasons: list[str] = []

        if probability >= self.config.critical_probability:
            reasons.append("VERY_HIGH_FRAUD_PROBABILITY")

        elif probability >= self.config.high_probability:
            reasons.append("HIGH_FRAUD_PROBABILITY")

        elif probability >= self.config.medium_probability:
            reasons.append("ELEVATED_FRAUD_PROBABILITY")

        if expected_loss >= self.config.critical_expected_loss:
            reasons.append("CRITICAL_EXPECTED_LOSS")

        elif expected_loss >= self.config.high_expected_loss:
            reasons.append("HIGH_EXPECTED_LOSS")

        elif expected_loss >= self.config.medium_expected_loss:
            reasons.append("ELEVATED_EXPECTED_LOSS")

        if decision == Decision.APPROVE:
            reasons.append("LOW_RISK")

        return reasons or ["POLICY_THRESHOLD_TRIGGERED"]

