from risk_engine.contracts import RiskDecision

from risk_persistence.postgres.models.risk_decision import (
    RiskDecision as RiskDecisionModel,
)


def risk_decision_from_contract(
    decision: RiskDecision,
) -> RiskDecisionModel:
    return RiskDecisionModel(
        transaction_id=decision.transaction_id,
        fraud_probability=decision.fraud_probability,
        expected_loss=decision.expected_loss,
        risk_level=decision.risk_level.value,
        decision=decision.decision.value,
        reason_codes=list(decision.reason_codes),
        policy_version=decision.policy_version,
        model_version=decision.model_version,
        calibration_version=decision.calibration_version,
    )
