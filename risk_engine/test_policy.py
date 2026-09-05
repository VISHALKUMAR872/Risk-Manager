from risk_engine.contracts import (
    RiskScore,
    RiskLevel,
    Decision,
)
from risk_engine.expected_loss import ExpectedLossCalculator
from risk_engine.policy import PolicyEngine


calculator = ExpectedLossCalculator(loss_given_fraud=0.80)
policy = PolicyEngine()


cases = [
    ("LOW", 0.10, 50),
    ("MEDIUM", 0.40, 100),
    ("HIGH", 0.70, 500),
    ("CRITICAL", 0.90, 1000),
]


for name, probability, amount in cases:
    risk_score = RiskScore(
        transaction_id=f"txn-{name.lower()}",
        fraud_probability=probability,
        model_version="development-v1",
        calibration_version="not-calibrated-v1",
    )

    expected_loss = calculator.calculate(
        risk_score=risk_score,
        exposure_amount=amount,
        currency="INR",
    )

    decision = policy.decide(
        risk_score=risk_score,
        expected_loss=expected_loss,
    )

    print(
        f"{name:8} | "
        f"probability={probability:.2f} | "
        f"amount={amount:.2f} | "
        f"loss={expected_loss.expected_loss:.2f} | "
        f"level={decision.risk_level} | "
        f"decision={decision.decision} | "
        f"reasons={decision.reason_codes}"
    )
