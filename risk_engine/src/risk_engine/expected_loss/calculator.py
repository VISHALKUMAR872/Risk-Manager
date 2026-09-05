from risk_engine.contracts import ExpectedLoss, RiskScore


class ExpectedLossCalculator:
    """
    Calculates expected monetary loss from calibrated fraud risk.

    Formula:

        expected_loss =
            fraud_probability
            * exposure_amount
            * loss_given_fraud
    """

    def __init__(
        self,
        loss_given_fraud: float = 0.80,
    ) -> None:
        if not 0.0 <= loss_given_fraud <= 1.0:
            raise ValueError("loss_given_fraud must be between 0 and 1")

        self.loss_given_fraud = loss_given_fraud

    def calculate(
        self,
        risk_score: RiskScore,
        exposure_amount: float,
        currency: str,
    ) -> ExpectedLoss:
        if exposure_amount < 0:
            raise ValueError("exposure_amount cannot be negative")

        expected_loss = (
            risk_score.fraud_probability
            * exposure_amount
            * self.loss_given_fraud
        )

        return ExpectedLoss(
            transaction_id=risk_score.transaction_id,
            fraud_probability=risk_score.fraud_probability,
            exposure_amount=exposure_amount,
            loss_given_fraud=self.loss_given_fraud,
            expected_loss=expected_loss,
            currency=currency,
            expected_loss_version="expected-loss-v1",
        )