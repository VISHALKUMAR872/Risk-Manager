from risk_engine.contracts import FeatureVector
from risk_engine.inference.model import RiskModel


class DevelopmentRiskModel(RiskModel):
    """
    Temporary deterministic model used to validate
    the production inference pipeline before a trained
    model artifact is available.

    This is NOT the production fraud model.
    """

    @property
    def model_version(self) -> str:
        return "development-v1"

    def predict_probability(self, features: FeatureVector) -> float:
        score = 0.0

        score += min(features.customer_transactions_1m / 10.0, 0.20)
        score += min(features.customer_transactions_1h / 50.0, 0.10)
        score += min(features.device_transactions_1h / 50.0, 0.10)
        score += min(features.ip_transactions_1h / 50.0, 0.10)

        score += min(features.device_customer_count / 10.0, 0.15)
        score += min(features.ip_customer_count / 10.0, 0.15)
        score += min(features.payment_customer_count / 10.0, 0.10)
        score += min(features.merchant_transaction_count / 100.0, 0.10)

        return min(max(score, 0.0), 1.0)