from pathlib import Path

from catboost import CatBoostClassifier

from risk_engine.contracts import FeatureVector
from risk_engine.inference.model import RiskModel


FEATURE_COLUMNS_V3 = [
    "amount",
    "customer_transactions_1m",
    "customer_transactions_1h",
    "device_transactions_1h",
    "ip_transactions_1h",
    "customer_degree",
    "device_customer_count",
    "ip_customer_count",
    "payment_customer_count",
    "merchant_transaction_count",
]

FEATURE_COLUMNS_V5 = [
    "amount",
    "customer_transactions_1m",
    "customer_transactions_1h",
    "device_transactions_1h",
    "ip_transactions_1h",
    "customer_degree",
    "device_customer_count",
    "ip_customer_count",
]


class CatBoostRiskModel(RiskModel):
    """
    Production CatBoost fraud model.

    The loaded CatBoost artifact defines the exact feature schema.
    Supported production schemas are versioned explicitly.
    """

    SUPPORTED_SCHEMAS = {
        tuple(FEATURE_COLUMNS_V3),
        tuple(FEATURE_COLUMNS_V5),
    }

    def __init__(
        self,
        model_path: str | Path,
        version: str = "fraud-online-v5",
    ) -> None:
        self._model = CatBoostClassifier()

        self._model.load_model(str(model_path))

        self._version = version

        actual_features = self._model.feature_names_

        if actual_features is None:
            raise ValueError(
                "CatBoost model does not contain feature names."
            )

        actual_schema = tuple(actual_features)

        if actual_schema not in self.SUPPORTED_SCHEMAS:
            raise ValueError(
                "Unsupported CatBoost feature schema.\n"
                f"Actual: {actual_features}"
            )

        self._feature_columns = actual_features

    @property
    def model_version(self) -> str:
        return self._version

    @property
    def feature_columns(self) -> list[str]:
        return list(self._feature_columns)

    def predict_probability(
        self,
        features: FeatureVector,
    ) -> float:

        values_by_name = {
            "amount": features.amount,
            "customer_transactions_1m": (
                features.customer_transactions_1m
            ),
            "customer_transactions_1h": (
                features.customer_transactions_1h
            ),
            "device_transactions_1h": (
                features.device_transactions_1h
            ),
            "ip_transactions_1h": (
                features.ip_transactions_1h
            ),
            "customer_degree": (
                features.customer_degree
            ),
            "device_customer_count": (
                features.device_customer_count
            ),
            "ip_customer_count": (
                features.ip_customer_count
            ),
            "payment_customer_count": (
                features.payment_customer_count
            ),
            "merchant_transaction_count": (
                features.merchant_transaction_count
            ),
        }

        values = [[
            values_by_name[column]
            for column in self._feature_columns
        ]]

        probability = self._model.predict_proba(
            values
        )[0][1]

        return min(
            max(float(probability), 0.0),
            1.0,
        )