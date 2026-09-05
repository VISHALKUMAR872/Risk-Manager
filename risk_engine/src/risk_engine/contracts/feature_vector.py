from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class FeatureVector(BaseModel):
    """
    Canonical online feature vector consumed by inference.

    This is a model input contract, not a persistence model.
    """

    model_config = ConfigDict(extra="forbid")

    feature_version: str = "online-v1"

    transaction_id: str = Field(min_length=1)
    as_of_time: datetime

    # Transaction-native features
    amount: float = Field(ge=0)
    currency: str = Field(min_length=3, max_length=3)

    # Redis velocity features
    customer_transactions_1m: int = Field(ge=0)
    customer_transactions_1h: int = Field(ge=0)
    device_transactions_1h: int = Field(ge=0)
    ip_transactions_1h: int = Field(ge=0)

    # Neo4j relationship features
    customer_degree: int = Field(ge=0)
    device_customer_count: int = Field(ge=0)
    ip_customer_count: int = Field(ge=0)
    payment_customer_count: int = Field(ge=0)
    merchant_transaction_count: int = Field(ge=0)