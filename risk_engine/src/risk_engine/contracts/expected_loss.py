from pydantic import BaseModel, ConfigDict, Field


class ExpectedLoss(BaseModel):
    """
    Financial risk estimate for a transaction.

    Expected loss is an economic quantity and must not
    directly determine the final business decision.
    """

    model_config = ConfigDict(extra="forbid")

    transaction_id: str = Field(min_length=1)

    fraud_probability: float = Field(ge=0.0, le=1.0)

    exposure_amount: float = Field(ge=0.0)

    loss_given_fraud: float = Field(ge=0.0, le=1.0)

    expected_loss: float = Field(ge=0.0)

    currency: str = Field(min_length=3, max_length=3)

    expected_loss_version: str = Field(min_length=1)