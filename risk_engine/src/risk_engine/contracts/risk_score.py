from pydantic import BaseModel, ConfigDict, Field


class RiskScore(BaseModel):
    """
    Canonical calibrated risk output.

    This represents the probability that the transaction is fraudulent.
    It does not decide the business action.
    """

    model_config = ConfigDict(extra="forbid")

    transaction_id: str = Field(min_length=1)

    fraud_probability: float = Field(ge=0.0, le=1.0)

    model_version: str = Field(min_length=1)

    calibration_version: str = Field(min_length=1)