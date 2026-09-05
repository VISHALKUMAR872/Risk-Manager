from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Decision(StrEnum):
    APPROVE = "APPROVE"
    VERIFY = "VERIFY"
    REVIEW = "REVIEW"
    HOLD = "HOLD"


class RiskLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RiskDecision(BaseModel):
    """
    Final risk decision produced by the policy layer.

    This is a business decision, not a model output.
    """

    model_config = ConfigDict(extra="forbid")

    transaction_id: str = Field(min_length=1)

    fraud_probability: float = Field(ge=0.0, le=1.0)

    expected_loss: float = Field(ge=0.0)

    risk_level: RiskLevel

    decision: Decision

    reason_codes: list[str] = Field(default_factory=list)

    policy_version: str = Field(min_length=1)

    model_version: str = Field(min_length=1)

    calibration_version: str = Field(min_length=1)