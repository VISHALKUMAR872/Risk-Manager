from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict


class TransactionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    transaction_id: str
    event_id: str
    event_time: datetime
    customer_id: str
    merchant_id: str
    amount: Decimal
    currency: str
    device_id: str
    ip_address: str
    payment_method_id: str
    merchant_category: str
    country: str
    channel: str
    status: str
    created_at: datetime


class RiskDecisionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    transaction_id: str
    fraud_probability: float
    expected_loss: float
    risk_level: str
    decision: str
    reason_codes: list[str]
    policy_version: str
    model_version: str
    calibration_version: str
    created_at: datetime


class TransactionRiskResponse(BaseModel):
    transaction: TransactionResponse
    risk_decision: RiskDecisionResponse | None
