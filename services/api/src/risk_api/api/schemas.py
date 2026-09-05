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
    payment_method: str

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


class DashboardTransactionResponse(BaseModel):
    transaction_id: str
    event_id: str
    event_time: datetime

    customer_id: str
    merchant_id: str

    amount: Decimal
    currency: str

    device_id: str
    ip_address: str
    payment_method: str

    merchant_category: str
    country: str
    channel: str

    status: str
    created_at: datetime

    fraud_probability: float | None = None
    expected_loss: float | None = None

    risk_level: str | None = None
    decision: str | None = None

    reason_codes: list[str] = []

    policy_version: str | None = None
    model_version: str | None = None
    calibration_version: str | None = None

class DashboardSummaryResponse(BaseModel):
    transaction_count: int
    risk_decided_count: int
    pending_count: int
    failed_count: int

    intervention_count: int
    expected_loss: float
    high_risk_count: int

    decisions: dict[str, int]
    risk_levels: dict[str, int]

    average_transaction_amount: float

class NetworkNode(BaseModel):
    id: str
    type: str
    label: str

    selected: bool = False

    amount: float | None = None
    currency: str | None = None
    event_time: str | None = None


class NetworkEdge(BaseModel):
    source: str
    target: str
    type: str


class TransactionNetworkResponse(BaseModel):
    transaction_id: str
    nodes: list[NetworkNode]
    edges: list[NetworkEdge]
