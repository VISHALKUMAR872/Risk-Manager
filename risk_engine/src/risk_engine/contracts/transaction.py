from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class TransactionEvent(BaseModel):
    """
    Canonical transaction event shared by all services.

    This represents facts received from the transaction source.
    It must not contain derived ML features or risk decisions.
    """

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1)
    transaction_id: str = Field(min_length=1)
    event_time: datetime

    customer_id: str = Field(min_length=1)
    merchant_id: str = Field(min_length=1)

    amount: Decimal = Field(gt=0)
    currency: str = Field(min_length=3, max_length=3)

    device_id: str = Field(min_length=1)
    ip_address: str = Field(min_length=1)
    payment_method_id: str = Field(min_length=1)

    merchant_category: str = Field(min_length=1)
    country: str = Field(min_length=2, max_length=3)
    channel: str = Field(min_length=1)