from datetime import datetime
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Index, Numeric, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from risk_persistence.postgres.base import Base

class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    event_id: Mapped[str] = mapped_column(
        String(128),
        unique=True,
        nullable=False,
    )

    transaction_id: Mapped[str] = mapped_column(
        String(128),
        unique=True,
        nullable=False,
    )

    event_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    customer_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    merchant_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    amount: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
    )

    currency: Mapped[str] = mapped_column(
        String(3),
        nullable=False,
    )

    device_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    ip_address: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    payment_method_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    merchant_category: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    country: Mapped[str] = mapped_column(
        String(3),
        nullable=False,
    )

    channel: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="RECEIVED",
    )

    raw_payload: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


Index("ix_transactions_event_time", Transaction.event_time)
Index("ix_transactions_customer_id", Transaction.customer_id)
Index("ix_transactions_merchant_id", Transaction.merchant_id)
Index("ix_transactions_device_id", Transaction.device_id)
Index(
    "ix_transactions_payment_method_id",
    Transaction.payment_method_id,
)