from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    DateTime,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from risk_persistence.postgres.base import Base


class OutboxEvent(Base):
    __tablename__ = "outbox_events"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    event_type: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    aggregate_type: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    aggregate_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    topic: Mapped[str] = mapped_column(
        String(256),
        nullable=False,
    )

    message_key: Mapped[str] = mapped_column(
        String(256),
        nullable=False,
    )

    payload: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="PENDING",
    )

    attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    last_error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    next_attempt_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


Index(
    "ix_outbox_events_status_next_attempt",
    OutboxEvent.status,
    OutboxEvent.next_attempt_at,
)

Index(
    "ix_outbox_events_aggregate",
    OutboxEvent.aggregate_type,
    OutboxEvent.aggregate_id,
)