from dataclasses import dataclass
from typing import Final


FEATURE_VERSION: Final[str] = "online-v2"


@dataclass(frozen=True)
class OnlineFeatureDefinition:
    name: str
    semantic_definition: str
    source: str
    window_seconds: int | None
    point_in_time: bool = True


ONLINE_FEATURES: Final[tuple[OnlineFeatureDefinition, ...]] = (
    OnlineFeatureDefinition(
        name="amount",
        semantic_definition="Transaction amount.",
        source="transaction_event",
        window_seconds=None,
    ),
    OnlineFeatureDefinition(
        name="customer_transactions_1m",
        semantic_definition=(
            "Number of prior transactions for the customer "
            "within the preceding 1 minute."
        ),
        source="redis",
        window_seconds=60,
    ),
    OnlineFeatureDefinition(
        name="customer_transactions_1h",
        semantic_definition=(
            "Number of prior transactions for the customer "
            "within the preceding 1 hour."
        ),
        source="redis",
        window_seconds=3600,
    ),
    OnlineFeatureDefinition(
        name="device_transactions_1h",
        semantic_definition=(
            "Number of prior transactions for the device "
            "within the preceding 1 hour."
        ),
        source="redis",
        window_seconds=3600,
    ),
    OnlineFeatureDefinition(
        name="ip_transactions_1h",
        semantic_definition=(
            "Number of prior transactions for the IP address "
            "within the preceding 1 hour."
        ),
        source="redis",
        window_seconds=3600,
    ),
    OnlineFeatureDefinition(
        name="customer_degree",
        semantic_definition=(
            "Number of historical graph relationships for the customer "
            "before the current transaction."
        ),
        source="neo4j",
        window_seconds=None,
    ),
    OnlineFeatureDefinition(
        name="device_customer_count",
        semantic_definition=(
            "Distinct customers historically associated with the device "
            "before the current transaction."
        ),
        source="neo4j",
        window_seconds=None,
    ),
    OnlineFeatureDefinition(
        name="ip_customer_count",
        semantic_definition=(
            "Distinct customers historically associated with the IP "
            "before the current transaction."
        ),
        source="neo4j",
        window_seconds=None,
    ),
    OnlineFeatureDefinition(
        name="payment_customer_count",
        semantic_definition=(
            "Distinct customers historically associated with the payment "
            "method before the current transaction."
        ),
        source="neo4j",
        window_seconds=None,
    ),
    OnlineFeatureDefinition(
        name="merchant_transaction_count",
        semantic_definition=(
            "Number of prior transactions associated with the merchant "
            "before the current transaction."
        ),
        source="neo4j",
        window_seconds=None,
    ),
)