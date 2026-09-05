from typing import Protocol

from risk_engine.contracts import FeatureVector, TransactionEvent


class FeatureProvider(Protocol):
    """
    Provides the canonical feature vector required for risk inference.

    Implementations live outside the domain package.
    """

    async def build(
        self,
        event: TransactionEvent,
    ) -> FeatureVector:
        from typing import Protocol

from risk_engine.contracts import FeatureVector, TransactionEvent


class FeatureProvider(Protocol):
    """
    Provides the canonical feature vector required for risk inference.

    Implementations live outside the domain package.
    """

    async def build(
        self,
        event: TransactionEvent,
    ) -> FeatureVector:
        ...