from risk_engine.contracts import FeatureVector, TransactionEvent
from risk_features.online.service import OnlineFeatureService


class OnlineFeatureProvider:
    """
    Adapter between OnlineFeatureService and the risk-engine
    FeatureProvider contract.
    """

    def __init__(self, feature_service: OnlineFeatureService) -> None:
        self.feature_service = feature_service

    async def build(
        self,
        event: TransactionEvent,
    ) -> FeatureVector:
        return await self.feature_service.build(event)
