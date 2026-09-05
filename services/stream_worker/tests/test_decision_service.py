import asyncio
from datetime import datetime, timezone

from risk_engine.contracts import FeatureVector, TransactionEvent
from risk_engine.decision import DecisionService
from risk_engine.expected_loss import ExpectedLossCalculator
from risk_engine.inference import (
    DevelopmentRiskModel,
    InferenceService,
)
from risk_engine.policy import PolicyEngine


class FakeFeatureProvider:
    """
    Test-only feature provider.

    This allows us to test the complete decision pipeline
    without depending on Redis or Neo4j.
    """

    async def build(
        self,
        event: TransactionEvent,
    ) -> FeatureVector:

        return FeatureVector(
            transaction_id=event.transaction_id,
            as_of_time=datetime.now(timezone.utc),
            amount=float(event.amount),
            currency=event.currency,

            customer_transactions_1m=5,
            customer_transactions_1h=10,
            device_transactions_1h=2,
            ip_transactions_1h=2,

            customer_degree=4,
            device_customer_count=2,
            ip_customer_count=2,
            payment_customer_count=1,
            merchant_transaction_count=5,
        )


async def main() -> None:

    event = TransactionEvent(
        event_id="evt-decision-test",
        transaction_id="txn-decision-test",
        event_time=datetime.now(timezone.utc),

        customer_id="cust-test",
        merchant_id="merchant-test",

        amount="1499.00",
        currency="INR",

        device_id="device-test",
        ip_address="10.0.0.1",
        payment_method_id="payment-test",

        merchant_category="electronics",
        country="IN",
        channel="web",
    )

    service = DecisionService(
        feature_provider=FakeFeatureProvider(),

        inference_service=InferenceService(
            DevelopmentRiskModel()
        ),

        expected_loss_calculator=ExpectedLossCalculator(
            loss_given_fraud=0.80
        ),

        policy_engine=PolicyEngine(),
    )

    decision = await service.evaluate(event)

    print(decision.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())