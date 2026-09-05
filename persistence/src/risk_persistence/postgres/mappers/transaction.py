from risk_engine.contracts.transaction import TransactionEvent

from risk_persistence.postgres.models.transaction import Transaction


def transaction_from_event(
    event: TransactionEvent,
) -> Transaction:
    return Transaction(
        event_id=event.event_id,
        transaction_id=event.transaction_id,
        event_time=event.event_time,
        customer_id=event.customer_id,
        merchant_id=event.merchant_id,
        amount=event.amount,
        currency=event.currency,
        device_id=event.device_id,
        ip_address=event.ip_address,
        payment_method_id=event.payment_method_id,
        merchant_category=event.merchant_category,
        country=event.country,
        channel=event.channel,
        status="RECEIVED",
        raw_payload=event.model_dump(mode="json"),
    )