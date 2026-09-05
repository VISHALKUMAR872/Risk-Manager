import asyncio
import time

from prometheus_client import start_http_server

from risk_stream.consumers.transaction_consumer import (
    TransactionConsumer,
)
from risk_stream.metrics import (
    risk_decision_latency_seconds,
    transaction_retry_exhausted_total,
    transaction_retries_total,
    transactions_failed_total,
    transactions_processed_total,
)
from risk_stream.outbox_publisher import (
    OutboxPublisher,
)
from risk_stream.processors.transaction_processor import (
    TransactionProcessor,
)


METRICS_PORT = 9000
MAX_PROCESSING_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = (1, 2)


async def transaction_consumer_loop(
    consumer: TransactionConsumer,
    processor: TransactionProcessor,
) -> None:
    while True:
        received = consumer.consume_one(timeout=1.0)

        if received is None:
            await asyncio.sleep(0)
            continue

        attempt = 0
        processing_succeeded = False
        last_exception: Exception | None = None

        while attempt < MAX_PROCESSING_ATTEMPTS:
            attempt += 1

            started_at = time.perf_counter()

            try:
                print(
                    "Processing transaction: "
                    f"{received.event.transaction_id} "
                    f"attempt={attempt}/{MAX_PROCESSING_ATTEMPTS}"
                )

                await processor.process(received.event)

                processing_succeeded = True
                transactions_processed_total.inc()

                print(
                    "Transaction processing succeeded: "
                    f"{received.event.transaction_id} "
                    f"attempt={attempt}"
                )

                break

            except Exception as exc:
                last_exception = exc
                transactions_failed_total.inc()

                print(
                    "Transaction processing failed: "
                    f"{received.event.transaction_id} "
                    f"attempt={attempt}/"
                    f"{MAX_PROCESSING_ATTEMPTS}: {exc}"
                )

                if attempt < MAX_PROCESSING_ATTEMPTS:
                    transaction_retries_total.inc()

                    backoff = RETRY_BACKOFF_SECONDS[attempt - 1]

                    print(
                        "Retrying transaction after "
                        f"{backoff}s: "
                        f"{received.event.transaction_id}"
                    )

                    await asyncio.sleep(backoff)

            finally:
                risk_decision_latency_seconds.observe(
                    time.perf_counter() - started_at
                )

        if processing_succeeded:
            consumer.commit(received.message)

        else:
            transaction_retry_exhausted_total.inc()

            print(
                "Retry limit exhausted; publishing transaction to DLQ: "
                f"{received.event.transaction_id}"
            )

            try:
                consumer.publish_to_dlq(
                    message=received.message,
                    error_type=(
                        type(last_exception).__name__
                        if last_exception
                        else "UnknownError"
                    ),
                    error_message=(
                        str(last_exception)
                        if last_exception
                        else "Unknown processing failure"
                    ),
                    attempt_count=MAX_PROCESSING_ATTEMPTS,
                )

                consumer.commit(received.message)

                print(
                    "Committed exhausted transaction after DLQ publication: "
                    f"partition={received.message.partition()} "
                    f"offset={received.message.offset()}"
                )

            except Exception as dlq_exc:
                print(
                    "CRITICAL: failed to publish exhausted transaction to DLQ: "
                    f"{dlq_exc}"
                )


async def main() -> None:
    start_http_server(METRICS_PORT)

    print(
        "Risk Sentinel metrics server started. "
        f"http://localhost:{METRICS_PORT}/metrics"
    )

    consumer = TransactionConsumer()
    processor = TransactionProcessor()
    outbox_publisher = OutboxPublisher()

    consumer.start()

    print("Risk Sentinel Stream Worker is running.")
    print("Transaction consumer started.")
    print("Outbox publisher started.")

    consumer_task = asyncio.create_task(
        transaction_consumer_loop(
            consumer,
            processor,
        )
    )

    outbox_task = asyncio.create_task(
        outbox_publisher.run()
    )

    try:
        await asyncio.gather(
            consumer_task,
            outbox_task,
        )

    except asyncio.CancelledError:
        raise

    except KeyboardInterrupt:
        print("Shutting down stream worker...")

    finally:
        for task in (
            consumer_task,
            outbox_task,
        ):
            task.cancel()

        await asyncio.gather(
            consumer_task,
            outbox_task,
            return_exceptions=True,
        )

        await processor.close()
        outbox_publisher.close()
        consumer.close()


if __name__ == "__main__":
    asyncio.run(main())
