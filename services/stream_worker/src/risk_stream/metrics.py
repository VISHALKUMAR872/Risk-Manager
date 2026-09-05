from prometheus_client import Counter, Histogram


# ---------------------------------------------------------------------------
# Transaction processing
# ---------------------------------------------------------------------------

transactions_processed_total = Counter(
    "risk_transactions_processed_total",
    "Total number of transactions successfully processed.",
)

transactions_failed_total = Counter(
    "risk_transactions_failed_total",
    "Total number of transactions whose risk processing failed.",
)


# ---------------------------------------------------------------------------
# Kafka consumer
# ---------------------------------------------------------------------------

consumer_poll_errors_total = Counter(
    "risk_consumer_poll_errors_total",
    "Total number of Kafka consumer errors.",
)

consumer_commit_errors_total = Counter(
    "risk_consumer_commit_errors_total",
    "Total number of Kafka offset commit failures.",
)


# ---------------------------------------------------------------------------
# DLQ
# ---------------------------------------------------------------------------

dlq_messages_total = Counter(
    "risk_dlq_messages_total",
    "Total number of poison messages successfully published to the DLQ.",
)


# ---------------------------------------------------------------------------
# Risk decisions
#
# Labels are deliberately low-cardinality.
#
# NEVER use:
#   - transaction_id
#   - customer_id
#   - event_id
#   - IP address
#   - device_id
#   - payment_method_id
#   - merchant_id
#
# as Prometheus labels.
# ---------------------------------------------------------------------------

risk_decisions_total = Counter(
    "risk_decisions_total",
    "Total number of risk decisions produced.",
    ["decision"],
)

risk_levels_total = Counter(
    "risk_levels_total",
    "Total number of risk levels produced.",
    ["level"],
)


# ---------------------------------------------------------------------------
# Risk pipeline latency
# ---------------------------------------------------------------------------

risk_decision_latency_seconds = Histogram(
    "risk_decision_latency_seconds",
    "Time spent processing a transaction through the risk processor.",
    buckets=(
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
    ),
)

risk_evaluation_latency_seconds = Histogram(
    "risk_evaluation_latency_seconds",
    "Time spent evaluating transaction risk through features, inference, calibration, expected loss, and policy.",
    buckets=(
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
    ),
)

risk_postgres_latency_seconds = Histogram(
    "risk_postgres_latency_seconds",
    "Time spent performing PostgreSQL persistence operations.",
    buckets=(
        0.001,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
    ),
)

risk_redis_latency_seconds = Histogram(
    "risk_redis_latency_seconds",
    "Time spent updating Redis velocity state.",
    buckets=(
        0.001,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
    ),
)

risk_neo4j_latency_seconds = Histogram(
    "risk_neo4j_latency_seconds",
    "Time spent projecting transactions into Neo4j.",
    buckets=(
        0.001,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
    ),
)


# ---------------------------------------------------------------------------
# Dependency errors
# ---------------------------------------------------------------------------

risk_postgres_errors_total = Counter(
    "risk_postgres_errors_total",
    "Total number of PostgreSQL operation failures.",
)

risk_redis_errors_total = Counter(
    "risk_redis_errors_total",
    "Total number of Redis operation failures.",
)

risk_neo4j_errors_total = Counter(
    "risk_neo4j_errors_total",
    "Total number of Neo4j operation failures.",
)
risk_feature_build_latency_seconds = Histogram(
    "risk_feature_build_latency_seconds",
    "Time spent building the complete online feature vector.",
)

risk_feature_build_errors_total = Counter(
    "risk_feature_build_errors_total",
    "Total online feature-vector construction failures.",
)

risk_model_inference_latency_seconds = Histogram(
    "risk_model_inference_latency_seconds",
    "Time spent running the fraud risk model.",
)

risk_model_inference_errors_total = Counter(
    "risk_model_inference_errors_total",
    "Total fraud model inference failures.",
)

risk_calibration_latency_seconds = Histogram(
    "risk_calibration_latency_seconds",
    "Time spent calibrating the raw fraud probability.",
)

risk_calibration_errors_total = Counter(
    "risk_calibration_errors_total",
    "Total probability calibration failures.",
)

risk_expected_loss_latency_seconds = Histogram(
    "risk_expected_loss_latency_seconds",
    "Time spent calculating expected monetary loss.",
)

risk_expected_loss_errors_total = Counter(
    "risk_expected_loss_errors_total",
    "Total expected-loss calculation failures.",
)

risk_policy_latency_seconds = Histogram(
    "risk_policy_latency_seconds",
    "Time spent evaluating the deterministic risk policy.",
)

risk_policy_errors_total = Counter(
    "risk_policy_errors_total",
    "Total risk policy evaluation failures.",
)

# ---------------------------------------------------------------------------
# Risk evaluation errors
#
# This is intentionally NOT called risk_model_errors_total.
#
# TransactionProcessor currently wraps the complete
# DecisionService.evaluate() call, which includes:
#
#   feature retrieval
#   model inference
#   calibration
#   expected loss
#   policy evaluation
#
# Therefore this metric represents the whole evaluation stage.
# ---------------------------------------------------------------------------

risk_evaluation_errors_total = Counter(
    "risk_evaluation_errors_total",
    "Total number of failures during risk evaluation.",
)

# ---------------------------------------------------------------------------
# Retry / failure handling
# ---------------------------------------------------------------------------

transaction_retries_total = Counter(
    "risk_transaction_retries_total",
    "Total number of transaction processing retry attempts.",
)

transaction_retry_exhausted_total = Counter(
    "risk_transaction_retry_exhausted_total",
    "Total number of transactions that exhausted processing retries.",
)