from pathlib import Path

import pandas as pd

from catboost import CatBoostClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)


ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "models"

TARGET = "isFraud"

FEATURE_COLUMNS = [
    "amount",
    "customer_transactions_1m",
    "customer_transactions_1h",
    "device_transactions_1h",
    "ip_transactions_1h",
    "customer_degree",
    "device_customer_count",
    "ip_customer_count",
    "payment_customer_count",
    "merchant_transaction_count",
]


def main() -> None:
    train_path = ARTIFACT_DIR / "online_v3_train.parquet"
    valid_path = ARTIFACT_DIR / "online_v3_valid.parquet"

    model_path = (
        MODEL_DIR
        / "fraud_online_v3_catboost.cbm"
    )

    print("Loading training data...")
    train = pd.read_parquet(train_path)

    print("Loading validation data...")
    valid = pd.read_parquet(valid_path)

    missing_train = [
        column
        for column in FEATURE_COLUMNS
        if column not in train.columns
    ]

    missing_valid = [
        column
        for column in FEATURE_COLUMNS
        if column not in valid.columns
    ]

    if missing_train or missing_valid:
        raise ValueError(
            f"Missing training features: {missing_train}\n"
            f"Missing validation features: {missing_valid}"
        )

    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET]

    X_valid = valid[FEATURE_COLUMNS]
    y_valid = valid[TARGET]

    print("=" * 80)
    print("PRODUCTION-CONTRACT CATBOOST TRAINING")
    print("=" * 80)

    print(f"Train rows: {len(train):,}")
    print(f"Valid rows: {len(valid):,}")
    print(f"Features: {len(FEATURE_COLUMNS)}")

    print("\nFeatures:")

    for column in FEATURE_COLUMNS:
        print(f"  {column}")

    print("\nFraud rate:")
    print(
        f"Train: {y_train.mean() * 100:.4f}%"
    )
    print(
        f"Valid: {y_valid.mean() * 100:.4f}%"
    )

    print("\nTraining...")

    model = CatBoostClassifier(
        iterations=1000,
        depth=8,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=42,
        l2_leaf_reg=5,
        random_strength=1,
        border_count=128,
        od_type="Iter",
        od_wait=100,
        verbose=100,
        allow_writing_files=False,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_valid, y_valid),
        verbose=100,
    )

    print("\nTraining complete.")

    print(
        f"Best iteration: "
        f"{model.get_best_iteration()}"
    )

    print("\nGenerating validation probabilities...")

    probabilities = model.predict_proba(
        X_valid
    )[:, 1]

    roc_auc = roc_auc_score(
        y_valid,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_valid,
        probabilities,
    )

    print("\n" + "=" * 80)
    print("VALIDATION METRICS")
    print("=" * 80)

    print(f"ROC-AUC: {roc_auc:.6f}")
    print(f"PR-AUC:  {pr_auc:.6f}")

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    print("\nClassification report @ threshold 0.50:")

    print(
        classification_report(
            y_valid,
            predictions,
            digits=4,
        )
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_model(model_path)

    print(
        f"\nModel saved: {model_path}"
    )


if __name__ == "__main__":
    main()
