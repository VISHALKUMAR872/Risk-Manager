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


def evaluate(
    model: CatBoostClassifier,
    data: pd.DataFrame,
    name: str,
) -> None:
    X = data[FEATURE_COLUMNS]
    y = data[TARGET]

    probabilities = model.predict_proba(X)[:, 1]

    roc_auc = roc_auc_score(
        y,
        probabilities,
    )

    pr_auc = average_precision_score(
        y,
        probabilities,
    )

    print("\n" + "-" * 80)
    print(f"{name} METRICS")
    print("-" * 80)

    print(f"ROC-AUC: {roc_auc:.6f}")
    print(f"PR-AUC:  {pr_auc:.6f}")

    predictions = (
        probabilities >= 0.50
    ).astype("int8")

    print("\nClassification report @ 0.50:")
    print(
        classification_report(
            y,
            predictions,
            digits=4,
        )
    )


def main() -> None:
    train_path = ARTIFACT_DIR / "online_v4_train.parquet"
    calibration_path = (
        ARTIFACT_DIR / "online_v4_calibration.parquet"
    )
    test_path = ARTIFACT_DIR / "online_v4_test.parquet"

    model_path = (
        MODEL_DIR / "fraud_online_v4_catboost.cbm"
    )

    print("=" * 80)
    print("RISK SENTINEL — CATBOOST V4 TRAINING")
    print("=" * 80)

    print("\nLoading datasets...")

    train = pd.read_parquet(train_path)
    calibration = pd.read_parquet(calibration_path)
    test = pd.read_parquet(test_path)

    print(f"Train rows:       {len(train):,}")
    print(f"Calibration rows: {len(calibration):,}")
    print(f"Future test rows: {len(test):,}")

    # ------------------------------------------------------------------
    # Validate feature contract.
    # ------------------------------------------------------------------

    for name, data in (
        ("train", train),
        ("calibration", calibration),
        ("test", test),
    ):
        missing = [
            column
            for column in FEATURE_COLUMNS
            if column not in data.columns
        ]

        if missing:
            raise RuntimeError(
                f"{name} is missing features: {missing}"
            )

        if data[FEATURE_COLUMNS].isna().any().any():
            raise RuntimeError(
                f"{name} contains NaN production features."
            )

    # ------------------------------------------------------------------
    # Internal chronological validation.
    #
    # This validation period belongs entirely inside the training
    # period and is used only for CatBoost early stopping.
    # ------------------------------------------------------------------

    internal_split = int(len(train) * 0.90)

    fit_data = train.iloc[:internal_split].copy()
    early_stop_data = train.iloc[internal_split:].copy()

    X_fit = fit_data[FEATURE_COLUMNS]
    y_fit = fit_data[TARGET]

    X_early = early_stop_data[FEATURE_COLUMNS]
    y_early = early_stop_data[TARGET]

    print("\n" + "-" * 80)
    print("INTERNAL CATBOOST SPLIT")
    print("-" * 80)

    print(f"Fit rows:          {len(fit_data):,}")
    print(f"Early-stop rows:   {len(early_stop_data):,}")

    print(
        f"Fit time: "
        f"{fit_data.TransactionDT.min():,} -> "
        f"{fit_data.TransactionDT.max():,}"
    )

    print(
        f"Early-stop time: "
        f"{early_stop_data.TransactionDT.min():,} -> "
        f"{early_stop_data.TransactionDT.max():,}"
    )

    # ------------------------------------------------------------------
    # CatBoost.
    # ------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("TRAINING CATBOOST V4")
    print("-" * 80)

    model = CatBoostClassifier(
        iterations=1500,
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
        od_wait=120,
        verbose=100,
        allow_writing_files=False,
    )

    model.fit(
        X_fit,
        y_fit,
        eval_set=(X_early, y_early),
        verbose=100,
    )

    best_iteration = model.get_best_iteration()

    print(
        f"\nBest iteration: {best_iteration}"
    )

    # ------------------------------------------------------------------
    # Evaluation.
    # ------------------------------------------------------------------

    evaluate(
        model,
        early_stop_data,
        "INTERNAL EARLY-STOP",
    )

    evaluate(
        model,
        calibration,
        "CALIBRATION RAW MODEL",
    )

    evaluate(
        model,
        test,
        "FUTURE TEST RAW MODEL",
    )

    # ------------------------------------------------------------------
    # Save.
    # ------------------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_model(model_path)

    print(
        f"\nModel saved: {model_path}"
    )

    print("\n" + "=" * 80)
    print("CATBOOST V4 TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
