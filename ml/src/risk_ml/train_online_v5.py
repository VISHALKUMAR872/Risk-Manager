from __future__ import annotations

from pathlib import Path
import time

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
MODEL_DIR = ARTIFACTS / "models"

TRAIN_PATH = ARTIFACTS / "online_v4_train.parquet"
CALIBRATION_PATH = ARTIFACTS / "online_v4_calibration.parquet"
TEST_PATH = ARTIFACTS / "online_v4_test.parquet"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "isFraud"

V5_FEATURES = [
    "amount",
    "customer_transactions_1m",
    "customer_transactions_1h",
    "device_transactions_1h",
    "ip_transactions_1h",
    "customer_degree",
    "device_customer_count",
    "ip_customer_count",
]


CATBOOST_PARAMS = {
    "iterations": 1500,
    "depth": 8,
    "learning_rate": 0.05,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "auto_class_weights": "Balanced",
    "random_seed": 42,
    "l2_leaf_reg": 5,
    "random_strength": 1,
    "border_count": 128,
    "od_type": "Iter",
    "od_wait": 120,
    "verbose": 100,
    "allow_writing_files": False,
}


def evaluate(
    y: pd.Series,
    probabilities,
) -> tuple[float, float]:

    roc_auc = roc_auc_score(y, probabilities)
    pr_auc = average_precision_score(y, probabilities)

    return float(roc_auc), float(pr_auc)


def main() -> None:

    print("=" * 88)
    print("RISK SENTINEL — V5 ONLINE MODEL TRAINING")
    print("=" * 88)

    print("\nV5 feature set:")

    for feature in V5_FEATURES:
        print(f"  - {feature}")

    print("\nLoading datasets...")

    train = pd.read_parquet(TRAIN_PATH)
    calibration = pd.read_parquet(CALIBRATION_PATH)
    test = pd.read_parquet(TEST_PATH)

    print(f"Train:       {len(train):,}")
    print(f"Calibration: {len(calibration):,}")
    print(f"Future test: {len(test):,}")

    # -------------------------------------------------------------------------
    # Validate columns
    # -------------------------------------------------------------------------

    required_columns = V5_FEATURES + [TARGET]

    for dataset_name, df in [
        ("train", train),
        ("calibration", calibration),
        ("future_test", test),
    ]:

        missing = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{dataset_name} is missing columns: {missing}"
            )

    # -------------------------------------------------------------------------
    # Fraud rates
    # -------------------------------------------------------------------------

    print("\nFraud rates:")

    print(
        f"  Train:       {train[TARGET].mean():.6%}"
    )

    print(
        f"  Calibration: {calibration[TARGET].mean():.6%}"
    )

    print(
        f"  Future test: {test[TARGET].mean():.6%}"
    )

    # -------------------------------------------------------------------------
    # Chronological internal split
    # -------------------------------------------------------------------------

    split_index = int(len(train) * 0.90)

    fit_df = train.iloc[:split_index]
    early_stop_df = train.iloc[split_index:]

    print("\nInternal training split:")

    print(
        f"  Fit:          {len(fit_df):,}"
    )

    print(
        f"  Early-stop:   {len(early_stop_df):,}"
    )

    # -------------------------------------------------------------------------
    # Prepare matrices
    # -------------------------------------------------------------------------

    X_fit = fit_df[V5_FEATURES]
    y_fit = fit_df[TARGET].astype(int)

    X_eval = early_stop_df[V5_FEATURES]
    y_eval = early_stop_df[TARGET].astype(int)

    X_calibration = calibration[V5_FEATURES]
    y_calibration = calibration[TARGET].astype(int)

    X_test = test[V5_FEATURES]
    y_test = test[TARGET].astype(int)

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------

    print("\n" + "=" * 88)
    print("TRAINING CATBOOST")
    print("=" * 88)

    model = CatBoostClassifier(**CATBOOST_PARAMS)

    start = time.perf_counter()

    model.fit(
        X_fit,
        y_fit,
        eval_set=(X_eval, y_eval),
        use_best_model=True,
    )

    elapsed = time.perf_counter() - start

    # -------------------------------------------------------------------------
    # Predictions
    # -------------------------------------------------------------------------

    print("\nGenerating predictions...")

    p_eval = model.predict_proba(X_eval)[:, 1]
    p_calibration = model.predict_proba(X_calibration)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    internal_roc, internal_pr = evaluate(
        y_eval,
        p_eval,
    )

    calibration_roc, calibration_pr = evaluate(
        y_calibration,
        p_calibration,
    )

    future_roc, future_pr = evaluate(
        y_test,
        p_test,
    )

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------

    print("\n" + "=" * 88)
    print("V5 MODEL RESULTS")
    print("=" * 88)

    print("\nINTERNAL EARLY-STOP")
    print(
        f"ROC-AUC: {internal_roc:.6f}"
    )
    print(
        f"PR-AUC:  {internal_pr:.6f}"
    )

    print("\nCALIBRATION PERIOD — RAW MODEL")
    print(
        f"ROC-AUC: {calibration_roc:.6f}"
    )
    print(
        f"PR-AUC:  {calibration_pr:.6f}"
    )

    print("\nFUTURE TEST — RAW MODEL")
    print(
        f"ROC-AUC: {future_roc:.6f}"
    )
    print(
        f"PR-AUC:  {future_pr:.6f}"
    )

    print("\nMODEL")
    print(
        f"Best iteration: {model.get_best_iteration()}"
    )

    print(
        f"Training time:  {elapsed:.2f}s"
    )

    # -------------------------------------------------------------------------
    # Save model
    # -------------------------------------------------------------------------

    model_path = (
        MODEL_DIR
        / "fraud_online_v5_catboost.cbm"
    )

    model.save_model(model_path)

    print(
        f"\nModel saved:\n{model_path}"
    )

    # -------------------------------------------------------------------------
    # Feature importance
    # -------------------------------------------------------------------------

    print("\n" + "=" * 88)
    print("FEATURE IMPORTANCE")
    print("=" * 88)

    importances = model.get_feature_importance()

    feature_importance = sorted(
        zip(V5_FEATURES, importances),
        key=lambda x: x[1],
        reverse=True,
    )

    for feature, importance in feature_importance:
        print(
            f"{feature:<35} {importance:>10.4f}"
        )

    print("\n" + "=" * 88)
    print("V5 TRAINING COMPLETE")
    print("=" * 88)


if __name__ == "__main__":
    main()