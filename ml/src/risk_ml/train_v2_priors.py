from __future__ import annotations

from pathlib import Path
import time

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
MODEL_DIR = ARTIFACTS / "models"
REPORT_DIR = ARTIFACTS / "reports" / "v2"

TRAIN_PATH = ARTIFACTS / "online_v5_priors_train.parquet"
CALIBRATION_PATH = ARTIFACTS / "online_v5_priors_calibration.parquet"
TEST_PATH = ARTIFACTS / "online_v5_priors_test.parquet"

MODEL_PATH = MODEL_DIR / "fraud_online_v2_priors_catboost.cbm"
REPORT_PATH = REPORT_DIR / "v2_priors_raw_evaluation.csv"

TARGET = "isFraud"

# Exact V5 control features.
BASE_FEATURES = [
    "amount",
    "customer_transactions_1m",
    "customer_transactions_1h",
    "device_transactions_1h",
    "ip_transactions_1h",
    "customer_degree",
    "device_customer_count",
    "ip_customer_count",
]

# New V2 historical fraud features.
PRIOR_FEATURES = []

for entity in [
    "customer",
    "merchant",
    "device",
    "ip",
    "payment",
]:
    PRIOR_FEATURES.extend(
        [
            f"{entity}_historical_fraud_count",
            f"{entity}_historical_transaction_count",
            f"{entity}_historical_fraud_rate",
        ]
    )

V2_FEATURES = BASE_FEATURES + PRIOR_FEATURES

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


def evaluate(y: pd.Series, probabilities) -> tuple[float, float]:
    roc_auc = roc_auc_score(y, probabilities)
    pr_auc = average_precision_score(y, probabilities)

    return float(roc_auc), float(pr_auc)


def main() -> None:

    print("=" * 88)
    print("RISK SENTINEL — V2 MODEL A: HISTORICAL FRAUD PRIORS")
    print("=" * 88)

    print("\nBase features:")
    for feature in BASE_FEATURES:
        print(f"  - {feature}")

    print("\nHistorical-prior features:")
    for feature in PRIOR_FEATURES:
        print(f"  - {feature}")

    print(f"\nTotal features: {len(V2_FEATURES)}")

    print("\nLoading datasets...")

    train = pd.read_parquet(TRAIN_PATH)
    calibration = pd.read_parquet(CALIBRATION_PATH)
    test = pd.read_parquet(TEST_PATH)

    print(f"Train:       {len(train):,}")
    print(f"Calibration: {len(calibration):,}")
    print(f"Future test: {len(test):,}")

    required_columns = V2_FEATURES + [TARGET]

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

        if df[V2_FEATURES].isna().any().any():
            raise ValueError(
                f"{dataset_name} contains NaN values."
            )

    print("\nFraud rates:")
    print(f"  Train:       {train[TARGET].mean():.6%}")
    print(f"  Calibration: {calibration[TARGET].mean():.6%}")
    print(f"  Future test: {test[TARGET].mean():.6%}")

    # ------------------------------------------------------------------
    # Exact same chronological 90/10 internal split as V5.
    # ------------------------------------------------------------------

    split_index = int(len(train) * 0.90)

    fit_df = train.iloc[:split_index]
    early_stop_df = train.iloc[split_index:]

    print("\nInternal training split:")
    print(f"  Fit:          {len(fit_df):,}")
    print(f"  Early-stop:   {len(early_stop_df):,}")

    X_fit = fit_df[V2_FEATURES]
    y_fit = fit_df[TARGET].astype(int)

    X_eval = early_stop_df[V2_FEATURES]
    y_eval = early_stop_df[TARGET].astype(int)

    X_calibration = calibration[V2_FEATURES]
    y_calibration = calibration[TARGET].astype(int)

    X_test = test[V2_FEATURES]
    y_test = test[TARGET].astype(int)

    # ------------------------------------------------------------------
    # Train.
    # ------------------------------------------------------------------

    print("\n" + "=" * 88)
    print("TRAINING CATBOOST — V2 MODEL A")
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

    print(f"\nTraining time: {elapsed:.2f}s")

    # ------------------------------------------------------------------
    # Predictions.
    # ------------------------------------------------------------------

    print("\nGenerating predictions...")

    p_eval = model.predict_proba(X_eval)[:, 1]
    p_calibration = model.predict_proba(X_calibration)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]

    internal_roc, internal_pr = evaluate(y_eval, p_eval)
    calibration_roc, calibration_pr = evaluate(
        y_calibration,
        p_calibration,
    )
    future_roc, future_pr = evaluate(y_test, p_test)

    # ------------------------------------------------------------------
    # Results.
    # ------------------------------------------------------------------

    print("\n" + "=" * 88)
    print("V2 MODEL A RESULTS")
    print("=" * 88)

    print("\nINTERNAL EARLY-STOP")
    print(f"ROC-AUC: {internal_roc:.6f}")
    print(f"PR-AUC:  {internal_pr:.6f}")

    print("\nCALIBRATION PERIOD — RAW MODEL")
    print(f"ROC-AUC: {calibration_roc:.6f}")
    print(f"PR-AUC:  {calibration_pr:.6f}")

    print("\nFUTURE TEST — RAW MODEL")
    print(f"ROC-AUC: {future_roc:.6f}")
    print(f"PR-AUC:  {future_pr:.6f}")

    print("\nMODEL")
    print(f"Best iteration: {model.get_best_iteration()}")
    print(f"Training time:  {elapsed:.2f}s")

    # ------------------------------------------------------------------
    # Save model.
    # ------------------------------------------------------------------

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model.save_model(MODEL_PATH)

    print(f"\nModel saved:")
    print(MODEL_PATH)

    # ------------------------------------------------------------------
    # Feature importance.
    # ------------------------------------------------------------------

    print("\n" + "=" * 88)
    print("FEATURE IMPORTANCE")
    print("=" * 88)

    importances = model.get_feature_importance()

    feature_importance = sorted(
        zip(V2_FEATURES, importances),
        key=lambda x: x[1],
        reverse=True,
    )

    for feature, importance in feature_importance:
        print(f"{feature:<45} {importance:>10.4f}")

    # ------------------------------------------------------------------
    # Save evaluation report.
    # ------------------------------------------------------------------

    report = pd.DataFrame(
        [
            {
                "model": "v2_priors_catboost",
                "partition": "internal_early_stop",
                "roc_auc": internal_roc,
                "pr_auc": internal_pr,
            },
            {
                "model": "v2_priors_catboost",
                "partition": "calibration_raw",
                "roc_auc": calibration_roc,
                "pr_auc": calibration_pr,
            },
            {
                "model": "v2_priors_catboost",
                "partition": "future_test_raw",
                "roc_auc": future_roc,
                "pr_auc": future_pr,
            },
        ]
    )

    report.to_csv(REPORT_PATH, index=False)

    print(f"\nEvaluation report saved:")
    print(REPORT_PATH)

    print("\n" + "=" * 88)
    print("V2 MODEL A TRAINING COMPLETE")
    print("=" * 88)


if __name__ == "__main__":
    main()
