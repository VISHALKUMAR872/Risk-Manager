from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "artifacts" / "online_v5_priors_train.parquet"
CAL_PATH = ROOT / "artifacts" / "online_v5_priors_calibration.parquet"
TEST_PATH = ROOT / "artifacts" / "online_v5_priors_test.parquet"

CATBOOST_PATH = ROOT / "artifacts" / "models" / "fraud_online_v2_priors_catboost.cbm"

REPORT_PATH = ROOT / "artifacts" / "reports" / "v2" / "v2_model_benchmark.json"
CSV_PATH = ROOT / "artifacts" / "reports" / "v2" / "v2_model_benchmark.csv"


FEATURES = [
    "amount",
    "customer_transactions_1m",
    "customer_transactions_1h",
    "device_transactions_1h",
    "ip_transactions_1h",
    "customer_degree",
    "device_customer_count",
    "ip_customer_count",
    "customer_historical_fraud_count",
    "customer_historical_transaction_count",
    "customer_historical_fraud_rate",
    "merchant_historical_fraud_count",
    "merchant_historical_transaction_count",
    "merchant_historical_fraud_rate",
    "device_historical_fraud_count",
    "device_historical_transaction_count",
    "device_historical_fraud_rate",
    "ip_historical_fraud_count",
    "ip_historical_transaction_count",
    "ip_historical_fraud_rate",
    "payment_historical_fraud_count",
    "payment_historical_transaction_count",
    "payment_historical_fraud_rate",
]


def evaluate(name: str, y: np.ndarray, p: np.ndarray, train_seconds: float) -> dict:
    return {
        "model": name,
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "mean_probability": float(p.mean()),
        "std_probability": float(p.std()),
        "min_probability": float(p.min()),
        "max_probability": float(p.max()),
        "train_seconds": float(train_seconds),
    }


def main() -> None:
    print("=" * 72)
    print("RISK SENTINEL V2 — MODEL BENCHMARK")
    print("=" * 72)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(TRAIN_PATH)
    calibration = pd.read_parquet(CAL_PATH)
    test = pd.read_parquet(TEST_PATH)

    for name, df in [
        ("train", train),
        ("calibration", calibration),
        ("test", test),
    ]:
        missing = [f for f in FEATURES if f not in df.columns]
        if missing:
            raise ValueError(f"{name} missing features: {missing}")

    X_train = train[FEATURES]
    y_train = train["isFraud"].to_numpy(dtype=int)

    X_cal = calibration[FEATURES]
    y_cal = calibration["isFraud"].to_numpy(dtype=int)

    X_test = test[FEATURES]
    y_test = test["isFraud"].to_numpy(dtype=int)

    print()
    print("DATA")
    print(f"  Train:       {len(train):,}")
    print(f"  Calibration: {len(calibration):,}")
    print(f"  Future test: {len(test):,}")
    print(f"  Features:    {len(FEATURES)}")
    print(f"  Test fraud:  {int(y_test.sum()):,}")
    print(f"  Test rate:   {y_test.mean():.6%}")

    results = []

    # ------------------------------------------------------------------
    # CATBOOST REFERENCE
    # ------------------------------------------------------------------

    print()
    print("-" * 72)
    print("CATBOOST REFERENCE")
    print("-" * 72)

    cat = CatBoostClassifier()
    cat.load_model(str(CATBOOST_PATH))

    if list(cat.feature_names_) != FEATURES:
        raise ValueError("CatBoost feature order does not match FEATURES")

    start = time.perf_counter()
    p_cat = cat.predict_proba(X_test)[:, 1]
    inference_seconds = time.perf_counter() - start

    cat_result = evaluate(
        "CatBoost",
        y_test,
        p_cat,
        train_seconds=0.0,
    )
    cat_result["inference_seconds"] = float(inference_seconds)

    results.append(cat_result)

    print(f"  ROC-AUC: {cat_result['roc_auc']:.6f}")
    print(f"  PR-AUC:  {cat_result['pr_auc']:.6f}")
    print(f"  Brier:   {cat_result['brier']:.6f}")

    # ------------------------------------------------------------------
    # LIGHTGBM
    # ------------------------------------------------------------------

    print()
    print("-" * 72)
    print("LIGHTGBM")
    print("-" * 72)

    lgb_model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=5.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    start = time.perf_counter()

    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_cal, y_cal)],
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=120,
                verbose=False,
            )
        ],
    )

    lgb_train_seconds = time.perf_counter() - start

    start = time.perf_counter()
    p_lgb = lgb_model.predict_proba(X_test)[:, 1]
    lgb_inference_seconds = time.perf_counter() - start

    lgb_result = evaluate(
        "LightGBM",
        y_test,
        p_lgb,
        train_seconds=lgb_train_seconds,
    )
    lgb_result["inference_seconds"] = float(lgb_inference_seconds)
    lgb_result["best_iteration"] = int(lgb_model.best_iteration_)

    results.append(lgb_result)

    print(f"  Best iteration: {lgb_model.best_iteration_}")
    print(f"  Training time:  {lgb_train_seconds:.2f}s")
    print(f"  ROC-AUC:        {lgb_result['roc_auc']:.6f}")
    print(f"  PR-AUC:         {lgb_result['pr_auc']:.6f}")
    print(f"  Brier:          {lgb_result['brier']:.6f}")

    # ------------------------------------------------------------------
    # XGBOOST
    # ------------------------------------------------------------------

    print()
    print("-" * 72)
    print("XGBOOST")
    print("-" * 72)

    xgb_model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=1500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=5.0,
        reg_alpha=0.0,
        eval_metric="auc",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    start = time.perf_counter()

    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_cal, y_cal)],
        verbose=False,
    )

    xgb_train_seconds = time.perf_counter() - start

    # Select best iteration using calibration AUC.
    calibration_probabilities = xgb_model.predict_proba(X_cal)[:, 1]

    calibration_auc = []

    for iteration in range(1, 1501):
        p_iteration = xgb_model.predict_proba(
            X_cal,
            iteration_range=(0, iteration),
        )[:, 1]

        calibration_auc.append(
            roc_auc_score(y_cal, p_iteration)
        )

    best_iteration = int(np.argmax(calibration_auc)) + 1

    start = time.perf_counter()

    p_xgb = xgb_model.predict_proba(
        X_test,
        iteration_range=(0, best_iteration),
    )[:, 1]

    xgb_inference_seconds = time.perf_counter() - start

    xgb_result = evaluate(
        "XGBoost",
        y_test,
        p_xgb,
        train_seconds=xgb_train_seconds,
    )

    xgb_result["inference_seconds"] = float(xgb_inference_seconds)
    xgb_result["best_iteration"] = best_iteration
    xgb_result["calibration_auc_at_best_iteration"] = float(
        calibration_auc[best_iteration - 1]
    )

    results.append(xgb_result)

    print(f"  Best iteration: {best_iteration}")
    print(f"  Training time:  {xgb_train_seconds:.2f}s")
    print(f"  ROC-AUC:        {xgb_result['roc_auc']:.6f}")
    print(f"  PR-AUC:         {xgb_result['pr_auc']:.6f}")
    print(f"  Brier:          {xgb_result['brier']:.6f}")

    # ------------------------------------------------------------------
    # COMPARISON
    # ------------------------------------------------------------------

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(
        "pr_auc",
        ascending=False,
    )

    result_df.to_csv(CSV_PATH, index=False)

    report = {
        "experiment": "V2 23-feature model benchmark",
        "features": FEATURES,
        "feature_count": len(FEATURES),
        "train_rows": int(len(train)),
        "calibration_rows": int(len(calibration)),
        "future_test_rows": int(len(test)),
        "future_test_fraud": int(y_test.sum()),
        "future_test_fraud_rate": float(y_test.mean()),
        "models": results,
        "selection_metric": "future_test_pr_auc_for_reporting_only",
        "note": (
            "Future-test labels were not used for training or early stopping. "
            "LightGBM used calibration early stopping. XGBoost selected its "
            "best iteration using calibration AUC. CatBoost is the previously "
            "trained V2 reference model."
        ),
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("BENCHMARK SUMMARY")
    print("=" * 72)

    print(
        result_df[
            [
                "model",
                "roc_auc",
                "pr_auc",
                "brier",
                "train_seconds",
            ]
        ].to_string(index=False)
    )

    print()
    print(f"CSV:    {CSV_PATH}")
    print(f"REPORT: {REPORT_PATH}")
    print()
    print("V2 MODEL BENCHMARK: COMPLETE")


if __name__ == "__main__":
    main()
