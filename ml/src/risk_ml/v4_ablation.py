from __future__ import annotations

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

ARTIFACTS = ROOT / "artifacts"

TRAIN_PATH = ARTIFACTS / "online_v4_train.parquet"
CAL_PATH = ARTIFACTS / "online_v4_calibration.parquet"
TEST_PATH = ARTIFACTS / "online_v4_test.parquet"

MODEL_DIR = ARTIFACTS / "models"
REPORT_DIR = ARTIFACTS / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CONFIGURATION
# =============================================================================

TARGET = "isFraud"


# Exact V4 feature set.
ALL_FEATURES = [
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


# ---------------------------------------------------------------------------
# Ablation variants
# ---------------------------------------------------------------------------
#
# Variant 1:
#   Exact V4 baseline.
#
# Variant 2:
#   Remove RED features:
#       payment_customer_count
#       merchant_transaction_count
#
# Variant 3:
#   Remove RED features plus the two problematic device features:
#       device_transactions_1h
#       device_customer_count
#
# ---------------------------------------------------------------------------

VARIANTS = {
    "v4_all": ALL_FEATURES,

    "v4_no_red": [
        "amount",
        "customer_transactions_1m",
        "customer_transactions_1h",
        "device_transactions_1h",
        "ip_transactions_1h",
        "customer_degree",
        "device_customer_count",
        "ip_customer_count",
    ],

    "v4_no_red_no_device": [
        "amount",
        "customer_transactions_1m",
        "customer_transactions_1h",
        "ip_transactions_1h",
        "customer_degree",
        "ip_customer_count",
    ],
}


# Keep the CatBoost setup consistent across variants.
CATBOOST_PARAMS = {
    "iterations": 500,
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
    "od_wait": 60,
    "verbose": 100,
    "allow_writing_files": False,
}


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(
    y: pd.Series,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """
    Calculate ranking metrics.

    ROC-AUC:
        Measures ranking quality across all thresholds.

    PR-AUC:
        Particularly important for fraud because fraud is a minority class.
    """

    return {
        "roc_auc": float(
            roc_auc_score(y, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(y, probabilities)
        ),
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 88)
    print("RISK SENTINEL — V4 ABLATION STUDY")
    print("=" * 88)

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------

    print("\nLoading temporal partitions...")

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training artifact not found:\n{TRAIN_PATH}"
        )

    if not CAL_PATH.exists():
        raise FileNotFoundError(
            f"Calibration artifact not found:\n{CAL_PATH}"
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Future-test artifact not found:\n{TEST_PATH}"
        )

    train = pd.read_parquet(TRAIN_PATH)
    calibration = pd.read_parquet(CAL_PATH)
    test = pd.read_parquet(TEST_PATH)

    print(f"Train:       {len(train):,}")
    print(f"Calibration: {len(calibration):,}")
    print(f"Future test: {len(test):,}")

    # -------------------------------------------------------------------------
    # Validate target
    # -------------------------------------------------------------------------

    for name, df in [
        ("train", train),
        ("calibration", calibration),
        ("future_test", test),
    ]:

        if TARGET not in df.columns:
            raise ValueError(
                f"{TARGET!r} missing from {name} partition."
            )

    # -------------------------------------------------------------------------
    # Verify all features exist
    # -------------------------------------------------------------------------

    for variant_name, features in VARIANTS.items():

        for feature in features:

            for partition_name, df in [
                ("train", train),
                ("calibration", calibration),
                ("future_test", test),
            ]:

                if feature not in df.columns:
                    raise ValueError(
                        f"Feature {feature!r} missing from "
                        f"{partition_name} for variant {variant_name!r}."
                    )

    # -------------------------------------------------------------------------
    # Print fraud rates
    # -------------------------------------------------------------------------

    print("\nFraud rates:")

    for name, df in [
        ("train", train),
        ("calibration", calibration),
        ("future_test", test),
    ]:

        fraud_rate = df[TARGET].mean()

        print(
            f"  {name:<12}: {fraud_rate:.6%}"
        )

    # -------------------------------------------------------------------------
    # Store results
    # -------------------------------------------------------------------------

    results: list[dict] = []

    # =========================================================================
    # Run each variant
    # =========================================================================

    for variant_name, features in VARIANTS.items():

        print("\n" + "=" * 88)
        print(f"VARIANT: {variant_name}")
        print("=" * 88)

        print("\nFeatures:")

        for feature in features:
            print(f"  - {feature}")

        # ---------------------------------------------------------------------
        # Internal chronological split
        # ---------------------------------------------------------------------
        #
        # Same logic used during V4 training:
        #
        #   first 90% -> model fitting
        #   last 10%  -> early stopping
        #
        # Calibration and future test remain completely untouched.
        # ---------------------------------------------------------------------

        split_index = int(len(train) * 0.90)

        fit_df = train.iloc[:split_index]
        eval_df = train.iloc[split_index:]

        X_fit = fit_df[features]
        y_fit = fit_df[TARGET].astype(int)

        X_eval = eval_df[features]
        y_eval = eval_df[TARGET].astype(int)

        X_cal = calibration[features]
        y_cal = calibration[TARGET].astype(int)

        X_test = test[features]
        y_test = test[TARGET].astype(int)

        # ---------------------------------------------------------------------
        # Create model
        # ---------------------------------------------------------------------

        model = CatBoostClassifier(
            **CATBOOST_PARAMS
        )

        print("\nTraining CatBoost...")

        start_time = time.perf_counter()

        model.fit(
            X_fit,
            y_fit,
            eval_set=(X_eval, y_eval),
            use_best_model=True,
        )

        train_seconds = time.perf_counter() - start_time

        # ---------------------------------------------------------------------
        # Predictions
        # ---------------------------------------------------------------------

        print("\nGenerating predictions...")

        probabilities_eval = model.predict_proba(
            X_eval
        )[:, 1]

        probabilities_cal = model.predict_proba(
            X_cal
        )[:, 1]

        probabilities_test = model.predict_proba(
            X_test
        )[:, 1]

        # ---------------------------------------------------------------------
        # Metrics
        # ---------------------------------------------------------------------

        eval_metrics = calculate_metrics(
            y_eval,
            probabilities_eval,
        )

        calibration_metrics = calculate_metrics(
            y_cal,
            probabilities_cal,
        )

        future_metrics = calculate_metrics(
            y_test,
            probabilities_test,
        )

        # ---------------------------------------------------------------------
        # Save model
        # ---------------------------------------------------------------------

        model_path = (
            MODEL_DIR
            / f"fraud_{variant_name}_catboost.cbm"
        )

        model.save_model(model_path)

        # ---------------------------------------------------------------------
        # Record results
        # ---------------------------------------------------------------------

        result = {
            "variant": variant_name,

            "feature_count": len(features),

            "features": ",".join(features),

            "best_iteration": int(
                model.get_best_iteration()
            ),

            # Internal early-stop evaluation
            "internal_roc_auc": eval_metrics["roc_auc"],
            "internal_pr_auc": eval_metrics["pr_auc"],

            # Calibration-period ranking
            "calibration_roc_auc": calibration_metrics["roc_auc"],
            "calibration_pr_auc": calibration_metrics["pr_auc"],

            # Untouched future-test ranking
            "future_roc_auc": future_metrics["roc_auc"],
            "future_pr_auc": future_metrics["pr_auc"],

            # Generalization degradation
            "future_vs_internal_roc_drop": (
                future_metrics["roc_auc"]
                - eval_metrics["roc_auc"]
            ),

            "future_vs_internal_pr_drop": (
                future_metrics["pr_auc"]
                - eval_metrics["pr_auc"]
            ),

            "train_seconds": train_seconds,

            "model_path": str(model_path),
        }

        results.append(result)

        # ---------------------------------------------------------------------
        # Console output
        # ---------------------------------------------------------------------

        print("\nMetrics:")

        print(
            f"  Internal ROC-AUC: "
            f"{eval_metrics['roc_auc']:.6f}"
        )

        print(
            f"  Internal PR-AUC:  "
            f"{eval_metrics['pr_auc']:.6f}"
        )

        print(
            f"  Calibration ROC:  "
            f"{calibration_metrics['roc_auc']:.6f}"
        )

        print(
            f"  Calibration PR:   "
            f"{calibration_metrics['pr_auc']:.6f}"
        )

        print(
            f"  Future ROC-AUC:    "
            f"{future_metrics['roc_auc']:.6f}"
        )

        print(
            f"  Future PR-AUC:     "
            f"{future_metrics['pr_auc']:.6f}"
        )

        print(
            f"  Best iteration:    "
            f"{model.get_best_iteration()}"
        )

        print(
            f"  Training time:     "
            f"{train_seconds:.1f}s"
        )

        print(
            f"  Model saved:       "
            f"{model_path}"
        )

    # =========================================================================
    # Build comparison report
    # =========================================================================

    report = pd.DataFrame(results)

    # -------------------------------------------------------------------------
    # Baseline = exact V4
    # -------------------------------------------------------------------------

    baseline_rows = report[
        report["variant"] == "v4_all"
    ]

    if baseline_rows.empty:
        raise RuntimeError(
            "Baseline v4_all result is missing."
        )

    baseline = baseline_rows.iloc[0]

    # -------------------------------------------------------------------------
    # Calculate deltas versus V4 baseline
    # -------------------------------------------------------------------------

    report["future_roc_delta_vs_v4"] = (
        report["future_roc_auc"]
        - baseline["future_roc_auc"]
    )

    report["future_pr_delta_vs_v4"] = (
        report["future_pr_auc"]
        - baseline["future_pr_auc"]
    )

    # -------------------------------------------------------------------------
    # Save CSV
    # -------------------------------------------------------------------------

    report_path = (
        REPORT_DIR
        / "v4_ablation_results.csv"
    )

    report.to_csv(
        report_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Save JSON summary
    # -------------------------------------------------------------------------

    summary = {
        "baseline": {
            "variant": "v4_all",

            "future_roc_auc": float(
                baseline["future_roc_auc"]
            ),

            "future_pr_auc": float(
                baseline["future_pr_auc"]
            ),
        },

        "variants": report[
            [
                "variant",
                "feature_count",
                "future_roc_auc",
                "future_pr_auc",
                "future_roc_delta_vs_v4",
                "future_pr_delta_vs_v4",
                "best_iteration",
            ]
        ].to_dict(
            orient="records"
        ),

        "decision_rule": (
            "Ablation results are evidence for feature selection. "
            "They do not establish production readiness."
        ),
    }

    summary_path = (
        REPORT_DIR
        / "v4_ablation_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # =========================================================================
    # Final comparison
    # =========================================================================

    print("\n" + "=" * 88)
    print("ABLATION COMPARISON")
    print("=" * 88)

    display_columns = [
        "variant",
        "feature_count",
        "internal_roc_auc",
        "internal_pr_auc",
        "future_roc_auc",
        "future_pr_auc",
        "future_roc_delta_vs_v4",
        "future_pr_delta_vs_v4",
    ]

    display = report[display_columns]

    print(
        display.to_string(
            index=False,
            formatters={
                "internal_roc_auc": (
                    lambda x: f"{x:.6f}"
                ),

                "internal_pr_auc": (
                    lambda x: f"{x:.6f}"
                ),

                "future_roc_auc": (
                    lambda x: f"{x:.6f}"
                ),

                "future_pr_auc": (
                    lambda x: f"{x:.6f}"
                ),

                "future_roc_delta_vs_v4": (
                    lambda x: f"{x:+.6f}"
                ),

                "future_pr_delta_vs_v4": (
                    lambda x: f"{x:+.6f}"
                ),
            },
        )
    )

    # =========================================================================
    # Find best variants
    # =========================================================================

    best_roc_index = report[
        "future_roc_auc"
    ].idxmax()

    best_pr_index = report[
        "future_pr_auc"
    ].idxmax()

    best_roc = report.loc[
        best_roc_index
    ]

    best_pr = report.loc[
        best_pr_index
    ]

    print("\n" + "=" * 88)
    print("BEST VARIANTS")
    print("=" * 88)

    print("\nBest future ROC-AUC:")

    print(
        f"  {best_roc['variant']}"
        f" = {best_roc['future_roc_auc']:.6f}"
        f" "
        f"(delta "
        f"{best_roc['future_roc_delta_vs_v4']:+.6f})"
    )

    print("\nBest future PR-AUC:")

    print(
        f"  {best_pr['variant']}"
        f" = {best_pr['future_pr_auc']:.6f}"
        f" "
        f"(delta "
        f"{best_pr['future_pr_delta_vs_v4']:+.6f})"
    )

    # =========================================================================
    # Interpretation
    # =========================================================================

    print("\n" + "=" * 88)
    print("INTERPRETATION")
    print("=" * 88)

    no_red = report[
        report["variant"] == "v4_no_red"
    ].iloc[0]

    no_red_no_device = report[
        report["variant"] == "v4_no_red_no_device"
    ].iloc[0]

    print("\nRED-feature ablation:")

    if no_red["future_roc_auc"] > baseline["future_roc_auc"]:
        print(
            "  Removing RED features IMPROVED future ROC-AUC."
        )
    else:
        print(
            "  Removing RED features did NOT improve future ROC-AUC."
        )

    if no_red["future_pr_auc"] > baseline["future_pr_auc"]:
        print(
            "  Removing RED features IMPROVED future PR-AUC."
        )
    else:
        print(
            "  Removing RED features did NOT improve future PR-AUC."
        )

    print("\nDevice-feature ablation:")

    if (
        no_red_no_device["future_roc_auc"]
        > no_red["future_roc_auc"]
    ):
        print(
            "  Removing device features IMPROVED "
            "future ROC-AUC."
        )
    else:
        print(
            "  Removing device features did NOT improve "
            "future ROC-AUC."
        )

    if (
        no_red_no_device["future_pr_auc"]
        > no_red["future_pr_auc"]
    ):
        print(
            "  Removing device features IMPROVED "
            "future PR-AUC."
        )
    else:
        print(
            "  Removing device features did NOT improve "
            "future PR-AUC."
        )

    # =========================================================================
    # Reports
    # =========================================================================

    print("\n" + "=" * 88)
    print("REPORTS")
    print("=" * 88)

    print(
        f"\nCSV:\n  {report_path}"
    )

    print(
        f"\nJSON:\n  {summary_path}"
    )

    print("\n" + "=" * 88)
    print("V4 ABLATION STUDY COMPLETE")
    print("=" * 88)


if __name__ == "__main__":
    main()