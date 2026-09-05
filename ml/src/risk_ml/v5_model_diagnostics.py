from __future__ import annotations

from pathlib import Path
import json
import math
import warnings

import joblib
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, Pool
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


warnings.filterwarnings("ignore")


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

ARTIFACTS = ROOT / "artifacts"
MODEL_DIR = ARTIFACTS / "models"
REPORT_DIR = ARTIFACTS / "reports"

MODEL_PATH = (
    MODEL_DIR / "fraud_online_v5_catboost.cbm"
)

CALIBRATOR_PATH = (
    MODEL_DIR / "fraud_online_v5_isotonic_calibrator.joblib"
)

TEST_PATH = (
    ARTIFACTS / "online_v4_test.parquet"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

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


# ---------------------------------------------------------------------------
# Risk Sentinel policy
# ---------------------------------------------------------------------------
#
# These thresholds correspond to the current policy:
#
#   probability >= 0.90 OR expected loss >= 1000 -> HOLD
#   probability >= 0.70 OR expected loss >= 500  -> REVIEW
#   probability >= 0.40 OR expected loss >= 100  -> VERIFY
#   otherwise                                      -> APPROVE
#
# ---------------------------------------------------------------------------

HOLD_PROBABILITY = 0.90
REVIEW_PROBABILITY = 0.70
VERIFY_PROBABILITY = 0.40

HOLD_EXPECTED_LOSS = 1000.0
REVIEW_EXPECTED_LOSS = 500.0
VERIFY_EXPECTED_LOSS = 100.0

DEFAULT_LOSS_GIVEN_FRAUD = 0.80


# =============================================================================
# HELPERS
# =============================================================================

def policy_decision(
    probability: float,
    amount: float,
    loss_given_fraud: float = DEFAULT_LOSS_GIVEN_FRAUD,
) -> tuple[str, float]:

    expected_loss = (
        probability
        * amount
        * loss_given_fraud
    )

    if (
        probability >= HOLD_PROBABILITY
        or expected_loss >= HOLD_EXPECTED_LOSS
    ):
        return "HOLD", expected_loss

    if (
        probability >= REVIEW_PROBABILITY
        or expected_loss >= REVIEW_EXPECTED_LOSS
    ):
        return "REVIEW", expected_loss

    if (
        probability >= VERIFY_PROBABILITY
        or expected_loss >= VERIFY_EXPECTED_LOSS
    ):
        return "VERIFY", expected_loss

    return "APPROVE", expected_loss


def calculate_ece(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, pd.DataFrame]:

    """
    Expected Calibration Error.

    Equal-width probability bins:
        [0.0, 0.1)
        [0.1, 0.2)
        ...
        [0.9, 1.0]

    ECE = sum(
        bin_fraction * |average_prediction - observed_frequency|
    )
    """

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    rows = []
    ece = 0.0

    for i in range(n_bins):

        lower = edges[i]
        upper = edges[i + 1]

        if i == n_bins - 1:
            mask = (
                (probabilities >= lower)
                & (probabilities <= upper)
            )
        else:
            mask = (
                (probabilities >= lower)
                & (probabilities < upper)
            )

        count = int(mask.sum())

        if count == 0:
            continue

        avg_probability = float(
            probabilities[mask].mean()
        )

        observed_rate = float(
            y_true[mask].mean()
        )

        fraction = count / len(y_true)

        calibration_error = abs(
            avg_probability
            - observed_rate
        )

        ece += (
            fraction
            * calibration_error
        )

        rows.append(
            {
                "bin": i,
                "lower": lower,
                "upper": upper,
                "count": count,
                "fraction": fraction,
                "mean_probability": avg_probability,
                "observed_fraud_rate": observed_rate,
                "absolute_error": calibration_error,
            }
        )

    return ece, pd.DataFrame(rows)


def safe_percentage(
    numerator: float,
    denominator: float,
) -> float:

    if denominator == 0:
        return 0.0

    return 100.0 * numerator / denominator


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 88)
    print("RISK SENTINEL — V5 MODEL + CALIBRATION + POLICY DIAGNOSTICS")
    print("=" * 88)

    # =========================================================================
    # Validate artifacts
    # =========================================================================

    required_files = [
        MODEL_PATH,
        CALIBRATOR_PATH,
        TEST_PATH,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required artifact not found:\n{path}"
            )

    # =========================================================================
    # Load future test
    # =========================================================================

    print("\nLoading untouched future-test partition...")

    test = pd.read_parquet(
        TEST_PATH
    )

    print(
        f"Future test rows: {len(test):,}"
    )

    required_columns = (
        V5_FEATURES
        + [TARGET]
    )

    missing = [
        column
        for column in required_columns
        if column not in test.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    # =========================================================================
    # Load model
    # =========================================================================

    print("\nLoading V5 CatBoost model...")

    model = CatBoostClassifier()

    model.load_model(
        MODEL_PATH
    )

    print(
        f"Model: {MODEL_PATH.name}"
    )

    # =========================================================================
    # Load calibrator
    # =========================================================================

    print("\nLoading V5 isotonic calibrator...")

    calibrator = joblib.load(
        CALIBRATOR_PATH
    )

    print(
        f"Calibrator: {CALIBRATOR_PATH.name}"
    )

    # =========================================================================
    # Generate probabilities
    # =========================================================================

    print("\nGenerating V5 predictions...")

    X_test = test[V5_FEATURES]

    y_test = test[TARGET].astype(int).to_numpy()

    raw_probability = model.predict_proba(
        X_test
    )[:, 1]

    calibrated_probability = calibrator.predict(
        raw_probability
    )

    # =========================================================================
    # 1. CALIBRATION DIAGNOSTICS
    # =========================================================================

    print("\n" + "=" * 88)
    print("1. CALIBRATION DIAGNOSTICS")
    print("=" * 88)

    raw_brier = brier_score_loss(
        y_test,
        raw_probability,
    )

    calibrated_brier = brier_score_loss(
        y_test,
        calibrated_probability,
    )

    raw_ece, raw_bins = calculate_ece(
        y_test,
        raw_probability,
    )

    calibrated_ece, calibrated_bins = calculate_ece(
        y_test,
        calibrated_probability,
    )

    print(
        f"\nRaw Brier:        {raw_brier:.6f}"
    )

    print(
        f"Calibrated Brier: {calibrated_brier:.6f}"
    )

    print(
        f"Brier improvement:"
        f" {raw_brier - calibrated_brier:.6f}"
    )

    print(
        f"\nRaw ECE:          {raw_ece:.6f}"
    )

    print(
        f"Calibrated ECE:   {calibrated_ece:.6f}"
    )

    print("\nProbability extremes:")

    print(
        f"Raw P >= 0.90: "
        f"{(raw_probability >= 0.90).sum():,}"
    )

    print(
        f"Calibrated P >= 0.90: "
        f"{(calibrated_probability >= 0.90).sum():,}"
    )

    print(
        f"Raw P >= 0.70: "
        f"{(raw_probability >= 0.70).sum():,}"
    )

    print(
        f"Calibrated P >= 0.70: "
        f"{(calibrated_probability >= 0.70).sum():,}"
    )

    print(
        f"Raw P >= 0.40: "
        f"{(raw_probability >= 0.40).sum():,}"
    )

    print(
        f"Calibrated P >= 0.40: "
        f"{(calibrated_probability >= 0.40).sum():,}"
    )

    print(
        f"Calibrated P == 0: "
        f"{(calibrated_probability == 0).sum():,}"
    )

    print(
        f"Calibrated P == 1: "
        f"{(calibrated_probability == 1).sum():,}"
    )

    # Save calibration bins.

    raw_bins.to_csv(
        REPORT_DIR
        / "v5_raw_calibration_bins.csv",
        index=False,
    )

    calibrated_bins.to_csv(
        REPORT_DIR
        / "v5_calibrated_calibration_bins.csv",
        index=False,
    )

    # =========================================================================
    # 2. SHAP / MODEL INTERPRETATION
    # =========================================================================

    print("\n" + "=" * 88)
    print("2. SHAP / MODEL INTERPRETATION")
    print("=" * 88)

    # CatBoost provides native SHAP calculation.

    print(
        "\nCalculating SHAP values..."
    )

    # CatBoost requires a Pool for native SHAP calculation.
    #
    # The full future-test set is large, so use a deterministic sample for
    # interpretation. This does NOT change model evaluation metrics.
    SHAP_SAMPLE_SIZE = min(20000, len(X_test))

    X_shap = X_test.sample(
        n=SHAP_SAMPLE_SIZE,
        random_state=42,
    )

    shap_pool = Pool(
        X_shap,
    )

    shap_values = model.get_feature_importance(
        data=shap_pool,
        type="ShapValues",
    )

    # Last column is the expected value/base value.

    shap_matrix = shap_values[:, :-1]

    mean_abs_shap = np.abs(
        shap_matrix
    ).mean(axis=0)

    mean_signed_shap = (
        shap_matrix.mean(axis=0)
    )

    shap_rows = []

    for index, feature in enumerate(
        V5_FEATURES
    ):

        shap_rows.append(
            {
                "feature": feature,
                "mean_abs_shap": float(
                    mean_abs_shap[index]
                ),
                "mean_signed_shap": float(
                    mean_signed_shap[index]
                ),
                "catboost_importance": float(
                    model.get_feature_importance()[
                        index
                    ]
                ),
            }
        )

    shap_report = pd.DataFrame(
        shap_rows
    ).sort_values(
        "mean_abs_shap",
        ascending=False,
    )

    print("\nGlobal SHAP importance:")

    for _, row in shap_report.iterrows():

        print(
            f"{row['feature']:<35}"
            f" mean|SHAP|="
            f"{row['mean_abs_shap']:.6f}"
            f"  "
            f"mean SHAP="
            f"{row['mean_signed_shap']:+.6f}"
        )

    shap_report.to_csv(
        REPORT_DIR
        / "v5_shap_global.csv",
        index=False,
    )

    # =========================================================================
    # 3. POLICY SIMULATION
    # =========================================================================

    print("\n" + "=" * 88)
    print("3. RISK SENTINEL POLICY SIMULATION")
    print("=" * 88)

    amounts = pd.to_numeric(
        test["amount"],
        errors="coerce",
    ).fillna(0.0).to_numpy()

    decisions = []
    expected_losses = []

    for probability, amount in zip(
        calibrated_probability,
        amounts,
    ):

        decision, expected_loss = (
            policy_decision(
                float(probability),
                float(amount),
            )
        )

        decisions.append(
            decision
        )

        expected_losses.append(
            expected_loss
        )

    decisions = np.asarray(
        decisions
    )

    expected_losses = np.asarray(
        expected_losses
    )

    policy_df = pd.DataFrame(
        {
            "actual_fraud": y_test,
            "amount": amounts,
            "raw_probability": raw_probability,
            "calibrated_probability": calibrated_probability,
            "expected_loss": expected_losses,
            "decision": decisions,
        }
    )

    # -------------------------------------------------------------------------
    # Decision distribution
    # -------------------------------------------------------------------------

    print("\nDecision distribution:")

    decision_order = [
        "APPROVE",
        "VERIFY",
        "REVIEW",
        "HOLD",
    ]

    policy_rows = []

    total_fraud = int(
        policy_df["actual_fraud"].sum()
    )

    total_amount = float(
        policy_df["amount"].sum()
    )

    for decision in decision_order:

        subset = policy_df[
            policy_df["decision"] == decision
        ]

        count = len(subset)

        fraud_count = int(
            subset["actual_fraud"].sum()
        )

        amount_sum = float(
            subset["amount"].sum()
        )

        expected_loss_sum = float(
            subset["expected_loss"].sum()
        )

        fraud_rate = (
            fraud_count / count
            if count
            else 0.0
        )

        capture_rate = (
            fraud_count / total_fraud
            if total_fraud
            else 0.0
        )

        percentage = (
            count / len(policy_df)
            if len(policy_df)
            else 0.0
        )

        avg_amount = (
            amount_sum / count
            if count
            else 0.0
        )

        avg_expected_loss = (
            expected_loss_sum / count
            if count
            else 0.0
        )

        row = {
            "decision": decision,
            "transactions": count,
            "transaction_pct": percentage * 100.0,
            "fraud_count": fraud_count,
            "fraud_rate_pct": fraud_rate * 100.0,
            "fraud_capture_pct": capture_rate * 100.0,
            "total_amount": amount_sum,
            "average_amount": avg_amount,
            "total_expected_loss": expected_loss_sum,
            "average_expected_loss": avg_expected_loss,
        }

        policy_rows.append(
            row
        )

        print(
            f"\n{decision}"
        )

        print(
            f"  Transactions: "
            f"{count:,} "
            f"({percentage:.2%})"
        )

        print(
            f"  Fraud:        "
            f"{fraud_count:,}"
        )

        print(
            f"  Fraud rate:   "
            f"{fraud_rate:.2%}"
        )

        print(
            f"  Fraud capture:"
            f" {capture_rate:.2%}"
        )

        print(
            f"  Avg amount:   "
            f"{avg_amount:.2f}"
        )

        print(
            f"  Expected loss:"
            f" {expected_loss_sum:.2f}"
        )

    policy_summary = pd.DataFrame(
        policy_rows
    )

    policy_summary.to_csv(
        REPORT_DIR
        / "v5_policy_summary.csv",
        index=False,
    )

    policy_df.to_parquet(
        REPORT_DIR
        / "v5_policy_test_predictions.parquet",
        index=False,
    )

    # =========================================================================
    # 4. HIGH-RISK CAPTURE
    # =========================================================================

    print("\n" + "=" * 88)
    print("4. HIGH-RISK CAPTURE")
    print("=" * 88)

    for threshold in [
        0.40,
        0.70,
        0.90,
    ]:

        mask = (
            calibrated_probability
            >= threshold
        )

        count = int(
            mask.sum()
        )

        fraud_count = int(
            y_test[mask].sum()
        )

        capture_rate = (
            fraud_count / total_fraud
            if total_fraud
            else 0.0
        )

        precision = (
            fraud_count / count
            if count
            else 0.0
        )

        print(
            f"\nP >= {threshold:.2f}"
        )

        print(
            f"  Transactions: "
            f"{count:,} "
            f"({count / len(y_test):.2%})"
        )

        print(
            f"  Fraud: "
            f"{fraud_count:,}"
        )

        print(
            f"  Precision: "
            f"{precision:.2%}"
        )

        print(
            f"  Fraud capture: "
            f"{capture_rate:.2%}"
        )

    # =========================================================================
    # 5. TOP RISK TRANSACTIONS
    # =========================================================================

    print("\n" + "=" * 88)
    print("5. TOP CALIBRATED RISK CASES")
    print("=" * 88)

    top_cases = policy_df.sort_values(
        "calibrated_probability",
        ascending=False,
    ).head(20)

    top_cases.to_csv(
        REPORT_DIR
        / "v5_top_risk_cases.csv",
        index=False,
    )

    print(
        top_cases[
            [
                "actual_fraud",
                "amount",
                "calibrated_probability",
                "expected_loss",
                "decision",
            ]
        ].to_string(
            index=False
        )
    )

    # =========================================================================
    # 6. SUMMARY JSON
    # =========================================================================

    summary = {
        "model": "fraud-online-v5",

        "future_test_rows": len(test),

        "future_fraud_rate": float(
            np.mean(y_test)
        ),

        "calibration": {
            "raw_brier": raw_brier,
            "calibrated_brier": calibrated_brier,
            "brier_improvement": (
                raw_brier
                - calibrated_brier
            ),
            "raw_ece": raw_ece,
            "calibrated_ece": calibrated_ece,
        },

        "extreme_probabilities": {
            "calibrated_zero": int(
                (calibrated_probability == 0).sum()
            ),
            "calibrated_one": int(
                (calibrated_probability == 1).sum()
            ),
            "calibrated_ge_090": int(
                (calibrated_probability >= 0.90).sum()
            ),
            "calibrated_ge_070": int(
                (calibrated_probability >= 0.70).sum()
            ),
            "calibrated_ge_040": int(
                (calibrated_probability >= 0.40).sum()
            ),
        },

        "policy": {
            "loss_given_fraud": DEFAULT_LOSS_GIVEN_FRAUD,
            "hold_probability": HOLD_PROBABILITY,
            "review_probability": REVIEW_PROBABILITY,
            "verify_probability": VERIFY_PROBABILITY,
            "hold_expected_loss": HOLD_EXPECTED_LOSS,
            "review_expected_loss": REVIEW_EXPECTED_LOSS,
            "verify_expected_loss": VERIFY_EXPECTED_LOSS,
        },

        "feature_importance": (
            shap_report.to_dict(
                orient="records"
            )
        ),

        "policy_summary": (
            policy_summary.to_dict(
                orient="records"
            )
        ),
    }

    summary_path = (
        REPORT_DIR
        / "v5_model_diagnostics_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # =========================================================================
    # FINAL
    # =========================================================================

    print("\n" + "=" * 88)
    print("REPORTS GENERATED")
    print("=" * 88)

    print(
        f"\nSHAP:\n"
        f"  {REPORT_DIR / 'v5_shap_global.csv'}"
    )

    print(
        f"\nCalibration bins:\n"
        f"  {REPORT_DIR / 'v5_raw_calibration_bins.csv'}\n"
        f"  {REPORT_DIR / 'v5_calibrated_calibration_bins.csv'}"
    )

    print(
        f"\nPolicy summary:\n"
        f"  {REPORT_DIR / 'v5_policy_summary.csv'}"
    )

    print(
        f"\nPolicy predictions:\n"
        f"  {REPORT_DIR / 'v5_policy_test_predictions.parquet'}"
    )

    print(
        f"\nTop risk cases:\n"
        f"  {REPORT_DIR / 'v5_top_risk_cases.csv'}"
    )

    print(
        f"\nSummary:\n"
        f"  {summary_path}"
    )

    print("\n" + "=" * 88)
    print("V5 MODEL DIAGNOSTICS COMPLETE")
    print("=" * 88)


if __name__ == "__main__":
    main()