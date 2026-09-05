from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
MODELS = ARTIFACTS / "models"
REPORTS = ARTIFACTS / "reports"

TEST_PATH = ARTIFACTS / "online_v4_test.parquet"
MODEL_PATH = MODELS / "fraud_online_v5_catboost.cbm"
CALIBRATOR_PATH = MODELS / "fraud_online_v5_isotonic_calibrator.joblib"
LOCKED_POLICY_PATH = REPORTS / "v5_locked_policies.json"

FINAL_METRICS_PATH = REPORTS / "v5_final_test_metrics.json"
FINAL_OPERATING_POINTS_PATH = REPORTS / "v5_final_operating_points.csv"
FINAL_PREDICTIONS_PATH = REPORTS / "v5_final_predictions.parquet"

LGF = 0.80

FEATURES = [
    "amount",
    "customer_transactions_1m",
    "customer_transactions_1h",
    "device_transactions_1h",
    "ip_transactions_1h",
    "customer_degree",
    "device_customer_count",
    "ip_customer_count",
]


def apply_policy(
    probability: np.ndarray,
    expected_loss: np.ndarray,
    policy: dict,
) -> np.ndarray:

    decision = np.full(
        len(probability),
        "APPROVE",
        dtype=object,
    )

    verify = (
        (probability >= policy["verify_probability"])
        | (expected_loss >= policy["verify_expected_loss"])
    )

    review = (
        (probability >= policy["review_probability"])
        | (expected_loss >= policy["review_expected_loss"])
    )

    hold = (
        (probability >= policy["hold_probability"])
        | (expected_loss >= policy["hold_expected_loss"])
    )

    decision[verify] = "VERIFY"
    decision[review] = "REVIEW"
    decision[hold] = "HOLD"

    return decision


def calculate_ece(
    y_true: np.ndarray,
    probability: np.ndarray,
    n_bins: int = 10,
) -> float:

    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    ece = 0.0

    for index in range(n_bins):

        lower = edges[index]
        upper = edges[index + 1]

        if index == n_bins - 1:
            mask = (
                (probability >= lower)
                & (probability <= upper)
            )
        else:
            mask = (
                (probability >= lower)
                & (probability < upper)
            )

        count = int(mask.sum())

        if count == 0:
            continue

        mean_probability = float(
            probability[mask].mean()
        )

        observed_rate = float(
            y_true[mask].mean()
        )

        ece += (
            count / len(y_true)
        ) * abs(
            mean_probability - observed_rate
        )

    return float(ece)


def evaluate_operating_point(
    name: str,
    decision: np.ndarray,
    probability: np.ndarray,
    amount: np.ndarray,
    actual: np.ndarray,
) -> dict:

    total = len(actual)

    fraud_mask = actual == 1
    legitimate_mask = actual == 0

    challenged = decision != "APPROVE"
    approved = ~challenged

    fraud_total = int(fraud_mask.sum())
    legitimate_total = int(legitimate_mask.sum())

    fraud_challenged = int(
        actual[challenged].sum()
    )

    legitimate_challenged = int(
        legitimate_mask[challenged].sum()
    )

    # A challenge is treated as an intervention that prevents the fraud
    # from remaining in the APPROVE population. This is an evaluation
    # assumption, not a production guarantee.
    gross_fraud_amount = float(
        amount[fraud_mask].sum()
    )

    intercepted_fraud_amount = float(
        amount[fraud_mask & challenged].sum()
    )

    baseline_loss = float(
        np.sum(
            amount[fraud_mask] * LGF
        )
    )

    residual_loss = float(
        np.sum(
            amount[fraud_mask & approved] * LGF
        )
    )

    actual_loss_avoided = (
        baseline_loss - residual_loss
    )

    true_positive = fraud_challenged
    false_positive = legitimate_challenged

    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive)
        else 0.0
    )

    recall = (
        true_positive / fraud_total
        if fraud_total
        else 0.0
    )

    f1 = (
        2.0 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    false_positive_rate = (
        false_positive / legitimate_total
        if legitimate_total
        else 0.0
    )

    return {
        "operating_point": name,

        "transactions": total,
        "fraud_transactions": fraud_total,
        "legitimate_transactions": legitimate_total,

        "approve_count": int(
            (decision == "APPROVE").sum()
        ),
        "verify_count": int(
            (decision == "VERIFY").sum()
        ),
        "review_count": int(
            (decision == "REVIEW").sum()
        ),
        "hold_count": int(
            (decision == "HOLD").sum()
        ),

        "intervention_count": int(
            challenged.sum()
        ),
        "intervention_rate": float(
            challenged.mean()
        ),

        "fraud_captured": fraud_challenged,
        "fraud_capture_rate": recall,

        "legitimate_challenged": legitimate_challenged,
        "legitimate_challenge_rate": (
            legitimate_challenged / legitimate_total
            if legitimate_total
            else 0.0
        ),

        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,

        "gross_fraud_amount": gross_fraud_amount,
        "intercepted_fraud_amount": intercepted_fraud_amount,

        "baseline_actual_loss": baseline_loss,
        "residual_actual_loss": residual_loss,
        "actual_loss_avoided": actual_loss_avoided,
        "loss_avoidance_rate": (
            actual_loss_avoided / baseline_loss
            if baseline_loss
            else 0.0
        ),

        "mean_probability": float(
            probability.mean()
        ),
    }


def main() -> None:

    print("=" * 88)
    print("RISK SENTINEL — V5 FINAL HELD-OUT TEST EVALUATION")
    print("=" * 88)

    REPORTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Load locked policies
    # -------------------------------------------------------------------------

    print("\nLoading LOCKED policies...")

    with LOCKED_POLICY_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        locked_artifact = json.load(file)

    if locked_artifact.get(
        "future_test_used_for_selection"
    ) is not False:

        raise RuntimeError(
            "Policy lock artifact does not prove that "
            "future test was excluded from selection."
        )

    policies = locked_artifact["policies"]

    if set(policies) != {
        "conservative",
        "balanced",
        "sensitive",
    }:

        raise RuntimeError(
            "Expected conservative, balanced, and sensitive "
            "locked policies."
        )

    # -------------------------------------------------------------------------
    # Load future test
    # -------------------------------------------------------------------------

    print(
        "\nLoading untouched future-test partition..."
    )

    test = pd.read_parquet(
        TEST_PATH
    )

    print(
        f"Future test rows: {len(test):,}"
    )

    required = FEATURES + [
        "isFraud",
        "amount",
    ]

    missing = [
        column
        for column in required
        if column not in test.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing required columns: {missing}"
        )

    # -------------------------------------------------------------------------
    # Load model and calibrator
    # -------------------------------------------------------------------------

    print("\nLoading V5 CatBoost model...")

    model = CatBoostClassifier()

    model.load_model(
        str(MODEL_PATH)
    )

    print(
        "\nLoading V5 isotonic calibrator..."
    )

    calibrator = joblib.load(
        CALIBRATOR_PATH
    )

    # -------------------------------------------------------------------------
    # Generate probabilities
    # -------------------------------------------------------------------------

    X = test[FEATURES]

    y = (
        test["isFraud"]
        .astype(int)
        .to_numpy()
    )

    amount = (
        test["amount"]
        .astype(float)
        .to_numpy()
    )

    print(
        "\nGenerating final held-out predictions..."
    )

    raw_probability = model.predict_proba(
        X
    )[:, 1]

    calibrated_probability = np.asarray(
        calibrator.predict(
            raw_probability
        ),
        dtype=float,
    )

    expected_loss = (
        calibrated_probability
        * amount
        * LGF
    )

    # -------------------------------------------------------------------------
    # Predictive metrics
    # -------------------------------------------------------------------------

    roc_auc = float(
        roc_auc_score(
            y,
            calibrated_probability,
        )
    )

    pr_auc = float(
        average_precision_score(
            y,
            calibrated_probability,
        )
    )

    brier = float(
        brier_score_loss(
            y,
            calibrated_probability,
        )
    )

    ece = calculate_ece(
        y,
        calibrated_probability,
    )

    # Use 0.50 only as a standard binary classification diagnostic.
    # It is NOT used by the Risk Sentinel policy.
    binary_prediction = (
        calibrated_probability >= 0.50
    ).astype(int)

    binary_precision = float(
        precision_score(
            y,
            binary_prediction,
            zero_division=0,
        )
    )

    binary_recall = float(
        recall_score(
            y,
            binary_prediction,
            zero_division=0,
        )
    )

    binary_f1 = float(
        f1_score(
            y,
            binary_prediction,
            zero_division=0,
        )
    )

    print("\n" + "=" * 88)
    print("PREDICTIVE PERFORMANCE")
    print("=" * 88)

    print(
        f"ROC-AUC: {roc_auc:.6f}"
    )

    print(
        f"PR-AUC:  {pr_auc:.6f}"
    )

    print(
        f"Brier:   {brier:.6f}"
    )

    print(
        f"ECE:     {ece:.6f}"
    )

    print(
        "\nStandard P>=0.50 diagnostic:"
    )

    print(
        f"Precision: {binary_precision:.6f}"
    )

    print(
        f"Recall:    {binary_recall:.6f}"
    )

    print(
        f"F1:        {binary_f1:.6f}"
    )

    # -------------------------------------------------------------------------
    # Operating points
    # -------------------------------------------------------------------------

    operating_rows = []
    prediction_frames = []

    for name in [
        "conservative",
        "balanced",
        "sensitive",
    ]:

        policy = policies[name]

        decision = apply_policy(
            calibrated_probability,
            expected_loss,
            policy,
        )

        metrics = evaluate_operating_point(
            name,
            decision,
            calibrated_probability,
            amount,
            y,
        )

        operating_rows.append(
            metrics
        )

        prediction_frames.append(
            pd.DataFrame(
                {
                    "operating_point": name,
                    "actual_fraud": y,
                    "amount": amount,
                    "raw_probability": raw_probability,
                    "calibrated_probability": (
                        calibrated_probability
                    ),
                    "expected_loss": expected_loss,
                    "decision": decision,
                }
            )
        )

        print("\n" + "-" * 88)
        print(
            name.upper()
        )
        print("-" * 88)

        print(
            f"APPROVE: "
            f"{metrics['approve_count']:,}"
        )

        print(
            f"VERIFY:  "
            f"{metrics['verify_count']:,}"
        )

        print(
            f"REVIEW:  "
            f"{metrics['review_count']:,}"
        )

        print(
            f"HOLD:    "
            f"{metrics['hold_count']:,}"
        )

        print(
            f"Intervention rate: "
            f"{metrics['intervention_rate']:.2%}"
        )

        print(
            f"Fraud captured: "
            f"{metrics['fraud_captured']:,} "
            f"({metrics['fraud_capture_rate']:.2%})"
        )

        print(
            f"Precision: "
            f"{metrics['precision']:.2%}"
        )

        print(
            f"Recall: "
            f"{metrics['recall']:.2%}"
        )

        print(
            f"F1: "
            f"{metrics['f1']:.4f}"
        )

        print(
            f"Legitimate challenged: "
            f"{metrics['legitimate_challenged']:,} "
            f"({metrics['legitimate_challenge_rate']:.2%})"
        )

        print(
            f"False-positive rate: "
            f"{metrics['false_positive_rate']:.2%}"
        )

        print(
            f"Gross fraud amount: "
            f"{metrics['gross_fraud_amount']:,.2f}"
        )

        print(
            f"Fraud amount intercepted: "
            f"{metrics['intercepted_fraud_amount']:,.2f}"
        )

        print(
            f"Actual loss avoided: "
            f"{metrics['actual_loss_avoided']:,.2f}"
        )

        print(
            f"Loss avoidance: "
            f"{metrics['loss_avoidance_rate']:.2%}"
        )

    # -------------------------------------------------------------------------
    # Save final artifacts
    # -------------------------------------------------------------------------

    operating_df = pd.DataFrame(
        operating_rows
    )

    operating_df.to_csv(
        FINAL_OPERATING_POINTS_PATH,
        index=False,
    )

    prediction_df = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    prediction_df.to_parquet(
        FINAL_PREDICTIONS_PATH,
        index=False,
    )

    final_summary = {
        "model": locked_artifact["model"],
        "calibrator": locked_artifact["calibrator"],
        "evaluation_dataset": "online_v4_test.parquet",
        "dataset_role": "FINAL_HELD_OUT_TEST",
        "threshold_selection_dataset": locked_artifact[
            "selection_dataset"
        ],
        "future_test_used_for_selection": False,
        "loss_given_fraud": LGF,

        "predictive_metrics": {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "brier": brier,
            "ece": ece,
            "standard_probability_050": {
                "precision": binary_precision,
                "recall": binary_recall,
                "f1": binary_f1,
            },
        },

        "operating_points": operating_rows,
    }

    FINAL_METRICS_PATH.write_text(
        json.dumps(
            final_summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 88)
    print("FINAL R7 ARTIFACTS")
    print("=" * 88)

    print(
        f"\nMetrics:\n"
        f"  {FINAL_METRICS_PATH}"
    )

    print(
        f"\nOperating points:\n"
        f"  {FINAL_OPERATING_POINTS_PATH}"
    )

    print(
        f"\nPredictions:\n"
        f"  {FINAL_PREDICTIONS_PATH}"
    )

    print("\n" + "=" * 88)
    print("V5 FINAL HELD-OUT TEST EVALUATION COMPLETE")
    print("=" * 88)


if __name__ == "__main__":
    main()
