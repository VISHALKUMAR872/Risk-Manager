from __future__ import annotations

"""
Risk Sentinel V6 — held-out evaluation + economic cost analysis.

Purpose
-------
Evaluate the exact V5 production model/calibrator on the untouched future-test
partition and produce:
  1. full precision-recall curve data + PNG
  2. threshold sweep
  3. current Conservative/Balanced/Sensitive policy operating points
  4. Wilson 95% confidence intervals for precision/recall
  5. explicit false-positive intervention costs
  6. fraud-loss / false-negative cost accounting
  7. net economic value under configurable assumptions
  8. a compact JSON report suitable for the Track 02 submission

Important
---------
The economic costs below are ILLUSTRATIVE DEFAULTS, not measured merchant
costs. Change them to your validated assumptions before presenting them as
business estimates.

The untouched future-test set is used only for final evaluation. No threshold
selection is performed on this partition.
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
MODELS = ARTIFACTS / "models"
REPORTS = ARTIFACTS / "reports"

TEST_PATH = ARTIFACTS / "online_v4_test.parquet"
MODEL_PATH = MODELS / "fraud_online_v5_catboost.cbm"
CALIBRATOR_PATH = MODELS / "fraud_online_v5_isotonic_calibrator.joblib"

TARGET = "isFraud"
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

# These are the locked V5 production-policy operating points.
POLICIES = {
    "conservative": {
        "verify_probability": 0.30,
        "review_probability": 0.35,
        "hold_probability": 0.60,
        "verify_expected_loss": 125.0,
        "review_expected_loss": 300.0,
        "hold_expected_loss": 700.0,
    },
    "balanced": {
        "verify_probability": 0.25,
        "review_probability": 0.30,
        "hold_probability": 0.60,
        "verify_expected_loss": 100.0,
        "review_expected_loss": 300.0,
        "hold_expected_loss": 700.0,
    },
    "sensitive": {
        "verify_probability": 0.25,
        "review_probability": 0.30,
        "hold_probability": 0.60,
        "verify_expected_loss": 75.0,
        "review_expected_loss": 300.0,
        "hold_expected_loss": 700.0,
    },
}

# Illustrative intervention costs for legitimate transactions.
# These MUST be described as assumptions unless replaced with merchant data.
DEFAULT_FP_COSTS = {
    "VERIFY": 20.0,
    "REVIEW": 75.0,
    "HOLD": 150.0,
}

DEFAULT_FRAUD_INTERCEPTION_RATE = 1.0


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Wilson 95% interval for a binomial proportion."""
    if total <= 0:
        return 0.0, 0.0

    p = successes / total
    denominator = 1.0 + (z * z) / total
    centre = (
        p + (z * z) / (2.0 * total)
    ) / denominator
    margin = (
        z
        * np.sqrt(
            (
                p * (1.0 - p) / total
                + (z * z) / (4.0 * total * total)
            )
        )
        / denominator
    )

    return (
        max(0.0, centre - margin),
        min(1.0, centre + margin),
    )


def apply_policy(
    probability: np.ndarray,
    expected_loss: np.ndarray,
    policy: dict[str, float],
) -> np.ndarray:
    decision = np.full(
        len(probability),
        "APPROVE",
        dtype=object,
    )

    hold = (
        (probability >= policy["hold_probability"])
        | (
            expected_loss
            >= policy["hold_expected_loss"]
        )
    )

    review = (
        (probability >= policy["review_probability"])
        | (
            expected_loss
            >= policy["review_expected_loss"]
        )
    )

    verify = (
        (probability >= policy["verify_probability"])
        | (
            expected_loss
            >= policy["verify_expected_loss"]
        )
    )

    decision[verify] = "VERIFY"
    decision[review] = "REVIEW"
    decision[hold] = "HOLD"

    return decision


def evaluate_policy(
    name: str,
    decision: np.ndarray,
    probability: np.ndarray,
    amount: np.ndarray,
    actual: np.ndarray,
    fp_costs: dict[str, float],
    fraud_interception_rate: float,
) -> dict[str, float | int | str]:
    challenged = decision != "APPROVE"

    legitimate = actual == 0
    fraud = actual == 1

    challenged_count = int(challenged.sum())
    fraud_challenged = int(
        (challenged & fraud).sum()
    )
    legitimate_challenged = int(
        (challenged & legitimate).sum()
    )

    fraud_total = int(fraud.sum())
    legitimate_total = int(legitimate.sum())

    tp = fraud_challenged
    fp = legitimate_challenged
    fn = int((~challenged & fraud).sum())
    tn = int((~challenged & legitimate).sum())

    precision = (
        tp / challenged_count
        if challenged_count
        else 0.0
    )
    recall = (
        tp / fraud_total
        if fraud_total
        else 0.0
    )
    fpr = (
        fp / legitimate_total
        if legitimate_total
        else 0.0
    )

    precision_ci = wilson_interval(
        tp,
        challenged_count,
    )
    recall_ci = wilson_interval(
        tp,
        fraud_total,
    )

    baseline_fraud_loss = float(
        amount[fraud].sum() * LGF
    )

    intercepted_fraud_loss = float(
        amount[challenged & fraud].sum()
        * LGF
        * fraud_interception_rate
    )

    residual_fraud_loss = max(
        0.0,
        baseline_fraud_loss
        - intercepted_fraud_loss,
    )

    legitimate_intervention_cost = 0.0
    for level, cost in fp_costs.items():
        legitimate_intervention_cost += float(
            (
                (decision == level)
                & legitimate
            ).sum()
        ) * cost

    fraud_intervention_cost = 0.0
    for level, cost in fp_costs.items():
        fraud_intervention_cost += float(
            (
                (decision == level)
                & fraud
            ).sum()
        ) * cost

    total_intervention_cost = (
        legitimate_intervention_cost
        + fraud_intervention_cost
    )

    loss_avoided = intercepted_fraud_loss

    net_economic_value = (
        loss_avoided
        - total_intervention_cost
    )

    gross_fraud_value = float(
        amount[fraud].sum()
    )

    return {
        "policy": name,
        "n_total": len(actual),
        "n_fraud": fraud_total,
        "n_legitimate": legitimate_total,
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
        "intervention_count": challenged_count,
        "intervention_rate": (
            challenged_count / len(actual)
            if len(actual)
            else 0.0
        ),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "precision_ci_low": precision_ci[0],
        "precision_ci_high": precision_ci[1],
        "recall": recall,
        "recall_ci_low": recall_ci[0],
        "recall_ci_high": recall_ci[1],
        "fpr": fpr,
        "fraud_captured": fraud_challenged,
        "fraud_capture_rate": recall,
        "legitimate_challenged": legitimate_challenged,
        "legitimate_challenge_rate": fpr,
        "gross_fraud_value": gross_fraud_value,
        "baseline_fraud_loss": baseline_fraud_loss,
        "intercepted_fraud_loss": intercepted_fraud_loss,
        "residual_fraud_loss": residual_fraud_loss,
        "loss_avoided": loss_avoided,
        "loss_avoidance_rate": (
            loss_avoided / baseline_fraud_loss
            if baseline_fraud_loss
            else 0.0
        ),
        "legitimate_intervention_cost": (
            legitimate_intervention_cost
        ),
        "fraud_intervention_cost": (
            fraud_intervention_cost
        ),
        "total_intervention_cost": (
            total_intervention_cost
        ),
        "net_economic_value": (
            net_economic_value
        ),
        "mean_probability": float(
            probability.mean()
        ),
        "assumed_fraud_interception_rate": (
            fraud_interception_rate
        ),
        "assumed_lgf": LGF,
    }


def threshold_metrics(
    y: np.ndarray,
    probability: np.ndarray,
    amount: np.ndarray,
    fp_cost: float,
) -> pd.DataFrame:
    """
    Probability-only threshold sweep.

    This is deliberately separate from the production policy because the
    production policy combines probability and expected-loss triggers.
    """
    rows: list[dict[str, float | int]] = []

    fraud = y == 1
    legitimate = y == 0

    for threshold in np.linspace(
        0.0,
        1.0,
        1001,
    ):
        flagged = probability >= threshold

        tp = int((flagged & fraud).sum())
        fp = int((flagged & legitimate).sum())
        fn = int((~flagged & fraud).sum())
        tn = int((~flagged & legitimate).sum())

        flagged_count = tp + fp
        fraud_total = int(fraud.sum())
        legitimate_total = int(legitimate.sum())

        precision = (
            tp / flagged_count
            if flagged_count
            else 1.0
        )
        recall = (
            tp / fraud_total
            if fraud_total
            else 0.0
        )
        fpr = (
            fp / legitimate_total
            if legitimate_total
            else 0.0
        )

        loss_avoided = float(
            amount[flagged & fraud].sum()
            * LGF
            * DEFAULT_FRAUD_INTERCEPTION_RATE
        )

        fp_cost_total = float(fp) * fp_cost

        rows.append(
            {
                "threshold": threshold,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "flagged": flagged_count,
                "precision": precision,
                "recall": recall,
                "fpr": fpr,
                "intervention_rate": (
                    flagged_count / len(y)
                ),
                "loss_avoided": loss_avoided,
                "fp_cost": fp_cost_total,
                "net_value": (
                    loss_avoided
                    - fp_cost_total
                ),
            }
        )

    return pd.DataFrame(rows)


def plot_pr_curve(
    y: np.ndarray,
    probability: np.ndarray,
    policy_results: pd.DataFrame,
    path: Path,
) -> None:
    precision, recall, thresholds = (
        precision_recall_curve(
            y,
            probability,
        )
    )

    figure, axis = plt.subplots(
        figsize=(10, 7)
    )

    axis.plot(
        recall,
        precision,
        linewidth=2.0,
        label="V5 calibrated model",
    )

    for _, row in policy_results.iterrows():
        axis.scatter(
            row["recall"],
            row["precision"],
            s=70,
            label=(
                f'{row["policy"].title()} policy '
                f'({row["intervention_rate"] * 100:.2f}% intervention)'
            ),
        )

        axis.annotate(
            row["policy"].title(),
            (
                row["recall"],
                row["precision"],
            ),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=9,
        )

    prevalence = float(y.mean())

    axis.axhline(
        prevalence,
        linestyle="--",
        linewidth=1.0,
        label=f"Fraud prevalence = {prevalence:.2%}",
    )

    axis.set_title(
        "Risk Sentinel V6 — Precision–Recall Curve"
    )
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.02)
    axis.grid(alpha=0.20)
    axis.legend(loc="best", fontsize=8)

    figure.tight_layout()
    figure.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_net_value(
    threshold: pd.DataFrame,
    path: Path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    axis.plot(
        threshold["threshold"],
        threshold["net_value"],
        linewidth=1.8,
    )

    best = threshold.loc[
        threshold["net_value"].idxmax()
    ]

    axis.scatter(
        best["threshold"],
        best["net_value"],
        s=70,
        label=(
            f'Best probability-only threshold '
            f'{best["threshold"]:.3f}'
        ),
    )

    axis.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    axis.set_title(
        "Risk Sentinel V6 — Net Economic Value vs Probability Threshold"
    )
    axis.set_xlabel("Fraud-probability threshold")
    axis.set_ylabel("Net economic value (INR)")
    axis.grid(alpha=0.20)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Risk Sentinel V6 held-out evaluation "
            "and economic cost analysis."
        )
    )

    parser.add_argument(
        "--verify-cost",
        type=float,
        default=DEFAULT_FP_COSTS["VERIFY"],
        help="Illustrative INR cost of a legitimate VERIFY.",
    )
    parser.add_argument(
        "--review-cost",
        type=float,
        default=DEFAULT_FP_COSTS["REVIEW"],
        help="Illustrative INR cost of a legitimate REVIEW.",
    )
    parser.add_argument(
        "--hold-cost",
        type=float,
        default=DEFAULT_FP_COSTS["HOLD"],
        help="Illustrative INR cost of a legitimate HOLD.",
    )
    parser.add_argument(
        "--fraud-interception-rate",
        type=float,
        default=DEFAULT_FRAUD_INTERCEPTION_RATE,
        help=(
            "Fraction of challenged fraud loss assumed prevented. "
            "Default 1.0 matches the current modeled capture assumption."
        ),
    )

    args = parser.parse_args()

    if not 0.0 <= args.fraud_interception_rate <= 1.0:
        raise ValueError(
            "--fraud-interception-rate must be in [0, 1]."
        )

    fp_costs = {
        "VERIFY": args.verify_cost,
        "REVIEW": args.review_cost,
        "HOLD": args.hold_cost,
    }

    REPORTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 88)
    print("RISK SENTINEL — V6 HELD-OUT ECONOMIC EVALUATION")
    print("=" * 88)

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Future-test dataset not found:\n{TEST_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"V5 model not found:\n{MODEL_PATH}"
        )

    if not CALIBRATOR_PATH.exists():
        raise FileNotFoundError(
            f"V5 calibrator not found:\n{CALIBRATOR_PATH}"
        )

    print("\nLoading untouched future-test partition...")
    df = pd.read_parquet(TEST_PATH)

    required = FEATURES + [TARGET]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing required columns: {missing}"
        )

    y = (
        df[TARGET]
        .astype(int)
        .to_numpy()
    )

    amount = (
        pd.to_numeric(
            df["amount"],
            errors="coerce",
        )
        .fillna(0.0)
        .to_numpy()
    )

    print(f"Future test rows : {len(df):,}")
    print(f"Fraud rows       : {int(y.sum()):,}")
    print(
        f"Fraud prevalence : {y.mean():.6%}"
    )
    print(
        f"Gross fraud value: "
        f"₹{amount[y == 1].sum():,.2f}"
    )
    print(
        f"Modeled LGF      : {LGF:.0%}"
    )

    print("\nLoading V5 CatBoost model...")
    model = CatBoostClassifier()
    model.load_model(
        str(MODEL_PATH)
    )

    print("Loading V5 isotonic calibrator...")
    calibrator = joblib.load(
        CALIBRATOR_PATH
    )

    X = df[FEATURES]

    raw_probability = (
        model.predict_proba(X)[:, 1]
    )

    calibrated_probability = np.asarray(
        calibrator.predict(
            raw_probability
        ),
        dtype=float,
    )

    calibrated_probability = np.clip(
        calibrated_probability,
        0.0,
        1.0,
    )

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

    print("\n" + "=" * 88)
    print("MODEL METRICS — UNTOUCHED FUTURE TEST")
    print("=" * 88)
    print(
        f"ROC-AUC : {roc_auc:.6f}"
    )
    print(
        f"PR-AUC  : {pr_auc:.6f}"
    )

    expected_loss = (
        calibrated_probability
        * amount
        * LGF
    )

    print("\n" + "=" * 88)
    print("LOCKED POLICY OPERATING POINTS")
    print("=" * 88)

    policy_rows: list[dict[str, float | int | str]] = []

    for name, policy in POLICIES.items():
        decision = apply_policy(
            calibrated_probability,
            expected_loss,
            policy,
        )

        result = evaluate_policy(
            name,
            decision,
            calibrated_probability,
            amount,
            y,
            fp_costs,
            args.fraud_interception_rate,
        )

        policy_rows.append(result)

        print(f"\n{name.upper()}")
        print("-" * 60)
        print(
            f"Intervention rate : "
            f"{result['intervention_rate']:.4%}"
        )
        print(
            f"TP / FP / TN / FN  : "
            f"{result['tp']} / {result['fp']} / "
            f"{result['tn']} / {result['fn']}"
        )
        print(
            f"Precision          : "
            f"{result['precision']:.2%} "
            f"(95% CI "
            f"{result['precision_ci_low']:.2%}–"
            f"{result['precision_ci_high']:.2%})"
        )
        print(
            f"Recall             : "
            f"{result['recall']:.2%} "
            f"(95% CI "
            f"{result['recall_ci_low']:.2%}–"
            f"{result['recall_ci_high']:.2%})"
        )
        print(
            f"False-positive rate: "
            f"{result['fpr']:.2%}"
        )
        print(
            f"Fraud captured     : "
            f"{result['fraud_captured']:,} / "
            f"{result['n_fraud']:,}"
        )
        print(
            f"Baseline fraud loss: "
            f"₹{result['baseline_fraud_loss']:,.2f}"
        )
        print(
            f"Loss avoided       : "
            f"₹{result['loss_avoided']:,.2f}"
        )
        print(
            f"FP intervention cost: "
            f"₹{result['legitimate_intervention_cost']:,.2f}"
        )
        print(
            f"Total intervention cost: "
            f"₹{result['total_intervention_cost']:,.2f}"
        )
        print(
            f"Net economic value : "
            f"₹{result['net_economic_value']:,.2f}"
        )

    policy_df = pd.DataFrame(
        policy_rows
    )

    policy_path = (
        REPORTS
        / "v6_policy_economic_evaluation.csv"
    )

    policy_df.to_csv(
        policy_path,
        index=False,
    )

    print("\n" + "=" * 88)
    print("PROBABILITY-ONLY THRESHOLD SWEEP")
    print("=" * 88)

    threshold_df = threshold_metrics(
        y,
        calibrated_probability,
        amount,
        fp_cost=fp_costs["VERIFY"],
    )

    threshold_path = (
        REPORTS
        / "v6_probability_threshold_sweep.csv"
    )

    threshold_df.to_csv(
        threshold_path,
        index=False,
    )

    best = threshold_df.loc[
        threshold_df["net_value"].idxmax()
    ]

    print(
        f"Best probability-only threshold "
        f"under VERIFY-cost assumption: "
        f"{best['threshold']:.3f}"
    )
    print(
        f"  precision       = "
        f"{best['precision']:.2%}"
    )
    print(
        f"  recall          = "
        f"{best['recall']:.2%}"
    )
    print(
        f"  FPR             = "
        f"{best['fpr']:.2%}"
    )
    print(
        f"  flagged         = "
        f"{int(best['flagged']):,}"
    )
    print(
        f"  loss avoided    = "
        f"₹{best['loss_avoided']:,.2f}"
    )
    print(
        f"  FP cost         = "
        f"₹{best['fp_cost']:,.2f}"
    )
    print(
        f"  net value       = "
        f"₹{best['net_value']:,.2f}"
    )

    print("\nGenerating PR curve...")
    pr_curve_path = (
        REPORTS
        / "v6_precision_recall_curve.png"
    )

    plot_pr_curve(
        y,
        calibrated_probability,
        policy_df,
        pr_curve_path,
    )

    print("Generating net-value curve...")
    net_value_path = (
        REPORTS
        / "v6_net_economic_value_curve.png"
    )

    plot_net_value(
        threshold_df,
        net_value_path,
    )

    # Full sklearn PR curve data.
    precision, recall, thresholds = (
        precision_recall_curve(
            y,
            calibrated_probability,
        )
    )

    pr_rows = pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "threshold": np.r_[
                thresholds,
                np.nan,
            ],
        }
    )

    pr_path = (
        REPORTS
        / "v6_precision_recall_curve.csv"
    )

    pr_rows.to_csv(
        pr_path,
        index=False,
    )

    summary = {
        "evaluation_version": "v6",
        "model_version": "fraud-online-v5",
        "calibration_version": "isotonic-online-v5",
        "dataset": str(TEST_PATH),
        "dataset_partition": "untouched_future_test",
        "rows": int(len(df)),
        "fraud_rows": int(y.sum()),
        "legitimate_rows": int((y == 0).sum()),
        "fraud_prevalence": float(y.mean()),
        "gross_fraud_value": float(
            amount[y == 1].sum()
        ),
        "lgf": LGF,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "economic_assumptions": {
            "verify_fp_cost_inr": fp_costs["VERIFY"],
            "review_fp_cost_inr": fp_costs["REVIEW"],
            "hold_fp_cost_inr": fp_costs["HOLD"],
            "fraud_interception_rate": (
                args.fraud_interception_rate
            ),
            "note": (
                "Illustrative assumptions unless "
                "replaced by validated merchant data."
            ),
        },
        "policies": policy_rows,
        "best_probability_only_threshold": {
            key: (
                float(value)
                if isinstance(
                    value,
                    (np.floating, float),
                )
                else int(value)
                if isinstance(
                    value,
                    (np.integer, int),
                )
                else value
            )
            for key, value in best.to_dict().items()
        },
        "artifacts": {
            "policy_csv": str(policy_path),
            "threshold_csv": str(threshold_path),
            "pr_curve_csv": str(pr_path),
            "pr_curve_png": str(pr_curve_path),
            "net_value_png": str(net_value_path),
        },
    }

    summary_path = (
        REPORTS
        / "v6_economic_evaluation_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 88)
    print("V6 EVALUATION COMPLETE")
    print("=" * 88)
    print(f"Summary : {summary_path}")
    print(f"Policy  : {policy_path}")
    print(f"PR CSV  : {pr_path}")
    print(f"PR PNG  : {pr_curve_path}")
    print(f"Value   : {net_value_path}")

    print("\nIMPORTANT:")
    print(
        "Economic costs are assumptions unless replaced "
        "with validated merchant measurements."
    )
    print(
        "Production-policy points combine probability "
        "and expected-loss triggers; they are marked on "
        "the PR curve as operating points, not pure "
        "probability thresholds."
    )


if __name__ == "__main__":
    main()
