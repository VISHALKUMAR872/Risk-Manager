from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
MODELS = ARTIFACTS / "models"
REPORTS = ARTIFACTS / "reports"

TEST_PATH = ARTIFACTS / "online_v4_test.parquet"
MODEL_PATH = MODELS / "fraud_online_v5_catboost.cbm"
CALIBRATOR_PATH = MODELS / "fraud_online_v5_isotonic_calibrator.joblib"

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


POLICIES = {
    "current": {
        "verify_probability": 0.40,
        "review_probability": 0.70,
        "hold_probability": 0.90,
        "verify_expected_loss": 100.0,
        "review_expected_loss": 500.0,
        "hold_expected_loss": 1000.0,
    },
    "balanced": {
        "verify_probability": 0.30,
        "review_probability": 0.60,
        "hold_probability": 0.85,
        "verify_expected_loss": 100.0,
        "review_expected_loss": 500.0,
        "hold_expected_loss": 1000.0,
    },
    "sensitive": {
        "verify_probability": 0.25,
        "review_probability": 0.50,
        "hold_probability": 0.80,
        "verify_expected_loss": 75.0,
        "review_expected_loss": 400.0,
        "hold_expected_loss": 800.0,
    },
    "conservative": {
        "verify_probability": 0.35,
        "review_probability": 0.65,
        "hold_probability": 0.90,
        "verify_expected_loss": 150.0,
        "review_expected_loss": 600.0,
        "hold_expected_loss": 1200.0,
    },
}


def apply_policy(
    probability: np.ndarray,
    expected_loss: np.ndarray,
    policy: dict,
) -> np.ndarray:
    decision = np.full(len(probability), "APPROVE", dtype=object)

    hold = (
        (probability >= policy["hold_probability"])
        | (expected_loss >= policy["hold_expected_loss"])
    )

    review = (
        (probability >= policy["review_probability"])
        | (expected_loss >= policy["review_expected_loss"])
    )

    verify = (
        (probability >= policy["verify_probability"])
        | (expected_loss >= policy["verify_expected_loss"])
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
) -> dict:
    total = len(actual)
    fraud_total = int(actual.sum())
    legitimate_total = total - fraud_total

    baseline_loss = float(
        np.sum(actual * amount * LGF)
    )

    actual_loss = actual * amount * LGF

    # Operationally, VERIFY/REVIEW/HOLD are the challenged population.
    challenged = decision != "APPROVE"
    challenged_count = int(challenged.sum())

    fraud_challenged = int(actual[challenged].sum())
    legitimate_challenged = int((1 - actual[challenged]).sum())

    # Loss remaining in APPROVE population.
    approved = decision == "APPROVE"
    residual_loss = float(actual_loss[approved].sum())

    loss_avoided = baseline_loss - residual_loss

    result = {
        "policy": name,
        "transactions": total,
        "fraud_transactions": fraud_total,
        "legitimate_transactions": legitimate_total,

        "approve_count": int((decision == "APPROVE").sum()),
        "verify_count": int((decision == "VERIFY").sum()),
        "review_count": int((decision == "REVIEW").sum()),
        "hold_count": int((decision == "HOLD").sum()),

        "approve_rate": float((decision == "APPROVE").mean()),
        "verify_rate": float((decision == "VERIFY").mean()),
        "review_rate": float((decision == "REVIEW").mean()),
        "hold_rate": float((decision == "HOLD").mean()),

        "challenged_count": challenged_count,
        "challenge_rate": float(challenged.mean()),

        "fraud_challenged": fraud_challenged,
        "fraud_capture_rate": (
            fraud_challenged / fraud_total
            if fraud_total
            else 0.0
        ),

        "legitimate_challenged": legitimate_challenged,
        "legitimate_challenge_rate": (
            legitimate_challenged / legitimate_total
            if legitimate_total
            else 0.0
        ),

        "challenge_precision": (
            fraud_challenged / challenged_count
            if challenged_count
            else 0.0
        ),

        "baseline_actual_loss": baseline_loss,
        "residual_actual_loss": residual_loss,
        "actual_loss_avoided": loss_avoided,
        "loss_avoidance_rate": (
            loss_avoided / baseline_loss
            if baseline_loss
            else 0.0
        ),

        "mean_probability": float(probability.mean()),
    }

    for level in ["VERIFY", "REVIEW", "HOLD"]:
        mask = decision == level

        if mask.any():
            result[f"{level.lower()}_fraud_rate"] = float(
                actual[mask].mean()
            )
            result[f"{level.lower()}_avg_amount"] = float(
                amount[mask].mean()
            )
        else:
            result[f"{level.lower()}_fraud_rate"] = 0.0
            result[f"{level.lower()}_avg_amount"] = 0.0

    return result


def calibration_support(
    probability: np.ndarray,
    actual: np.ndarray,
) -> pd.DataFrame:
    rows = []

    for value in [0.0, 1.0]:
        mask = np.isclose(probability, value)

        if not mask.any():
            continue

        rows.append(
            {
                "probability": value,
                "count": int(mask.sum()),
                "fraud_count": int(actual[mask].sum()),
                "fraud_rate": float(actual[mask].mean()),
                "min_amount": float(
                    df.loc[mask, "amount"].min()
                ),
                "max_amount": float(
                    df.loc[mask, "amount"].max()
                ),
                "mean_amount": float(
                    df.loc[mask, "amount"].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


print("=" * 88)
print("RISK SENTINEL — V5 POLICY SENSITIVITY + CALIBRATION SUPPORT")
print("=" * 88)

REPORTS.mkdir(parents=True, exist_ok=True)

print("\nLoading untouched future-test partition...")
df = pd.read_parquet(TEST_PATH)

print(f"Future test rows: {len(df):,}")

missing = [column for column in FEATURES + ["isFraud"] if column not in df.columns]

if missing:
    raise RuntimeError(f"Missing required columns: {missing}")

X = df[FEATURES]
y = df["isFraud"].astype(int).to_numpy()
amount = df["amount"].astype(float).to_numpy()

print("\nLoading V5 model...")
model = CatBoostClassifier()
model.load_model(str(MODEL_PATH))

print("\nLoading V5 calibrator...")
calibrator = joblib.load(CALIBRATOR_PATH)

print("\nGenerating calibrated probabilities...")
raw_probability = model.predict_proba(X)[:, 1]
calibrated_probability = np.asarray(
    calibrator.predict(raw_probability)
)

expected_loss = calibrated_probability * amount * LGF


print("\n" + "=" * 88)
print("1. POLICY SENSITIVITY")
print("=" * 88)

results = []

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
    )

    results.append(result)

    print(f"\n{name.upper()}")
    print("-" * 40)

    print(
        f"APPROVE : {(decision == 'APPROVE').sum():,}"
        f" ({(decision == 'APPROVE').mean() * 100:.2f}%)"
    )

    print(
        f"VERIFY  : {(decision == 'VERIFY').sum():,}"
        f" ({(decision == 'VERIFY').mean() * 100:.2f}%)"
    )

    print(
        f"REVIEW  : {(decision == 'REVIEW').sum():,}"
        f" ({(decision == 'REVIEW').mean() * 100:.2f}%)"
    )

    print(
        f"HOLD    : {(decision == 'HOLD').sum():,}"
        f" ({(decision == 'HOLD').mean() * 100:.2f}%)"
    )

    print(
        f"Fraud capture      : "
        f"{result['fraud_capture_rate'] * 100:.2f}%"
    )

    print(
        f"Challenge precision: "
        f"{result['challenge_precision'] * 100:.2f}%"
    )

    print(
        f"Legitimate challenge: "
        f"{result['legitimate_challenge_rate'] * 100:.2f}%"
    )

    print(
        f"Actual loss avoided: "
        f"{result['actual_loss_avoided']:,.2f}"
    )

    print(
        f"Loss avoidance rate: "
        f"{result['loss_avoidance_rate'] * 100:.2f}%"
    )


policy_summary = pd.DataFrame(results)

summary_path = REPORTS / "v5_policy_sensitivity.csv"
policy_summary.to_csv(summary_path, index=False)


print("\n" + "=" * 88)
print("2. THRESHOLD CAPTURE CURVES")
print("=" * 88)

threshold_rows = []

for threshold in [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]:
    mask = calibrated_probability >= threshold

    count = int(mask.sum())
    fraud_count = int(y[mask].sum())

    threshold_rows.append(
        {
            "probability_threshold": threshold,
            "transactions": count,
            "transaction_rate": count / len(y),
            "fraud_count": fraud_count,
            "precision": (
                fraud_count / count if count else 0.0
            ),
            "fraud_capture": (
                fraud_count / y.sum() if y.sum() else 0.0
            ),
            "legitimate_challenge_rate": (
                int((1 - y[mask]).sum())
                / int((1 - y).sum())
                if count and (1 - y).sum()
                else 0.0
            ),
        }
    )

threshold_df = pd.DataFrame(threshold_rows)

threshold_path = REPORTS / "v5_probability_threshold_sensitivity.csv"
threshold_df.to_csv(threshold_path, index=False)


print(
    threshold_df.to_string(
        index=False,
        formatters={
            "transaction_rate": "{:.4f}".format,
            "precision": "{:.4f}".format,
            "fraud_capture": "{:.4f}".format,
            "legitimate_challenge_rate": "{:.4f}".format,
        },
    )
)


print("\n" + "=" * 88)
print("3. CALIBRATION SUPPORT")
print("=" * 88)

support_rows = []

for value in [0.0, 1.0]:
    mask = np.isclose(calibrated_probability, value)

    if not mask.any():
        continue

    support_rows.append(
        {
            "probability": value,
            "count": int(mask.sum()),
            "fraud_count": int(y[mask].sum()),
            "fraud_rate": float(y[mask].mean()),
            "mean_amount": float(amount[mask].mean()),
            "min_amount": float(amount[mask].min()),
            "max_amount": float(amount[mask].max()),
        }
    )

support_df = pd.DataFrame(support_rows)

support_path = REPORTS / "v5_calibration_extreme_support.csv"
support_df.to_csv(support_path, index=False)

print(support_df.to_string(index=False))


print("\n" + "=" * 88)
print("4. CALIBRATED PROBABILITY SUPPORT")
print("=" * 88)

quantiles = np.quantile(
    calibrated_probability,
    [
        0.00,
        0.01,
        0.05,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
        1.00,
    ],
)

quantile_df = pd.DataFrame(
    {
        "quantile": [
            "min",
            "p01",
            "p05",
            "p10",
            "p25",
            "p50",
            "p75",
            "p90",
            "p95",
            "p99",
            "max",
        ],
        "probability": quantiles,
    }
)

quantile_path = REPORTS / "v5_calibrated_probability_quantiles.csv"
quantile_df.to_csv(quantile_path, index=False)

print(quantile_df.to_string(index=False))


print("\n" + "=" * 88)
print("5. OUTPUTS")
print("=" * 88)

print(f"Policy sensitivity:")
print(f"  {summary_path}")

print(f"Probability threshold sensitivity:")
print(f"  {threshold_path}")

print(f"Calibration extreme support:")
print(f"  {support_path}")

print(f"Probability quantiles:")
print(f"  {quantile_path}")

summary = {
    "future_test_rows": len(df),
    "future_test_fraud": int(y.sum()),
    "future_test_fraud_rate": float(y.mean()),
    "policies": results,
    "calibration_extremes": support_rows,
}

json_path = REPORTS / "v5_policy_sensitivity_summary.json"

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"JSON summary:")
print(f"  {json_path}")

print("\nV5 POLICY SENSITIVITY COMPLETE")