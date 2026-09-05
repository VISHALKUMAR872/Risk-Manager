from __future__ import annotations

"""
Risk Sentinel V6.1 — four-action economic policy optimizer.

Selection/evaluation protocol
------------------------------
- Policy thresholds are selected ONLY on a calibration/selection partition.
- The untouched future-test partition is evaluated exactly once after selection.
- The V5 CatBoost model and V5 isotonic calibrator are reused unchanged.
- The policy has six monotone thresholds:
    verify_probability <= review_probability <= hold_probability
    verify_expected_loss <= review_expected_loss <= hold_expected_loss
- Decision is driven by probability OR expected-loss triggers.
- Economic value uses the same illustrative cost model as V6.

Outputs
-------
ml/artifacts/reports/v6_1_policy_optimization.json
ml/artifacts/reports/v6_1_policy_optimization.csv
ml/artifacts/reports/v6_1_selected_policies.json
ml/artifacts/reports/v6_1_frontier.csv
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

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

DEFAULT_SELECTION_NAMES = [
    "online_v4_calibration.parquet",
    "online_v5_calibration.parquet",
    "online_v4_calib.parquet",
    "online_v5_calib.parquet",
]

DEFAULT_FP_COSTS = {
    "VERIFY": 20.0,
    "REVIEW": 75.0,
    "HOLD": 150.0,
}

# These are intentionally modest deterministic grids. Coordinate descent
# evaluates ~6 * N candidates per pass rather than a combinatorial 6-D grid.
PROBABILITY_GRID = np.unique(
    np.concatenate(
        [
            np.array(
                [
                    0.02,
                    0.05,
                    0.075,
                    0.10,
                    0.116,
                    0.125,
                    0.15,
                    0.175,
                    0.20,
                    0.225,
                    0.25,
                    0.275,
                    0.30,
                    0.35,
                    0.40,
                    0.50,
                    0.60,
                    0.70,
                    0.80,
                    0.90,
                ]
            ),
            np.linspace(0.0, 1.0, 51),
        ]
    )
)

EXPECTED_LOSS_GRID = np.unique(
    np.concatenate(
        [
            np.array(
                [
                    0.0,
                    25.0,
                    50.0,
                    75.0,
                    100.0,
                    125.0,
                    150.0,
                    200.0,
                    250.0,
                    300.0,
                    400.0,
                    500.0,
                    700.0,
                    1000.0,
                    1500.0,
                    2000.0,
                    3000.0,
                ]
            ),
            np.linspace(0.0, 2000.0, 41),
        ]
    )
)

# Friction budgets used for the three selected operating modes.
FPR_BUDGETS = {
    "low_friction": 0.0025,
    "balanced": 0.0100,
    "max_protection": 1.0,
}


def discover_selection_path(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = ARTIFACTS / path
        if not path.exists():
            raise FileNotFoundError(
                f"Selection/calibration partition not found: {path}"
            )
        return path

    for name in DEFAULT_SELECTION_NAMES:
        path = ARTIFACTS / name
        if path.exists():
            return path

    candidates = sorted(
        path
        for path in ARTIFACTS.glob("*.parquet")
        if "test" not in path.name.lower()
        and (
            "calib" in path.name.lower()
            or "calibration" in path.name.lower()
            or "selection" in path.name.lower()
        )
    )

    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        "Could not discover a calibration/selection parquet. "
        "Pass --selection-path <file>."
    )


def load_scores(
    path: Path,
    model: CatBoostClassifier,
    calibrator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_parquet(path)

    required = FEATURES + [TARGET, "amount"]
    missing = [
        column for column in required if column not in df.columns
    ]
    if missing:
        raise RuntimeError(
            f"{path} is missing required columns: {missing}"
        )

    y = df[TARGET].astype(int).to_numpy()
    amount = (
        pd.to_numeric(df["amount"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    raw = model.predict_proba(df[FEATURES])[:, 1]
    probability = np.asarray(
        calibrator.predict(raw),
        dtype=float,
    )
    probability = np.clip(probability, 0.0, 1.0)

    expected_loss = probability * amount * LGF
    return y, amount, expected_loss


def apply_policy(
    probability: np.ndarray,
    expected_loss: np.ndarray,
    thresholds: dict[str, float],
) -> np.ndarray:
    decision = np.full(
        probability.shape[0],
        "APPROVE",
        dtype=object,
    )

    verify = (
        (probability >= thresholds["verify_probability"])
        | (expected_loss >= thresholds["verify_expected_loss"])
    )
    review = (
        (probability >= thresholds["review_probability"])
        | (expected_loss >= thresholds["review_expected_loss"])
    )
    hold = (
        (probability >= thresholds["hold_probability"])
        | (expected_loss >= thresholds["hold_expected_loss"])
    )

    decision[verify] = "VERIFY"
    decision[review] = "REVIEW"
    decision[hold] = "HOLD"
    return decision


def evaluate(
    y: np.ndarray,
    amount: np.ndarray,
    probability: np.ndarray,
    expected_loss: np.ndarray,
    thresholds: dict[str, float],
    fp_costs: dict[str, float],
    fraud_interception_rate: float,
) -> dict[str, float | int]:
    decision = apply_policy(
        probability,
        expected_loss,
        thresholds,
    )

    fraud = y == 1
    legitimate = ~fraud
    challenged = decision != "APPROVE"

    tp = int((challenged & fraud).sum())
    fp = int((challenged & legitimate).sum())
    fn = int((~challenged & fraud).sum())
    tn = int((~challenged & legitimate).sum())

    n = len(y)
    fraud_total = int(fraud.sum())
    legitimate_total = int(legitimate.sum())
    intervention_count = tp + fp

    precision = (
        tp / intervention_count
        if intervention_count
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

    baseline_fraud_loss = float(
        amount[fraud].sum() * LGF
    )
    loss_avoided = float(
        amount[challenged & fraud].sum()
        * LGF
        * fraud_interception_rate
    )

    intervention_cost = sum(
        float(((decision == level) & y.astype(bool)).sum()) * cost
        for level, cost in fp_costs.items()
    )
    legitimate_intervention_cost = sum(
        float(((decision == level) & legitimate).sum()) * cost
        for level, cost in fp_costs.items()
    )
    fraud_intervention_cost = sum(
        float(((decision == level) & fraud).sum()) * cost
        for level, cost in fp_costs.items()
    )

    # Keep the name explicit: this is the full intervention cost.
    total_intervention_cost = (
        legitimate_intervention_cost
        + fraud_intervention_cost
    )

    return {
        **thresholds,
        "n_total": n,
        "n_fraud": fraud_total,
        "n_legitimate": legitimate_total,
        "approve_count": int((decision == "APPROVE").sum()),
        "verify_count": int((decision == "VERIFY").sum()),
        "review_count": int((decision == "REVIEW").sum()),
        "hold_count": int((decision == "HOLD").sum()),
        "intervention_count": intervention_count,
        "intervention_rate": intervention_count / n if n else 0.0,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "gross_fraud_value": float(amount[fraud].sum()),
        "baseline_fraud_loss": baseline_fraud_loss,
        "loss_avoided": loss_avoided,
        "loss_avoidance_rate": (
            loss_avoided / baseline_fraud_loss
            if baseline_fraud_loss
            else 0.0
        ),
        "legitimate_intervention_cost": legitimate_intervention_cost,
        "fraud_intervention_cost": fraud_intervention_cost,
        "total_intervention_cost": total_intervention_cost,
        "net_economic_value": (
            loss_avoided - total_intervention_cost
        ),
    }


def objective_value(
    result: dict[str, float | int],
    mode: str,
    fpr_budget: float,
) -> float:
    if float(result["fpr"]) > fpr_budget + 1e-12:
        return -1e30

    if mode == "net_value":
        return float(result["net_economic_value"])

    if mode == "loss_avoidance":
        return float(result["loss_avoided"])

    if mode == "recall":
        return float(result["recall"])

    raise ValueError(mode)


def better(
    candidate: dict[str, float | int],
    incumbent: dict[str, float | int] | None,
    mode: str,
    fpr_budget: float,
) -> bool:
    if float(candidate["fpr"]) > fpr_budget + 1e-12:
        return False

    if incumbent is None:
        return True

    c = objective_value(
        candidate,
        mode,
        fpr_budget,
    )
    i = objective_value(
        incumbent,
        mode,
        fpr_budget,
    )

    if c > i + 1e-9:
        return True

    if abs(c - i) <= 1e-9:
        # Tie-break toward lower friction, then higher precision.
        if float(candidate["intervention_rate"]) < float(
            incumbent["intervention_rate"]
        ) - 1e-12:
            return True
        if (
            abs(
                float(candidate["intervention_rate"])
                - float(incumbent["intervention_rate"])
            )
            <= 1e-12
            and float(candidate["precision"])
            > float(incumbent["precision"])
        ):
            return True

    return False


def coordinate_optimize(
    y: np.ndarray,
    amount: np.ndarray,
    probability: np.ndarray,
    expected_loss: np.ndarray,
    start: dict[str, float],
    mode: str,
    fpr_budget: float,
    fp_costs: dict[str, float],
    fraud_interception_rate: float,
    passes: int = 4,
) -> dict[str, float | int]:
    current = dict(start)

    parameter_order = [
        "verify_probability",
        "review_probability",
        "hold_probability",
        "verify_expected_loss",
        "review_expected_loss",
        "hold_expected_loss",
    ]

    grids = {
        "verify_probability": PROBABILITY_GRID,
        "review_probability": PROBABILITY_GRID,
        "hold_probability": PROBABILITY_GRID,
        "verify_expected_loss": EXPECTED_LOSS_GRID,
        "review_expected_loss": EXPECTED_LOSS_GRID,
        "hold_expected_loss": EXPECTED_LOSS_GRID,
    }

    for _ in range(passes):
        changed = False

        for parameter in parameter_order:
            incumbent_result = evaluate(
                y,
                amount,
                probability,
                expected_loss,
                current,
                fp_costs,
                fraud_interception_rate,
            )

            best_result = (
                incumbent_result
                if better(
                    incumbent_result,
                    None,
                    mode,
                    fpr_budget,
                )
                else None
            )

            for value in grids[parameter]:
                candidate = dict(current)
                candidate[parameter] = float(value)

                if (
                    candidate["verify_probability"] >= candidate["review_probability"]                   or candidate["review_probability"] >= candidate["hold_probability"]
                    or candidate["verify_expected_loss"] >= candidate["review_expected_loss"]                 or candidate["review_expected_loss"] >= candidate["hold_expected_loss"]
                ):
                    continue

                result = evaluate(
                    y,
                    amount,
                    probability,
                    expected_loss,
                    candidate,
                    fp_costs,
                    fraud_interception_rate,
                )

                if better(
                    result,
                    best_result,
                    mode,
                    fpr_budget,
                ):
                    best_result = result

            if best_result is not None:
                new_value = float(best_result[parameter])
                if abs(new_value - current[parameter]) > 1e-12:
                    changed = True
                current[parameter] = new_value

        if not changed:
            break

    final = evaluate(
        y,
        amount,
        probability,
        expected_loss,
        current,
        fp_costs,
        fraud_interception_rate,
    )
    return final


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Risk Sentinel V6.1 four-action economic policy optimizer."
    )
    parser.add_argument(
        "--selection-path",
        default=None,
        help=(
            "Calibration/selection parquet. If omitted, discover a "
            "*calibration* or *selection* parquet under ml/artifacts."
        ),
    )
    parser.add_argument(
        "--verify-cost",
        type=float,
        default=DEFAULT_FP_COSTS["VERIFY"],
    )
    parser.add_argument(
        "--review-cost",
        type=float,
        default=DEFAULT_FP_COSTS["REVIEW"],
    )
    parser.add_argument(
        "--hold-cost",
        type=float,
        default=DEFAULT_FP_COSTS["HOLD"],
    )
    parser.add_argument(
        "--fraud-interception-rate",
        type=float,
        default=1.0,
    )

    args = parser.parse_args()

    if not 0.0 <= args.fraud_interception_rate <= 1.0:
        raise ValueError(
            "--fraud-interception-rate must be in [0,1]"
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

    selection_path = discover_selection_path(
        args.selection_path
    )

    if not TEST_PATH.exists():
        raise FileNotFoundError(TEST_PATH)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)
    if not CALIBRATOR_PATH.exists():
        raise FileNotFoundError(CALIBRATOR_PATH)

    print("=" * 88)
    print("RISK SENTINEL — V6.1 FOUR-ACTION ECONOMIC POLICY OPTIMIZER")
    print("=" * 88)
    print(f"\nSelection partition: {selection_path}")

    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))
    calibrator = joblib.load(CALIBRATOR_PATH)

    print("Scoring calibration/selection partition...")
    y_sel, amount_sel, el_sel = load_scores(
        selection_path,
        model,
        calibrator,
    )

    # Reconstruct probability from expected_loss and amount is unsafe at zero
    # amount, so score the selection dataframe separately.
    selection_df = pd.read_parquet(selection_path)
    raw_sel = model.predict_proba(selection_df[FEATURES])[:, 1]
    p_sel = np.asarray(
        calibrator.predict(raw_sel),
        dtype=float,
    )
    p_sel = np.clip(p_sel, 0.0, 1.0)

    print(f"Selection rows   : {len(y_sel):,}")
    print(f"Selection fraud  : {int(y_sel.sum()):,}")
    print(f"Selection FPR base: {1.0 - y_sel.mean():.6%}")

    # Starting point = current locked Balanced policy.
    start = {
        "verify_probability": 0.25,
        "review_probability": 0.30,
        "hold_probability": 0.60,
        "verify_expected_loss": 100.0,
        "review_expected_loss": 300.0,
        "hold_expected_loss": 700.0,
    }

    selected: dict[str, dict[str, float | int]] = {}

    mode_specs = {
        "low_friction": (
            "net_value",
            FPR_BUDGETS["low_friction"],
        ),
        "balanced": (
            "net_value",
            FPR_BUDGETS["balanced"],
        ),
        "max_protection": (
            "loss_avoidance",
            FPR_BUDGETS["max_protection"],
        ),
        "max_recall_under_1pct_fpr": (
            "recall",
            0.01,
        ),
    }

    print("\n" + "=" * 88)
    print("CALIBRATION-PARTITION POLICY SEARCH")
    print("=" * 88)

    for name, (mode, budget) in mode_specs.items():
        result = coordinate_optimize(
            y_sel,
            amount_sel,
            p_sel,
            el_sel,
            start,
            mode,
            budget,
            fp_costs,
            args.fraud_interception_rate,
        )
        selected[name] = result

        print(f"\n{name.upper()}")
        print("-" * 60)
        print(
            "Thresholds: "
            f"p={result['verify_probability']:.3f}/"
            f"{result['review_probability']:.3f}/"
            f"{result['hold_probability']:.3f}, "
            f"EL=₹{result['verify_expected_loss']:.0f}/"
            f"{result['review_expected_loss']:.0f}/"
            f"{result['hold_expected_loss']:.0f}"
        )
        print(
            f"Intervention : {result['intervention_rate']:.3%}"
        )
        print(
            f"Precision    : {result['precision']:.2%}"
        )
        print(
            f"Recall       : {result['recall']:.2%}"
        )
        print(
            f"FPR          : {result['fpr']:.2%}"
        )
        print(
            f"Loss avoided : ₹{result['loss_avoided']:,.2f}"
        )
        print(
            f"Total cost   : ₹{result['total_intervention_cost']:,.2f}"
        )
        print(
            f"Net value    : ₹{result['net_economic_value']:,.2f}"
        )

    # Evaluate the selected policies ONCE on the untouched future test.
    print("\n" + "=" * 88)
    print("FINAL EVALUATION — UNTOUCHED FUTURE TEST")
    print("=" * 88)

    y_test, amount_test, el_test = load_scores(
        TEST_PATH,
        model,
        calibrator,
    )
    test_df = pd.read_parquet(TEST_PATH)
    raw_test = model.predict_proba(test_df[FEATURES])[:, 1]
    p_test = np.asarray(
        calibrator.predict(raw_test),
        dtype=float,
    )
    p_test = np.clip(p_test, 0.0, 1.0)

    roc_auc = float(
        roc_auc_score(y_test, p_test)
    )
    pr_auc = float(
        average_precision_score(y_test, p_test)
    )

    future_rows = []
    for name, selection_result in selected.items():
        thresholds = {
            key: float(selection_result[key])
            for key in [
                "verify_probability",
                "review_probability",
                "hold_probability",
                "verify_expected_loss",
                "review_expected_loss",
                "hold_expected_loss",
            ]
        }

        result = evaluate(
            y_test,
            amount_test,
            p_test,
            el_test,
            thresholds,
            fp_costs,
            args.fraud_interception_rate,
        )
        result["policy"] = name
        result["selection_fpr_budget"] = (
            FPR_BUDGETS[
                "max_protection"
                if name == "max_protection"
                else (
                    "low_friction"
                    if name == "low_friction"
                    else "balanced"
                    if name == "balanced"
                    else "balanced"
                )
            ]
            if name != "max_recall_under_1pct_fpr"
            else 0.01
        )
        future_rows.append(result)

        print(f"\n{name.upper()}")
        print("-" * 60)
        print(
            "Thresholds: "
            f"p={result['verify_probability']:.3f}/"
            f"{result['review_probability']:.3f}/"
            f"{result['hold_probability']:.3f}, "
            f"EL=₹{result['verify_expected_loss']:.0f}/"
            f"{result['review_expected_loss']:.0f}/"
            f"{result['hold_expected_loss']:.0f}"
        )
        print(
            f"TP/FP/TN/FN : "
            f"{result['tp']}/{result['fp']}/"
            f"{result['tn']}/{result['fn']}"
        )
        print(
            f"Precision    : {result['precision']:.2%}"
        )
        print(
            f"Recall       : {result['recall']:.2%}"
        )
        print(
            f"FPR          : {result['fpr']:.2%}"
        )
        print(
            f"Intervention : {result['intervention_rate']:.2%}"
        )
        print(
            f"Loss avoided : ₹{result['loss_avoided']:,.2f}"
        )
        print(
            f"Total cost   : ₹{result['total_intervention_cost']:,.2f}"
        )
        print(
            f"Net value    : ₹{result['net_economic_value']:,.2f}"
        )

    future_df = pd.DataFrame(future_rows)

    # A compact frontier sorted by actual future-test net value is useful
    # for presentation, but the test set is NOT used to choose thresholds.
    frontier = future_df[
        [
            "policy",
            "verify_probability",
            "review_probability",
            "hold_probability",
            "verify_expected_loss",
            "review_expected_loss",
            "hold_expected_loss",
            "intervention_rate",
            "precision",
            "recall",
            "fpr",
            "loss_avoided",
            "total_intervention_cost",
            "net_economic_value",
            "tp",
            "fp",
            "tn",
            "fn",
        ]
    ].sort_values(
        "net_economic_value",
        ascending=False,
    )

    policy_csv = (
        REPORTS /
        "v6_1_policy_optimization.csv"
    )
    frontier_csv = (
        REPORTS /
        "v6_1_frontier.csv"
    )
    selected_json = (
        REPORTS /
        "v6_1_selected_policies.json"
    )
    summary_json = (
        REPORTS /
        "v6_1_policy_optimization.json"
    )

    future_df.to_csv(
        policy_csv,
        index=False,
    )
    frontier.to_csv(
        frontier_csv,
        index=False,
    )

    selected_json.write_text(
        json.dumps(
            selected,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = {
        "version": "v6.1",
        "selection_partition": str(
            selection_path
        ),
        "future_test_partition": str(
            TEST_PATH
        ),
        "selection_rows": int(len(y_sel)),
        "future_test_rows": int(len(y_test)),
        "future_test_fraud_rows": int(y_test.sum()),
        "future_test_roc_auc": roc_auc,
        "future_test_pr_auc": pr_auc,
        "lgf": LGF,
        "illustrative_costs_inr": fp_costs,
        "fraud_interception_rate_assumption": (
            args.fraud_interception_rate
        ),
        "method": {
            "policy_search": "deterministic coordinate descent",
            "thresholds_selected_on": "calibration partition only",
            "future_test_used_for_selection": False,
            "policy": "probability OR expected-loss triggers",
            "monotonicity_constraints": True,
        },
        "selected_policies": selected,
        "future_test_results": future_rows,
        "artifacts": {
            "policy_csv": str(policy_csv),
            "frontier_csv": str(frontier_csv),
            "selected_json": str(selected_json),
        },
    }

    summary_json.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 88)
    print("V6.1 COMPLETE")
    print("=" * 88)
    print(f"Summary : {summary_json}")
    print(f"Policies: {policy_csv}")
    print(f"Frontier: {frontier_csv}")
    print(f"Selected: {selected_json}")
    print(
        "\nIMPORTANT: thresholds were selected on the calibration "
        "partition only; the 88,581-row future test is evaluation-only."
    )


if __name__ == "__main__":
    main()

