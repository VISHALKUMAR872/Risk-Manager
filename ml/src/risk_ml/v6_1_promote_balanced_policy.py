from __future__ import annotations

"""
Risk Sentinel â€” promote V6.1 Balanced policy.

This script updates the canonical risk-engine policy thresholds to the
selection-partition-validated V6.1 Balanced operating point.

It intentionally does NOT modify model/calibration artifacts.

Policy:
    VERIFY: p >= 0.275 OR expected_loss >= 50
    REVIEW: p >= 0.640 OR expected_loss >= 400
    HOLD:   p >= 0.640 OR expected_loss >= 700

The script:
1. locates the canonical PolicyEngine implementation
2. prints the existing policy
3. replaces only the policy version + six V6.1 threshold values
4. verifies monotonicity and exact expected values
5. writes a backup beside the source file
6. compiles the modified module

Run from repository root:
    uv run --directory .\services\api python .\src\risk_api\...
or simply execute with the ML environment if the path is correct.

IMPORTANT:
    This changes source code only. The service must be restarted for the
    running worker/API process to load the new policy.
"""

import argparse
import ast
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POLICY = ROOT / "risk_engine" / "src" / "risk_engine" / "policy.py"


def find_policy_file() -> Path:
    candidates = [
        ROOT / "risk_engine" / "src" / "risk_engine" / "policy.py",
        ROOT / "risk_engine" / "src" / "risk_engine" / "policy_engine.py",
    ]
    for path in candidates:
        if path.exists():
            return path

    matches = list(
        (ROOT / "risk_engine").rglob("policy*.py")
    )
    if matches:
        return matches[0]

    raise FileNotFoundError(
        "Could not locate PolicyEngine source under risk_engine/src/risk_engine."
    )


def extract(text: str, name: str) -> str:
    patterns = [
        rf'{re.escape(name)}\s*=\s*"([^"]+)"',
        rf"{re.escape(name)}\s*=\s*'([^']+)'",
        rf"{re.escape(name)}\s*=\s*([0-9]+(?:\.[0-9]+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return "<not found>"


def replace_required(text: str, name: str, value: str) -> str:
    patterns = [
        rf'({re.escape(name)}\s*=\s*)"[^"]*"',
        rf"({re.escape(name)}\s*=\s*)'[^']*'",
        rf"({re.escape(name)}\s*=\s*)[0-9]+(?:\.[0-9]+)?",
    ]
    for pattern in patterns:
        updated, count = re.subn(
            pattern,
            lambda m: m.group(1) + value,
            text,
            count=1,
        )
        if count:
            return updated

    raise RuntimeError(f"Could not locate required policy field: {name}")


def validate(text: str) -> dict:
    expected = {
        "VERSION": "policy-v6-balanced",
        "MEDIUM_PROBABILITY": "0.275",
        "HIGH_PROBABILITY": "0.640",
        "CRITICAL_PROBABILITY": "0.640",
        "MEDIUM_EXPECTED_LOSS": "50",
        "HIGH_EXPECTED_LOSS": "400",
        "CRITICAL_EXPECTED_LOSS": "700",
    }

    actual = {name: extract(text, name) for name in expected}

    if actual != expected:
        raise RuntimeError(
            "Policy validation failed.\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        )

    if not (
        float(actual["MEDIUM_PROBABILITY"])
        <= float(actual["HIGH_PROBABILITY"])
        <= float(actual["CRITICAL_PROBABILITY"])
    ):
        raise RuntimeError("Probability thresholds are not monotonic.")

    if not (
        float(actual["MEDIUM_EXPECTED_LOSS"])
        <= float(actual["HIGH_EXPECTED_LOSS"])
        <= float(actual["CRITICAL_EXPECTED_LOSS"])
    ):
        raise RuntimeError("Expected-loss thresholds are not monotonic.")

    return actual


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy-file",
        type=Path,
        default=None,
        help="Optional explicit PolicyEngine source path.",
    )
    args = parser.parse_args()

    policy_file = (
        args.policy_file.resolve()
        if args.policy_file
        else find_policy_file().resolve()
    )

    print("=" * 88)
    print("RISK SENTINEL â€” PROMOTE V6.1 BALANCED POLICY")
    print("=" * 88)
    print(f"Policy source: {policy_file}")
    print()

    if not policy_file.exists():
        raise FileNotFoundError(policy_file)

    original = policy_file.read_text(encoding="utf-8-sig")

    # Parse first so we never blindly edit a non-Python file.
    ast.parse(original)

    fields = [
        "VERSION",
        "MEDIUM_PROBABILITY",
        "HIGH_PROBABILITY",
        "CRITICAL_PROBABILITY",
        "MEDIUM_EXPECTED_LOSS",
        "HIGH_EXPECTED_LOSS",
        "CRITICAL_EXPECTED_LOSS",
    ]

    print("CURRENT POLICY")
    print("-" * 88)
    for field in fields:
        print(f"{field:28} {extract(original, field)}")
    print()

    updated = original
    replacements = {
        "VERSION": '"policy-v6-balanced"',
        "MEDIUM_PROBABILITY": "0.275",
        "HIGH_PROBABILITY": "0.640",
        "CRITICAL_PROBABILITY": "0.640",
        "MEDIUM_EXPECTED_LOSS": "50",
        "HIGH_EXPECTED_LOSS": "400",
        "CRITICAL_EXPECTED_LOSS": "700",
    }

    for field, value in replacements.items():
        updated = replace_required(updated, field, value)

    ast.parse(updated)
    actual = validate(updated)

    if updated == original:
        print("Policy is already exactly V6.1 Balanced.")
        print("No source change required.")
        return 0

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = policy_file.with_name(
        policy_file.stem + f".pre_v6_1_backup_{timestamp}.py"
    )
    shutil.copy2(policy_file, backup)

    policy_file.write_text(updated, encoding="utf-8")

    # Compile the exact file that was modified.
    result = subprocess.run(
        ["python", "-m", "py_compile", str(policy_file)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        # Restore immediately if syntax compilation fails.
        shutil.copy2(backup, policy_file)
        raise RuntimeError(
            "Modified policy did not compile; original restored.\n"
            + result.stderr
        )

    print("PROMOTED POLICY")
    print("-" * 88)
    for field, value in actual.items():
        print(f"{field:28} {value}")
    print()

    print(f"Backup: {backup}")
    print("Python compilation: PASS")
    print()
    print("=" * 88)
    print("V6.1 BALANCED POLICY SOURCE PROMOTION: PASS")
    print("=" * 88)
    print()
    print("Next required gate:")
    print("  Re-run model/calibration/policy parity against the canonical runtime.")
    print("Do not treat the running worker as updated until it has been restarted.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

