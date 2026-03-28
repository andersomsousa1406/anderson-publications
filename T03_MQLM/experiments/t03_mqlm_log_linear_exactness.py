"""Starter exactness test for the T03_MQLM theory."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def logarithmic_mean(a: float, b: float) -> float:
    if a <= 0.0 or b <= 0.0:
        raise ValueError("logarithmic mean requires positive inputs")
    if np.isclose(a, b):
        return float(a)
    return float((b - a) / (np.log(b) - np.log(a)))


def mqlm_interval_integral(x0: float, x1: float, f0: float, f1: float) -> float:
    return float(x1 - x0) * logarithmic_mean(f0, f1)


def exact_exponential_integral(k: float, x0: float, x1: float) -> float:
    if np.isclose(k, 0.0):
        return float(x1 - x0)
    return float((np.exp(k * x1) - np.exp(k * x0)) / k)


def main() -> None:
    x0, x1 = 0.0, 1.0
    ks = [0.0, 0.25, 0.5, 1.0, 2.0]
    rows = []

    for k in ks:
        f0 = float(np.exp(k * x0))
        f1 = float(np.exp(k * x1))
        mqlm = mqlm_interval_integral(x0, x1, f0, f1)
        exact = exact_exponential_integral(k, x0, x1)
        rows.append(
            {
                "k": k,
                "mqlm_integral": mqlm,
                "exact_integral": exact,
                "abs_error": abs(mqlm - exact),
            }
        )

    result = {
        "theory": "T03_MQLM",
        "test": "log_linear_exactness_on_exponential_family",
        "interval": [x0, x1],
        "rows": rows,
        "max_abs_error": max(row["abs_error"] for row in rows),
    }

    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "t03_mqlm_log_linear_exactness.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
