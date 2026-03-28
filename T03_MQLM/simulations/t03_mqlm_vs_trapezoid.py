"""Starter simulation comparing MQLM and trapezoidal quadrature on positive data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def logarithmic_mean(a: float, b: float) -> float:
    if np.isclose(a, b):
        return float(a)
    return float((b - a) / (np.log(b) - np.log(a)))


def mqlm_integral(xs: np.ndarray, ys: np.ndarray) -> float:
    total = 0.0
    for i in range(len(xs) - 1):
        total += (xs[i + 1] - xs[i]) * logarithmic_mean(float(ys[i]), float(ys[i + 1]))
    return float(total)


def trapezoid_integral(xs: np.ndarray, ys: np.ndarray) -> float:
    return float(np.trapezoid(ys, xs))


def exact_integral() -> float:
    # Integral of exp(x^2) on [0, 1] has no elementary primitive, so use a dense reference grid.
    grid = np.linspace(0.0, 1.0, 200001)
    vals = np.exp(grid * grid)
    return float(np.trapezoid(vals, grid))


def main() -> None:
    reference = exact_integral()
    ns = [4, 8, 16, 32, 64]
    rows = []

    for n in ns:
        xs = np.linspace(0.0, 1.0, n + 1)
        ys = np.exp(xs * xs)
        mqlm = mqlm_integral(xs, ys)
        trap = trapezoid_integral(xs, ys)
        rows.append(
            {
                "subintervals": n,
                "mqlm_error": abs(mqlm - reference),
                "trapezoid_error": abs(trap - reference),
            }
        )

    result = {
        "theory": "T03_MQLM",
        "simulation": "mqlm_vs_trapezoid_on_exp_x2",
        "reference_integral": reference,
        "rows": rows,
    }

    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "t03_mqlm_vs_trapezoid.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
