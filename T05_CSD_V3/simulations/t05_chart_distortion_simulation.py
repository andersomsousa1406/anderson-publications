"""Starter distortion simulation for the single-chart regime of T05_CSD_V3."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def chart_map(x: np.ndarray, alpha: float) -> np.ndarray:
    return x + alpha * np.tanh(x)


def jacobian_diag(x: np.ndarray, alpha: float) -> np.ndarray:
    return 1.0 + alpha * (1.0 - np.tanh(x) ** 2)


def main() -> None:
    alpha_values = [0.1, 0.25, 0.5, 0.75]
    xs = np.linspace(-2.0, 2.0, 400)
    rows = []

    for alpha in alpha_values:
        mapped = chart_map(xs, alpha)
        deriv = jacobian_diag(xs, alpha)
        rows.append(
            {
                "alpha": alpha,
                "mapped_span": float(mapped.max() - mapped.min()),
                "max_local_distortion": float(np.max(np.abs(deriv - 1.0))),
                "mean_local_distortion": float(np.mean(np.abs(deriv - 1.0))),
            }
        )

    result = {
        "theory": "T05_CSD_V3",
        "simulation": "single_chart_distortion_profile",
        "rows": rows,
    }

    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "t05_chart_distortion_simulation.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
