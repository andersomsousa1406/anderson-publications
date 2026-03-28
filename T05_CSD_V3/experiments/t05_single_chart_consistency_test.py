"""Starter single-chart consistency test for T05_CSD_V3."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def chart_map(x: np.ndarray, alpha: float) -> np.ndarray:
    """Simple nonlinear single-chart map used as a validation starter."""
    return x + alpha * np.tanh(x)


def chart_inverse(z: np.ndarray, alpha: float, n_iter: int = 30) -> np.ndarray:
    """Fixed-point inversion for the starter nonlinear chart."""
    x = np.array(z, dtype=float)
    for _ in range(n_iter):
        x = z - alpha * np.tanh(x)
    return x


def main() -> None:
    alpha = 0.35
    xs = np.linspace(-1.5, 1.5, 101)
    points = np.column_stack([xs, 0.5 * xs])
    mapped = chart_map(points, alpha)
    recovered = chart_inverse(mapped, alpha)
    errors = np.linalg.norm(recovered - points, axis=1)

    result = {
        "theory": "T05_CSD_V3",
        "test": "single_chart_forward_inverse_consistency",
        "alpha": alpha,
        "max_recovery_error": float(np.max(errors)),
        "mean_recovery_error": float(np.mean(errors)),
    }

    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "t05_single_chart_consistency_test.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
