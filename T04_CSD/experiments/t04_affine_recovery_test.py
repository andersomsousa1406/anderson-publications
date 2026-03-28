"""Starter affine-recovery test for the T04_CSD theory."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def weighted_mean(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(points * weights[:, None], axis=0) / np.sum(weights)


def weighted_covariance(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    center = weighted_mean(points, weights)
    centered = points - center
    return (centered.T * weights) @ centered / np.sum(weights)


def main() -> None:
    grid_x = np.linspace(-1.0, 1.0, 25)
    grid_y = np.linspace(-1.0, 1.0, 25)
    xx, yy = np.meshgrid(grid_x, grid_y)
    base_points = np.column_stack([xx.ravel(), yy.ravel()])
    base_weights = np.exp(-0.5 * (base_points[:, 0] ** 2 + 2.0 * base_points[:, 1] ** 2))

    A = np.array([[1.2, 0.35], [0.0, 0.85]], dtype=float)
    b = np.array([0.4, -0.3], dtype=float)
    transformed_points = base_points @ A.T + b
    transformed_weights = base_weights.copy()

    recovered_translation = weighted_mean(transformed_points, transformed_weights) - (
        weighted_mean(base_points @ A.T, transformed_weights)
    )

    base_cov = weighted_covariance(base_points, base_weights)
    transformed_cov = weighted_covariance(transformed_points, transformed_weights)
    predicted_cov = A @ base_cov @ A.T

    result = {
        "theory": "T04_CSD",
        "test": "affine_translation_and_covariance_recovery",
        "applied_translation": b.tolist(),
        "recovered_translation": recovered_translation.tolist(),
        "translation_error_norm": float(np.linalg.norm(recovered_translation - b)),
        "covariance_error_norm": float(np.linalg.norm(transformed_cov - predicted_cov)),
    }

    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "t04_affine_recovery_test.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
