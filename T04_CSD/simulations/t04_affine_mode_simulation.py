"""Starter simulation for affine mode separation in T04_CSD."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def weighted_mean(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(points * weights[:, None], axis=0) / np.sum(weights)


def main() -> None:
    grid_x = np.linspace(-1.0, 1.0, 21)
    grid_y = np.linspace(-1.0, 1.0, 21)
    xx, yy = np.meshgrid(grid_x, grid_y)
    base_points = np.column_stack([xx.ravel(), yy.ravel()])
    base_weights = np.exp(-0.5 * (1.5 * base_points[:, 0] ** 2 + 0.75 * base_points[:, 1] ** 2))

    transforms = [
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float),
        np.array([[1.1, 0.2], [0.0, 0.9]], dtype=float),
        np.array([[0.9, -0.25], [0.15, 1.15]], dtype=float),
    ]
    shifts = [
        np.array([0.0, 0.0], dtype=float),
        np.array([0.25, -0.15], dtype=float),
        np.array([-0.2, 0.3], dtype=float),
    ]

    frames = []
    base_center = weighted_mean(base_points, base_weights)
    for matrix, shift in zip(transforms, shifts, strict=True):
        pts = base_points @ matrix.T + shift
        center = weighted_mean(pts, base_weights)
        frames.append(
            {
                "matrix": matrix.tolist(),
                "shift": shift.tolist(),
                "center": center.tolist(),
                "center_shift_from_base": (center - base_center).tolist(),
            }
        )

    result = {
        "theory": "T04_CSD",
        "simulation": "affine_mode_separation",
        "base_center": base_center.tolist(),
        "frames": frames,
    }

    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "t04_affine_mode_simulation.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
