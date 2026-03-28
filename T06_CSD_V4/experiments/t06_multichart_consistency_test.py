"""Starter multichart consistency test for T06_CSD_V4."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def chart_left(x: np.ndarray) -> np.ndarray:
    return x + np.array([-0.5, 0.0])


def chart_right(x: np.ndarray) -> np.ndarray:
    return x + np.array([0.5, 0.0])


def fuse(points: np.ndarray) -> np.ndarray:
    return np.mean(points, axis=0)


def main() -> None:
    anchor = np.array([0.25, -0.1], dtype=float)
    left_repr = chart_left(anchor)
    right_repr = chart_right(anchor)
    fused = fuse(np.vstack([left_repr - np.array([-0.5, 0.0]), right_repr - np.array([0.5, 0.0])]))

    result = {
        "theory": "T06_CSD_V4",
        "test": "multichart_fusion_consistency",
        "anchor": anchor.tolist(),
        "left_chart_representation": left_repr.tolist(),
        "right_chart_representation": right_repr.tolist(),
        "fused_reconstruction": fused.tolist(),
        "fusion_error_norm": float(np.linalg.norm(fused - anchor)),
    }

    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "t06_multichart_consistency_test.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
