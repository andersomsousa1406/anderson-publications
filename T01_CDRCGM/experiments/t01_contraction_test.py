"""Numerical checks for the transformed contraction map of T01_CDRCGM."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def transformed_map(y: np.ndarray) -> np.ndarray:
    return (1.0 - np.sqrt(1.0 - y)) / 2.0


def transformed_map_derivative(y: np.ndarray) -> np.ndarray:
    return 1.0 / (4.0 * np.sqrt(1.0 - y))


def main() -> None:
    y0 = (2.0 - np.sqrt(2.0)) / 4.0
    grid = np.linspace(0.0, y0, 2001)
    deriv = transformed_map_derivative(grid)
    lipschitz = float(np.max(deriv))

    orbit = [float(y0)]
    for _ in range(12):
        orbit.append(float(transformed_map(np.array([orbit[-1]]))[0]))

    result = {
        "theory": "T01_CDRCGM",
        "test": "contraction_factor_on_invariant_interval",
        "interval": [0.0, float(y0)],
        "max_derivative": lipschitz,
        "is_contractive": bool(lipschitz < 1.0),
        "first_orbit_values": orbit,
    }

    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "t01_contraction_test.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
