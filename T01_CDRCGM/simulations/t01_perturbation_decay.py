"""Starter simulation for perturbation decay in T01_CDRCGM."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def transformed_map(y: np.ndarray) -> np.ndarray:
    return (1.0 - np.sqrt(1.0 - y)) / 2.0


def main() -> None:
    y0 = (2.0 - np.sqrt(2.0)) / 4.0
    delta0 = 1.0e-3
    steps = 20

    base = [y0]
    perturbed = [y0 + delta0]
    deltas = [delta0]

    for _ in range(steps):
        base.append(float(transformed_map(np.array([base[-1]]))[0]))
        perturbed.append(float(transformed_map(np.array([perturbed[-1]]))[0]))
        deltas.append(abs(perturbed[-1] - base[-1]))

    ratios = []
    for i in range(1, len(deltas)):
        prev = deltas[i - 1]
        ratios.append(float(deltas[i] / prev) if prev > 0.0 else 0.0)

    result = {
        "theory": "T01_CDRCGM",
        "simulation": "perturbation_decay",
        "initial_value": float(y0),
        "initial_perturbation": delta0,
        "steps": steps,
        "delta_sequence": deltas,
        "ratio_sequence": ratios,
        "tail_mean_ratio": float(np.mean(ratios[-5:])),
    }

    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "t01_perturbation_decay.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
