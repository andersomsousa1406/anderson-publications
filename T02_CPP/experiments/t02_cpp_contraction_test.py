"""Starter contraction test for the T02_CPP theory."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def contractive_step(x: np.ndarray, alpha: float) -> np.ndarray:
    """Simple diagonal contractive map used as a CPP starter test."""
    return alpha * x


def main() -> None:
    alpha = 0.82
    x0 = np.array([1.0, -0.5, 0.25], dtype=float)
    delta0 = np.array([1.0e-2, -5.0e-3, 2.5e-3], dtype=float)

    base = x0.copy()
    perturbed = x0 + delta0
    norms = [float(np.linalg.norm(delta0))]

    for _ in range(20):
        base = contractive_step(base, alpha)
        perturbed = contractive_step(perturbed, alpha)
        norms.append(float(np.linalg.norm(perturbed - base)))

    ratios = [norms[i] / norms[i - 1] for i in range(1, len(norms))]

    result = {
        "theory": "T02_CPP",
        "test": "contractive_propagation_linear_map",
        "alpha": alpha,
        "initial_state": x0.tolist(),
        "initial_perturbation": delta0.tolist(),
        "norm_sequence": norms,
        "ratio_sequence": ratios,
        "mean_ratio_tail": float(np.mean(ratios[-5:])),
        "is_contractive": bool(max(ratios) < 1.0),
    }

    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "t02_cpp_contraction_test.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
