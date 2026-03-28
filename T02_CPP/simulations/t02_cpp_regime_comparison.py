"""Starter simulation comparing contractive and expansive perturbation propagation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def iterate(alpha: float, steps: int = 20) -> dict[str, list[float]]:
    x0 = np.array([1.0, -0.5], dtype=float)
    delta0 = np.array([1.0e-2, -7.5e-3], dtype=float)
    base = x0.copy()
    perturbed = x0 + delta0
    norms = [float(np.linalg.norm(delta0))]

    for _ in range(steps):
        base = alpha * base
        perturbed = alpha * perturbed
        norms.append(float(np.linalg.norm(perturbed - base)))

    ratios = [norms[i] / norms[i - 1] for i in range(1, len(norms))]
    return {"norm_sequence": norms, "ratio_sequence": ratios}


def main() -> None:
    contractive = iterate(alpha=0.78)
    expansive = iterate(alpha=1.05)

    result = {
        "theory": "T02_CPP",
        "simulation": "contractive_vs_expansive_regime",
        "contractive_alpha": 0.78,
        "expansive_alpha": 1.05,
        "contractive": contractive,
        "expansive": expansive,
        "contractive_tail_mean": float(np.mean(contractive["ratio_sequence"][-5:])),
        "expansive_tail_mean": float(np.mean(expansive["ratio_sequence"][-5:])),
    }

    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "t02_cpp_regime_comparison.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
