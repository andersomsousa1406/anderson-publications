"""Starter simulation comparing differential and ranking SNR in T07_GONM."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def differential_snr(grad_scale: float, h: float, sigma: float, m: int) -> float:
    return float(np.sqrt(m) * h * grad_scale / sigma)


def ranking_snr(delta: float, sigma: float, m: int) -> float:
    return float(np.sqrt(m) * delta / sigma)


def main() -> None:
    sigma = 0.2
    m = 8
    delta = 0.05
    hs = [1.0e-1, 5.0e-2, 2.5e-2, 1.25e-2, 6.25e-3]
    grad_scales = [5.0e-2, 2.5e-2, 1.25e-2, 6.25e-3, 3.125e-3]

    rows = []
    for h, g in zip(hs, grad_scales, strict=True):
        rows.append(
            {
                "h": h,
                "grad_scale": g,
                "differential_snr": differential_snr(g, h, sigma, m),
                "ranking_snr": ranking_snr(delta, sigma, m),
            }
        )

    result = {
        "theory": "T07_GONM",
        "simulation": "resolution_barrier_profile",
        "sigma": sigma,
        "m": m,
        "delta": delta,
        "rows": rows,
    }

    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "t07_resolution_barrier_simulation.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
