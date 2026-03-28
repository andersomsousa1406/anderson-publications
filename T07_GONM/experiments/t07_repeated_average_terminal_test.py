"""Starter terminal-resolution test for T07_GONM."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def estimate_ranking_error(delta: float, sigma: float, m: int, trials: int = 20000, seed: int = 7) -> float:
    rng = np.random.default_rng(seed)
    noise_x = rng.normal(0.0, sigma / np.sqrt(m), size=trials)
    noise_z = rng.normal(0.0, sigma / np.sqrt(m), size=trials)
    diff = delta + noise_x - noise_z
    return float(np.mean(diff < 0.0))


def main() -> None:
    delta = 0.05
    sigma = 0.2
    ms = [1, 2, 4, 8, 16, 32]
    rows = []

    for m in ms:
        rows.append(
            {
                "m": m,
                "estimated_ranking_error": estimate_ranking_error(delta, sigma, m),
            }
        )

    result = {
        "theory": "T07_GONM",
        "test": "repeated_average_terminal_ranking_error",
        "delta": delta,
        "sigma": sigma,
        "rows": rows,
    }

    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "t07_repeated_average_terminal_test.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
