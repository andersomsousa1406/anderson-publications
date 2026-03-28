"""Starter transition simulation from V3-style to V4-style structural handling."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def transition_score(overlap: float, distortion: float) -> float:
    return float(overlap + distortion)


def main() -> None:
    overlaps = [0.05, 0.1, 0.2, 0.35, 0.5]
    distortions = [0.1, 0.2, 0.35, 0.5, 0.7]
    threshold = 0.65
    rows = []

    for overlap, distortion in zip(overlaps, distortions, strict=True):
        score = transition_score(overlap, distortion)
        rows.append(
            {
                "overlap": overlap,
                "distortion": distortion,
                "transition_score": score,
                "activate_v4": bool(score >= threshold),
            }
        )

    result = {
        "theory": "T06_CSD_V4",
        "simulation": "v3_to_v4_transition_profile",
        "threshold": threshold,
        "rows": rows,
    }

    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "t06_transition_simulation.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
