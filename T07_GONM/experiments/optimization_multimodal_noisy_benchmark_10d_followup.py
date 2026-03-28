from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import GONM.optimization_multimodal_noisy_benchmark_10d as bench10

OUT_DIR = ROOT / "GONM" / "results" / "benchmark_10d_followup"
SEEDS = [100 + 37 * i for i in range(6)]


def summarize_runs(rows):
    dists = np.array([r.distance_to_nearest_minimizer for r in rows], dtype=float)
    bests = np.array([r.best_true_value for r in rows], dtype=float)
    return {
        "best_true_mean": float(np.mean(bests)),
        "best_true_std": float(np.std(bests)),
        "dist_mean": float(np.mean(dists)),
        "dist_std": float(np.std(dists)),
        "dist_p25": float(np.quantile(dists, 0.25)),
        "dist_p50": float(np.quantile(dists, 0.50)),
        "dist_p75": float(np.quantile(dists, 0.75)),
    }


def hit_rate_at(rows, threshold: float) -> float:
    return float(np.mean([1.0 if r.distance_to_nearest_minimizer <= threshold else 0.0 for r in rows]))


def main() -> None:
    methods = {
        "sim_anneal": bench10.bench.run_simulated_annealing,
        "csd": bench10.bench.run_csd,
        "gonm_10d": bench10.run_gonm_10d,
    }
    budgets = [520, 1040]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summary_lines = ["# GONM 10D Follow-up", ""]

    for surface in bench10.SURFACES_10D:
        summary_lines.extend([f"## `{surface.name}`", ""])
        lo, hi = surface.bounds
        base_threshold = bench10.bench.basin_hit_threshold(surface)
        scaled_threshold = base_threshold * np.sqrt(surface.dim)
        loose_threshold = base_threshold * surface.dim
        for budget in budgets:
            summary_lines.append(f"### budget=`{budget}`")
            for method_name, method_fn in methods.items():
                rows = [method_fn(surface, seed=s, budget=budget, noise_std=0.18) for s in SEEDS]
                for row in rows:
                    row.method = method_name
                stats = summarize_runs(rows)
                summary_lines.append(
                    f"- `{method_name}`: best_true_mean=`{stats['best_true_mean']:.4e}`, "
                    f"dist_mean=`{stats['dist_mean']:.4e}`, dist_p50=`{stats['dist_p50']:.4e}`, "
                    f"hit_base=`{100.0 * hit_rate_at(rows, base_threshold):.1f}%`, "
                    f"hit_sqrtd=`{100.0 * hit_rate_at(rows, scaled_threshold):.1f}%`, "
                    f"hit_d=`{100.0 * hit_rate_at(rows, loose_threshold):.1f}%`"
                )
                all_rows.extend(
                    {
                        "surface": surface.name,
                        "budget": budget,
                        "method": method_name,
                        "summary": stats,
                        "hit_base": hit_rate_at(rows, base_threshold),
                        "hit_sqrtd": hit_rate_at(rows, scaled_threshold),
                        "hit_d": hit_rate_at(rows, loose_threshold),
                        "runs": [asdict(r) for r in rows],
                    }
                    for _ in [0]
                )
            summary_lines.append("")

    (OUT_DIR / "results.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    (OUT_DIR / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
