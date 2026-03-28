from __future__ import annotations

import importlib
import json
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import optimization_multimodal_noisy_benchmark as base_mod

OUT_DIR = ROOT / "GONM" / "results" / "ackley_budget_policies"
GEN_DIR = ROOT / "GONM" / "generated_budget_variants"
if str(GEN_DIR) not in sys.path:
    sys.path.append(str(GEN_DIR))
SEEDS = [100 + 37 * i for i in range(8)]
BASE_PATH = ROOT / "optimization_multimodal_noisy_benchmark.py"

REPLACEMENTS = {
    "ackley_reserve_128": [
        (
            '    final_gauge_reserve = 96 if surface.name == "ackley_2d" else 28',
            '    final_gauge_reserve = 128 if surface.name == "ackley_2d" else 28',
        ),
    ],
    "ackley_reserve_160": [
        (
            '    final_gauge_reserve = 96 if surface.name == "ackley_2d" else 28',
            '    final_gauge_reserve = 160 if surface.name == "ackley_2d" else 28',
        ),
    ],
    "ackley_reserve_128_phase1_40": [
        (
            '    final_gauge_reserve = 96 if surface.name == "ackley_2d" else 28',
            '    final_gauge_reserve = 128 if surface.name == "ackley_2d" else 28',
        ),
        (
            '    phase1_budget = min(int(0.48 * budget), max(0, budget - final_gauge_reserve - 120))',
            '    phase1_budget = min(int((0.40 if surface.name == "ackley_2d" else 0.48) * budget), max(0, budget - final_gauge_reserve - 120))',
        ),
    ],
    "ackley_reserve_160_phase1_36": [
        (
            '    final_gauge_reserve = 96 if surface.name == "ackley_2d" else 28',
            '    final_gauge_reserve = 160 if surface.name == "ackley_2d" else 28',
        ),
        (
            '    phase1_budget = min(int(0.48 * budget), max(0, budget - final_gauge_reserve - 120))',
            '    phase1_budget = min(int((0.36 if surface.name == "ackley_2d" else 0.48) * budget), max(0, budget - final_gauge_reserve - 120))',
        ),
    ],
}


def summarize_runs(runs):
    return {
        "best_true_mean": float(np.mean([r.best_true_value for r in runs])),
        "best_true_std": float(np.std([r.best_true_value for r in runs])),
        "dist_mean": float(np.mean([r.distance_to_nearest_minimizer for r in runs])),
        "dist_std": float(np.std([r.distance_to_nearest_minimizer for r in runs])),
        "hit_rate": float(np.mean([1.0 if r.final_hit_global_basin else 0.0 for r in runs])),
        "evals_mean": float(np.mean([r.evals for r in runs])),
        "runtime_ms_mean": float(np.mean([r.runtime_ms for r in runs])),
    }


def main() -> None:
    base_source = BASE_PATH.read_text(encoding="utf-8")
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    surface = next(s for s in base_mod.SURFACES if s.name == "ackley_2d")
    sim_runs = [base_mod.run_simulated_annealing(surface, seed=s) for s in SEEDS]
    current_runs = [base_mod.run_cpp_mqlm_csd(surface, seed=s) for s in SEEDS]
    rows.append({"method": "sim_anneal", "summary": summarize_runs(sim_runs), "runs": [asdict(r) for r in sim_runs]})
    rows.append({"method": "cpp_mqlm_csd_current", "summary": summarize_runs(current_runs), "runs": [asdict(r) for r in current_runs]})

    for name, replacements in REPLACEMENTS.items():
        variant_source = base_source
        for old, new in replacements:
            variant_source = variant_source.replace(old, new)
        variant_path = GEN_DIR / f"{name}.py"
        variant_path.write_text(variant_source, encoding="utf-8")
        importlib.invalidate_caches()
        if name in sys.modules:
            del sys.modules[name]
        mod = importlib.import_module(name)
        surface = next(s for s in mod.SURFACES if s.name == "ackley_2d")
        runs = [mod.run_cpp_mqlm_csd(surface, seed=s) for s in SEEDS]
        rows.append({"method": name, "summary": summarize_runs(runs), "runs": [asdict(r) for r in runs]})

    (OUT_DIR / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = ["# Ackley Budget Policies", ""]
    for row in rows:
        s = row["summary"]
        lines.append(
            f"- `{row['method']}`: best_true_mean=`{s['best_true_mean']:.4e}`, dist_mean=`{s['dist_mean']:.4e}`, hit_rate=`{100.0 * s['hit_rate']:.1f}%`, evals_mean=`{s['evals_mean']:.1f}`"
        )
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
