from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import optimization_multimodal_noisy_benchmark as bench

OUT_DIR = ROOT / "GONM" / "results" / "ackley_terminal_calibration"
SEEDS = list(range(8))


def summarize_runs(runs: list[bench.RunResult]) -> dict[str, float]:
    return {
        "best_true_mean": float(np.mean([r.best_true_value for r in runs])),
        "best_true_std": float(np.std([r.best_true_value for r in runs])),
        "dist_mean": float(np.mean([r.distance_to_nearest_minimizer for r in runs])),
        "dist_std": float(np.std([r.distance_to_nearest_minimizer for r in runs])),
        "hit_rate": float(np.mean([1.0 if r.final_hit_global_basin else 0.0 for r in runs])),
        "evals_mean": float(np.mean([r.evals for r in runs])),
        "runtime_ms_mean": float(np.mean([r.runtime_ms for r in runs])),
    }


def make_variant(name: str, n_samples: int, max_iters: int, use_diagonals: bool):
    original = bench.averaged_terminal_pattern_search

    def variant(
        oracle: bench.NoisyOracle,
        x0: np.ndarray,
        budget: int,
        n_samples_override: int = 9,
        max_iters_override: int = 24,
    ) -> tuple[np.ndarray, float, float]:
        x = np.asarray(x0, dtype=float).copy()
        lo, hi = oracle.surface.bounds
        step = 0.06 * (hi - lo) / 10.0
        current_noisy = oracle.averaged(x, n_samples)
        best_x = x.copy()
        best_true = oracle.true(x)
        best_noisy = current_noisy
        directions = [
            np.array([1.0, 0.0], dtype=float),
            np.array([-1.0, 0.0], dtype=float),
            np.array([0.0, 1.0], dtype=float),
            np.array([0.0, -1.0], dtype=float),
        ]
        if use_diagonals:
            s = float(np.sqrt(0.5))
            directions.extend(
                [
                    np.array([s, s], dtype=float),
                    np.array([-s, s], dtype=float),
                    np.array([s, -s], dtype=float),
                    np.array([-s, -s], dtype=float),
                ]
            )

        iter_count = 0
        while oracle.evals < budget and iter_count < max_iters and step > 1e-4:
            iter_count += 1
            improved = False
            for direction in directions:
                if oracle.evals >= budget:
                    break
                proposal = bench.clip_to_bounds(x + step * direction, oracle.surface.bounds)
                proposal_noisy = oracle.averaged(proposal, n_samples)
                if proposal_noisy < current_noisy:
                    x = proposal
                    current_noisy = proposal_noisy
                    improved = True
                    true_val = oracle.true(x)
                    if true_val < best_true:
                        best_x = x.copy()
                        best_true = true_val
                        best_noisy = current_noisy
                    break
            if improved:
                step *= 1.05
            else:
                step *= 0.5
        return best_x, best_true, best_noisy

    def run(surface: bench.Surface, seed: int, budget: int = 420, noise_std: float = 0.18) -> bench.RunResult:
        bench.averaged_terminal_pattern_search = variant
        try:
            result = bench.run_cpp_mqlm_csd(surface, seed, budget=budget, noise_std=noise_std)
            result.method = name
            return result
        finally:
            bench.averaged_terminal_pattern_search = original

    return run


def main() -> None:
    surface = next(s for s in bench.SURFACES if s.name == "ackley_2d")
    methods = [
        ("sim_anneal", bench.run_simulated_annealing),
        ("cpp_mqlm_csd_current", bench.run_cpp_mqlm_csd),
        ("cpp_mqlm_csd_avg11", make_variant("cpp_mqlm_csd_avg11", 11, 24, False)),
        ("cpp_mqlm_csd_diag9", make_variant("cpp_mqlm_csd_diag9", 9, 24, True)),
        ("cpp_mqlm_csd_diag11_long", make_variant("cpp_mqlm_csd_diag11_long", 11, 32, True)),
    ]

    rows: list[dict[str, object]] = []
    for name, runner in methods:
        runs = [runner(surface, seed=100 + 37 * seed, budget=420, noise_std=0.18) for seed in range(8)]
        rows.append({"method": name, "summary": summarize_runs(runs), "runs": [asdict(r) for r in runs]})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = ["# Ackley Terminal Calibration", ""]
    for row in rows:
        s = row["summary"]
        lines.append(
            f"- `{row['method']}`: best_true_mean=`{s['best_true_mean']:.4e}`, dist_mean=`{s['dist_mean']:.4e}`, hit_rate=`{100.0 * s['hit_rate']:.1f}%`, evals_mean=`{s['evals_mean']:.1f}`"
        )
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
