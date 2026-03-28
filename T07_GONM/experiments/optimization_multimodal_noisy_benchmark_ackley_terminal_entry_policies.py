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

OUT_DIR = ROOT / "GONM" / "results" / "ackley_terminal_entry_policies"


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


def make_policy(name: str, mode: str):
    original = bench.averaged_terminal_pattern_search

    def choose_anchor(oracle: bench.NoisyOracle, x0: np.ndarray) -> np.ndarray:
        x0 = np.asarray(x0, dtype=float)
        lo, hi = oracle.surface.bounds
        candidates = [x0]
        if mode in {"radial70", "bundle_radial", "bundle_orbital"}:
            for scale in (0.85, 0.70, 0.55, 0.40):
                candidates.append(bench.clip_to_bounds(scale * x0, oracle.surface.bounds))
        if mode == "bundle_orbital":
            radius = float(np.linalg.norm(x0))
            if radius > 1e-12:
                u = x0 / radius
                ortho = np.array([u[1], -u[0]], dtype=float)
                for scale in (0.70, 0.55):
                    base = scale * x0
                    tangent = 0.18 * radius * ortho
                    candidates.append(bench.clip_to_bounds(base + tangent, oracle.surface.bounds))
                    candidates.append(bench.clip_to_bounds(base - tangent, oracle.surface.bounds))
        best = candidates[0]
        best_score = oracle.averaged(best, 9)
        for cand in candidates[1:]:
            score = oracle.averaged(cand, 9)
            if score < best_score:
                best = cand
                best_score = score
        return best

    def variant(
        oracle: bench.NoisyOracle,
        x0: np.ndarray,
        budget: int,
        n_samples: int = 9,
        max_iters: int = 24,
    ) -> tuple[np.ndarray, float, float]:
        anchor = choose_anchor(oracle, x0)
        return original(oracle, anchor, budget, n_samples=n_samples, max_iters=max_iters)

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
        ("cpp_mqlm_csd_entry_radial70", make_policy("cpp_mqlm_csd_entry_radial70", "radial70")),
        ("cpp_mqlm_csd_entry_bundle_radial", make_policy("cpp_mqlm_csd_entry_bundle_radial", "bundle_radial")),
        ("cpp_mqlm_csd_entry_bundle_orbital", make_policy("cpp_mqlm_csd_entry_bundle_orbital", "bundle_orbital")),
    ]

    rows: list[dict[str, object]] = []
    for name, runner in methods:
        runs = [runner(surface, seed=100 + 37 * seed, budget=420, noise_std=0.18) for seed in range(8)]
        rows.append({"method": name, "summary": summarize_runs(runs), "runs": [asdict(r) for r in runs]})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = ["# Ackley Terminal Entry Policies", ""]
    for row in rows:
        s = row["summary"]
        lines.append(
            f"- `{row['method']}`: best_true_mean=`{s['best_true_mean']:.4e}`, dist_mean=`{s['dist_mean']:.4e}`, hit_rate=`{100.0 * s['hit_rate']:.1f}%`, evals_mean=`{s['evals_mean']:.1f}`"
        )
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
