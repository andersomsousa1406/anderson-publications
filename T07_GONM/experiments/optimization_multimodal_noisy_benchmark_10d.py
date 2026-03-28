from __future__ import annotations

import json
from dataclasses import asdict
from math import pi
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import optimization_multimodal_noisy_benchmark as bench

DIM = 10
OUT_DIR = ROOT / "GONM" / "results" / "benchmark_10d"
SEEDS = [100 + 37 * i for i in range(6)]


def rastrigin_10d(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(10.0 * len(x) + np.sum(x * x - 10.0 * np.cos(2.0 * pi * x)))


def ackley_10d(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    a = 20.0
    b = 0.2
    c = 2.0 * pi
    s1 = np.mean(x * x)
    s2 = np.mean(np.cos(c * x))
    return float(-a * np.exp(-b * np.sqrt(s1)) - np.exp(s2) + a + np.e)


def gaussian_mix_10d(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    centers = [
        np.array([-1.8, -1.2, 1.1, -0.7, 0.8, -1.4, 1.3, -0.9, 0.6, -1.1], dtype=float),
        np.array([1.6, 1.3, -1.0, 0.9, -0.7, 1.4, -1.1, 0.8, -0.6, 1.0], dtype=float),
        np.array([0.2, -1.9, 1.7, -1.5, 1.2, -0.3, 1.5, -1.1, 0.4, -0.8], dtype=float),
    ]
    amps = [1.00, 0.96, 0.92]
    sigmas = [0.75, 0.82, 0.88]
    total = 0.0
    for center, amp, sigma in zip(centers, amps, sigmas):
        r2 = float(np.sum((x - center) ** 2))
        total += amp * np.exp(-r2 / (2.0 * sigma * sigma))
    trend = 0.035 * float(np.sum(x * x))
    return float(trend - total)


SURFACES_10D = [
    bench.Surface("rastrigin_10d", (-5.12, 5.12), DIM, rastrigin_10d, 0.0, [tuple([0.0] * DIM)]),
    bench.Surface("ackley_10d", (-5.0, 5.0), DIM, ackley_10d, 0.0, [tuple([0.0] * DIM)]),
    bench.Surface(
        "gaussian_mix_10d",
        (-4.0, 4.0),
        DIM,
        gaussian_mix_10d,
        -1.0,
        [tuple([-1.8, -1.2, 1.1, -0.7, 0.8, -1.4, 1.3, -0.9, 0.6, -1.1])],
    ),
]


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


def cpp_local_refine(
    oracle: bench.NoisyOracle,
    x0: np.ndarray,
    budget: int,
    max_iters: int,
) -> tuple[np.ndarray, float, float]:
    x = np.asarray(x0, dtype=float).copy()
    lo, hi = oracle.surface.bounds
    eta = 0.18 * (hi - lo) / 10.0
    alpha = 0.40
    h = 0.06 * (hi - lo) / 10.0
    current_score = oracle.averaged(x, 3)
    best_x = x.copy()
    best_true = oracle.true(x)
    best_noisy = current_score
    for _ in range(max_iters):
        if oracle.evals >= budget:
            break
        grad = bench.finite_difference_grad(oracle, x, h, samples=1)
        axis_sens = bench.local_axis_sensitivity(oracle, x, h)
        normalized_grad = grad / axis_sens
        proposal = bench.clip_to_bounds(x - eta * normalized_grad, oracle.surface.bounds)
        x_new = bench.clip_to_bounds((1.0 - alpha) * x + alpha * proposal, oracle.surface.bounds)
        new_score = oracle.averaged(x_new, 3)
        if new_score <= current_score:
            x = x_new
            current_score = new_score
            eta *= 1.02
        else:
            eta *= 0.74
            alpha *= 0.98
        true_val = oracle.true(x)
        if true_val < best_true:
            best_x = x.copy()
            best_true = true_val
            best_noisy = current_score
    return best_x, best_true, best_noisy


def run_gonm_10d(surface: bench.Surface, seed: int, budget: int = 520, noise_std: float = 0.18) -> bench.RunResult:
    rng = np.random.default_rng(seed)
    oracle = bench.NoisyOracle(surface, noise_std, rng)
    lo, hi = surface.bounds
    population = rng.uniform(lo, hi, size=(36, surface.dim))
    best_x = population[0].copy()
    best_true = oracle.true(best_x)
    best_noisy = oracle.averaged(best_x, 2)
    start = perf_counter()

    phase1_budget = int(0.52 * budget)
    while oracle.evals < phase1_budget:
        noisy_vals = np.array([oracle.averaged(x, 2) for x in population], dtype=float)
        order = np.argsort(noisy_vals)
        elites = population[order[:12]]
        elite_scores = noisy_vals[order[:12]]
        labels = bench.simple_kmeans(elites, k=3, rng=rng)
        centers = []
        scores = []
        for j in range(3):
            mask = labels == j
            if not np.any(mask):
                continue
            centers.append(np.mean(elites[mask], axis=0))
            scores.append(float(np.mean(elite_scores[mask])))
        centers, scores = bench.merge_close_components(centers, scores, tol=0.18 * (hi - lo))
        ranked = sorted(zip(centers, scores), key=lambda item: item[1])[:3]
        if ranked:
            probe = ranked[0][0]
            probe_true = oracle.true(probe)
            if probe_true < best_true:
                best_x = probe.copy()
                best_true = probe_true
                best_noisy = oracle.averaged(probe, 2)
        new_population = []
        for center, score in ranked:
            spread = 0.10 * (hi - lo) * (0.8 + min(max(score, 0.0), 2.0) * 0.0)
            block = center + rng.normal(scale=spread, size=(10, surface.dim))
            new_population.append(bench.clip_to_bounds(block, surface.bounds))
        new_population.append(rng.uniform(lo, hi, size=(max(6, len(population) // 4), surface.dim)))
        population = np.vstack(new_population)[: len(population)]

    candidate_centers = [best_x.copy()]
    noisy_vals = np.array([oracle.averaged(x, 2) for x in population], dtype=float)
    order = np.argsort(noisy_vals)[:3]
    for idx in order:
        bench.append_unique_candidate(candidate_centers, population[idx], tol=0.10 * (hi - lo))

    for center in candidate_centers:
        if oracle.evals >= budget:
            break
        ref_x, ref_true, ref_noisy = cpp_local_refine(oracle, center, budget=budget, max_iters=14)
        if ref_true < best_true:
            best_x = ref_x.copy()
            best_true = ref_true
            best_noisy = ref_noisy

    runtime_ms = 1000.0 * (perf_counter() - start)
    best_dist = bench.distance_to_minimizers(best_x, surface.global_minimizers)
    return bench.RunResult(
        surface=surface.name,
        method="gonm_10d",
        seed=seed,
        best_true_value=best_true,
        best_noisy_value=best_noisy,
        distance_to_nearest_minimizer=best_dist,
        final_hit_global_basin=best_dist <= bench.basin_hit_threshold(surface),
        evals=oracle.evals,
        runtime_ms=runtime_ms,
    )


def main() -> None:
    methods = {
        "sim_anneal": bench.run_simulated_annealing,
        "cpp": bench.run_cpp,
        "csd": bench.run_csd,
        "gonm_10d": run_gonm_10d,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[bench.RunResult] = []
    summary_lines = ["# GONM 10D Benchmark", ""]

    for surface in SURFACES_10D:
        summary_lines.extend([f"## `{surface.name}`", ""])
        surface_rows: list[bench.RunResult] = []
        start = perf_counter()
        for method_name, method_fn in methods.items():
            for seed in SEEDS:
                result = method_fn(surface, seed=seed, budget=520, noise_std=0.18)
                result.method = method_name
                surface_rows.append(result)
        rows.extend(surface_rows)
        grouped: dict[str, list[bench.RunResult]] = {}
        for row in surface_rows:
            grouped.setdefault(row.method, []).append(row)
        best_method = None
        best_value = float("inf")
        for method_name in ("sim_anneal", "cpp", "csd", "gonm_10d"):
            stats = summarize_runs(grouped[method_name])
            if stats["best_true_mean"] < best_value:
                best_value = stats["best_true_mean"]
                best_method = method_name
            summary_lines.append(
                f"- `{method_name}`: best_true_mean=`{stats['best_true_mean']:.4e}`, "
                f"dist_mean=`{stats['dist_mean']:.4e}`, "
                f"hit_rate=`{100.0 * stats['hit_rate']:.1f}%`, "
                f"runtime_ms_mean=`{stats['runtime_ms_mean']:.1f}`, "
                f"evals_mean=`{stats['evals_mean']:.1f}`"
            )
        elapsed = 1000.0 * (perf_counter() - start)
        summary_lines.append(f"- melhor metodo medio: `{best_method}`")
        summary_lines.append(f"- tempo total da superficie: `{elapsed:.1f} ms`")
        summary_lines.append("")

    (OUT_DIR / "results.json").write_text(json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8")
    (OUT_DIR / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
