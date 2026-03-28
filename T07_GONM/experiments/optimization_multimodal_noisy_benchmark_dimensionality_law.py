from __future__ import annotations

import json
from dataclasses import asdict
from math import pi, sqrt
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import optimization_multimodal_noisy_benchmark as bench

OUT_DIR = ROOT / "GONM" / "results" / "dimensionality_law"
SEEDS = [100 + 37 * i for i in range(6)]
ACKLEY_DIMS = [2, 4, 8, 10]
BUDGETS = [520, 1040]
PHASE1_RATIOS = [0.52, 0.68, 0.80]
NOISE_STD = 0.18


def ackley_nd(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    a = 20.0
    b = 0.2
    c = 2.0 * pi
    s1 = np.mean(x * x)
    s2 = np.mean(np.cos(c * x))
    return float(-a * np.exp(-b * np.sqrt(s1)) - np.exp(s2) + a + np.e)


def make_ackley_surface(dim: int) -> bench.Surface:
    return bench.Surface(
        f"ackley_{dim}d",
        (-5.0, 5.0),
        dim,
        ackley_nd,
        0.0,
        [tuple([0.0] * dim)],
    )


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
    h = 0.06 * (hi - lo) / max(10.0, float(oracle.surface.dim))
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


def run_gonm_nd(
    surface: bench.Surface,
    seed: int,
    budget: int,
    noise_std: float,
    phase1_ratio: float,
) -> bench.RunResult:
    rng = np.random.default_rng(seed)
    oracle = bench.NoisyOracle(surface, noise_std, rng)
    lo, hi = surface.bounds
    population = rng.uniform(lo, hi, size=(36, surface.dim))
    best_x = population[0].copy()
    best_true = oracle.true(best_x)
    best_noisy = oracle.averaged(best_x, 2)
    start = perf_counter()

    phase1_budget = int(phase1_ratio * budget)
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
        for center, _score in ranked:
            spread = 0.10 * (hi - lo)
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
        method=f"gonm_p{int(round(100 * phase1_ratio))}",
        seed=seed,
        best_true_value=best_true,
        best_noisy_value=best_noisy,
        distance_to_nearest_minimizer=best_dist,
        final_hit_global_basin=best_dist <= bench.basin_hit_threshold(surface),
        evals=oracle.evals,
        runtime_ms=runtime_ms,
    )


def make_initial_probe(surface: bench.Surface, seed: int) -> np.ndarray:
    rng = np.random.default_rng(100000 + 97 * seed + 13 * surface.dim)
    lo, hi = surface.bounds
    return rng.uniform(lo, hi, size=surface.dim)


def residual_ratio(best_true: float, initial_true: float, optimum: float) -> float:
    denom = max(initial_true - optimum, 1e-12)
    numer = max(best_true - optimum, 0.0)
    return float(numer / denom)


def summarize(rows: list[dict]) -> dict[str, float]:
    bests = np.array([r["best_true_value"] for r in rows], dtype=float)
    dists = np.array([r["distance_to_nearest_minimizer"] for r in rows], dtype=float)
    residuals = np.array([r["residual_ratio"] for r in rows], dtype=float)
    return {
        "best_true_mean": float(np.mean(bests)),
        "dist_mean": float(np.mean(dists)),
        "dist_std": float(np.std(dists)),
        "residual_ratio_mean": float(np.mean(residuals)),
        "residual_ratio_std": float(np.std(residuals)),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_lines = ["# GONM Dimensionality Law", ""]
    results = {
        "phase1_sweep_10d": [],
        "scaling": [],
    }

    summary_lines.extend(["## 10D Phase-1 Sweep", ""])
    surface_10d = make_ackley_surface(10)
    for budget in BUDGETS:
        summary_lines.append(f"### `ackley_10d`, budget=`{budget}`")
        for ratio in PHASE1_RATIOS:
            rows = []
            for seed in SEEDS:
                result = run_gonm_nd(surface_10d, seed=seed, budget=budget, noise_std=NOISE_STD, phase1_ratio=ratio)
                probe = make_initial_probe(surface_10d, seed)
                initial_true = surface_10d.func(probe)
                rows.append(
                    {
                        **asdict(result),
                        "phase1_ratio": ratio,
                        "initial_true_value": initial_true,
                        "residual_ratio": residual_ratio(result.best_true_value, initial_true, surface_10d.global_min_value),
                    }
                )
            stats = summarize(rows)
            results["phase1_sweep_10d"].append(
                {
                    "surface": surface_10d.name,
                    "budget": budget,
                    "phase1_ratio": ratio,
                    "summary": stats,
                    "runs": rows,
                }
            )
            summary_lines.append(
                f"- `gonm_p{int(round(100 * ratio))}`: "
                f"best_true_mean=`{stats['best_true_mean']:.4e}`, "
                f"dist_mean=`{stats['dist_mean']:.4e}`, "
                f"residual_ratio_mean=`{stats['residual_ratio_mean']:.4e}`"
            )
        summary_lines.append("")

    summary_lines.extend(["## Ackley Scaling vs Dimension", ""])
    methods = {
        "sim_anneal": bench.run_simulated_annealing,
        "csd": bench.run_csd,
        "gonm_p80": lambda surface, seed, budget, noise_std: run_gonm_nd(
            surface, seed=seed, budget=budget, noise_std=noise_std, phase1_ratio=0.80
        ),
    }
    for budget in BUDGETS:
        summary_lines.append(f"### budget=`{budget}`")
        for dim in ACKLEY_DIMS:
            surface = make_ackley_surface(dim)
            summary_lines.append(f"- dim=`{dim}`")
            for method_name, method_fn in methods.items():
                rows = []
                for seed in SEEDS:
                    result = method_fn(surface, seed=seed, budget=budget, noise_std=NOISE_STD)
                    result.method = method_name
                    probe = make_initial_probe(surface, seed)
                    initial_true = surface.func(probe)
                    rows.append(
                        {
                            **asdict(result),
                            "initial_true_value": initial_true,
                            "residual_ratio": residual_ratio(result.best_true_value, initial_true, surface.global_min_value),
                        }
                    )
                stats = summarize(rows)
                wall_coeff = stats["best_true_mean"] / sqrt(dim)
                residual_wall_coeff = stats["residual_ratio_mean"] / sqrt(dim)
                results["scaling"].append(
                    {
                        "budget": budget,
                        "dim": dim,
                        "method": method_name,
                        "summary": stats,
                        "wall_coeff": wall_coeff,
                        "residual_wall_coeff": residual_wall_coeff,
                    }
                )
                summary_lines.append(
                    f"  - `{method_name}`: "
                    f"best_true_mean=`{stats['best_true_mean']:.4e}`, "
                    f"residual_ratio_mean=`{stats['residual_ratio_mean']:.4e}`, "
                    f"wall_coeff=`{wall_coeff:.4e}`, "
                    f"residual_wall_coeff=`{residual_wall_coeff:.4e}`"
                )
        summary_lines.append("")

    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (OUT_DIR / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
