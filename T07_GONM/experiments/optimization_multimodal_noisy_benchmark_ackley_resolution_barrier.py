from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import optimization_multimodal_noisy_benchmark as bench


OUT_DIR = ROOT / "GONM" / "results" / "ackley_resolution_barrier"
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


def run_noise_sweep(surface: bench.Surface) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for noise_std in [0.0, 0.03, 0.06, 0.09, 0.12, 0.18, 0.27]:
        for name, runner in [
            ("cpp_mqlm_csd", bench.run_cpp_mqlm_csd),
            ("sim_anneal", bench.run_simulated_annealing),
        ]:
            runs = [runner(surface, seed, budget=420, noise_std=noise_std) for seed in SEEDS]
            rows.append(
                {
                    "experiment": "noise_sweep",
                    "method": name,
                    "noise_std": noise_std,
                    "summary": summarize_runs(runs),
                }
            )
    return rows


def run_eps_sweep(surface: bench.Surface) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    original = bench.finite_difference_grad
    try:
        for h_scale in [0.25, 0.5, 1.0, 2.0, 4.0]:
            def scaled_grad(oracle: bench.NoisyOracle, x: np.ndarray, h: float, samples: int = 3, *, _scale: float = h_scale) -> np.ndarray:
                return original(oracle, x, h * _scale, samples=samples)

            bench.finite_difference_grad = scaled_grad
            runs = [bench.run_cpp_mqlm_csd(surface, seed, budget=420, noise_std=0.18) for seed in SEEDS]
            rows.append(
                {
                    "experiment": "eps_sweep",
                    "method": "cpp_mqlm_csd",
                    "h_scale": h_scale,
                    "summary": summarize_runs(runs),
                }
            )
    finally:
        bench.finite_difference_grad = original
    return rows


def infer_ackley_radius(target_value: float) -> float:
    lo = 0.0
    hi = 5.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        val = bench.ackley(np.array([mid, 0.0], dtype=float))
        if val < target_value:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def signal_floor_diagnostic(target_value: float, noise_std: float) -> dict[str, object]:
    radius = infer_ackley_radius(target_value)
    base = bench.ackley(np.array([radius, 0.0], dtype=float))
    deltas = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
    drops: list[dict[str, float]] = []
    for delta in deltas:
        inner = bench.ackley(np.array([max(radius - delta, 0.0), 0.0], dtype=float))
        drop = base - inner
        drops.append(
            {
                "delta_r": delta,
                "deterministic_drop": float(drop),
                "signal_to_noise_single_eval": float(drop / max(noise_std, 1e-12)),
                "signal_to_noise_avg3": float(drop / max(noise_std / np.sqrt(3.0), 1e-12)),
                "signal_to_noise_avg5": float(drop / max(noise_std / np.sqrt(5.0), 1e-12)),
            }
        )

    grad_rows: list[dict[str, float]] = []
    for h in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]:
        fp = bench.ackley(np.array([radius + h, 0.0], dtype=float))
        fm = bench.ackley(np.array([max(radius - h, 0.0), 0.0], dtype=float))
        grad_mag = abs((fp - fm) / (2.0 * h))
        grad_rows.append({"h": h, "estimated_radial_grad": float(grad_mag)})

    return {
        "experiment": "signal_floor",
        "target_best_true_value": target_value,
        "inferred_radius": radius,
        "deterministic_value_at_radius": float(base),
        "noise_std": noise_std,
        "drops": drops,
        "radial_gradient_estimates": grad_rows,
    }


def write_outputs(rows: list[dict[str, object]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Ackley Resolution Barrier")
    lines.append("")
    lines.append("## Noise Sweep")
    for row in [r for r in rows if r["experiment"] == "noise_sweep"]:
        summary = row["summary"]
        lines.append(
            f"- `{row['method']}` noise=`{row['noise_std']:.2f}`: "
            f"best_true_mean=`{summary['best_true_mean']:.4e}`, "
            f"dist_mean=`{summary['dist_mean']:.4e}`, "
            f"hit_rate=`{100.0 * summary['hit_rate']:.1f}%`"
        )

    lines.append("")
    lines.append("## Eps Sweep")
    for row in [r for r in rows if r["experiment"] == "eps_sweep"]:
        summary = row["summary"]
        lines.append(
            f"- `h_scale={row['h_scale']:.2f}`: "
            f"best_true_mean=`{summary['best_true_mean']:.4e}`, "
            f"dist_mean=`{summary['dist_mean']:.4e}`, "
            f"hit_rate=`{100.0 * summary['hit_rate']:.1f}%`"
        )

    lines.append("")
    lines.append("## Signal Floor")
    signal_row = next(row for row in rows if row["experiment"] == "signal_floor")
    lines.append(
        f"- inferred radius for target `best_true={signal_row['target_best_true_value']:.4e}`: "
        f"`r={signal_row['inferred_radius']:.4e}`"
    )
    for drop in signal_row["drops"]:
        lines.append(
            f"- `delta_r={drop['delta_r']:.1e}` -> "
            f"drop=`{drop['deterministic_drop']:.4e}`, "
            f"SNR1=`{drop['signal_to_noise_single_eval']:.4e}`, "
            f"SNR3=`{drop['signal_to_noise_avg3']:.4e}`, "
            f"SNR5=`{drop['signal_to_noise_avg5']:.4e}`"
        )

    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    surface = next(s for s in bench.SURFACES if s.name == "ackley_2d")
    rows: list[dict[str, object]] = []
    rows.extend(run_noise_sweep(surface))
    rows.extend(run_eps_sweep(surface))
    rows.append(signal_floor_diagnostic(target_value=0.29898, noise_std=0.18))
    write_outputs(rows)


if __name__ == "__main__":
    main()
