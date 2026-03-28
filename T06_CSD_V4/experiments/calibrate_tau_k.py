from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
V3_ROOT = ROOT / "V3"
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

from gauge_comparison import SCENARIOS, simulate
from quadratic_functional import QuadraticFunctionalWeights
from v4_solver import solve_v4_mixture, set_scenario_tau_k


@dataclass
class TauCalibrationRow:
    scenario: str
    tau_k: float
    target_k: int
    mean_best_error: float
    mean_target_error: float
    k_distribution: dict[int, int]


def reconstruction_error(final: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((final - pred) ** 2, axis=1))))


def best_trial_by_k(
    initial: np.ndarray,
    final: np.ndarray,
    weights: QuadraticFunctionalWeights,
    k: int,
    n_restarts: int = 4,
    max_iter: int = 12,
    min_component_size: int = 12,
    seed: int = 1234,
):
    best = None
    for restart in range(n_restarts):
        trial = solve_v4_mixture(
            initial=initial,
            final=final,
            n_components=k,
            weights_v3=weights,
            max_iter=max_iter,
            min_component_size=min_component_size,
            seed=seed + 1009 * restart + 7919 * k,
        )
        err = reconstruction_error(final, trial.prediction)
        if best is None or err < best[1]:
            best = (trial, err)
    if best is None:
        raise RuntimeError("No V4 trial produced during tau calibration.")
    return best


def calibrate_scenario_tau(
    scenario: dict[str, object],
    weights: QuadraticFunctionalWeights,
    n_seeds: int = 12,
    k_candidates: tuple[int, ...] = (2, 3, 4),
    competitiveness_cap: float = 0.05,
) -> TauCalibrationRow:
    tau_needs: dict[int, list[float]] = {k: [] for k in k_candidates}
    mean_errors: dict[int, list[float]] = {k: [] for k in k_candidates}

    times = np.linspace(0.0, 10.0, 180)
    for seed in range(n_seeds):
        rng = np.random.default_rng(9100 + 101 * seed)
        initial = scenario["builder"](rng)
        final = simulate(initial, times, scenario["velocity"])[-1]

        rec_by_k: dict[int, float] = {}
        for k in k_candidates:
            trial, err = best_trial_by_k(initial, final, weights, k=k, seed=8000 + seed)
            rec_by_k[k] = err
            mean_errors[k].append(err)

        best_err = min(rec_by_k.values())
        for k, err in rec_by_k.items():
            tau_needs[k].append(max(0.0, err / max(best_err, 1e-12) - 1.0))

    mean_rec = {k: float(np.mean(vals)) for k, vals in mean_errors.items()}
    best_mean = min(mean_rec.values())
    competitive_k = [k for k in k_candidates if mean_rec[k] <= (1.0 + competitiveness_cap) * best_mean]
    target_k = min(competitive_k)
    tau_k = float(np.quantile(np.asarray(tau_needs[target_k], dtype=float), 0.90))

    chosen_counter: Counter[int] = Counter()
    for seed in range(n_seeds):
        candidate = min(k for k in k_candidates if tau_needs[k][seed] <= tau_k + 1e-12)
        chosen_counter[candidate] += 1

    return TauCalibrationRow(
        scenario=str(scenario["slug"]),
        tau_k=tau_k,
        target_k=target_k,
        mean_best_error=best_mean,
        mean_target_error=mean_rec[target_k],
        k_distribution=dict(sorted(chosen_counter.items())),
    )


def run_calibration(n_seeds: int = 12) -> list[TauCalibrationRow]:
    weights = QuadraticFunctionalWeights()
    rows = [calibrate_scenario_tau(scenario, weights, n_seeds=n_seeds) for scenario in SCENARIOS]
    set_scenario_tau_k({row.scenario: row.tau_k for row in rows})
    return rows


def write_outputs(base_dir: Path, rows: list[TauCalibrationRow]) -> None:
    out_dir = base_dir / "results" / "tau_k_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rows.json").write_text(
        json.dumps([asdict(row) for row in rows], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Calibracao de tau_K por Cenario",
        "",
        "## Resultado",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"### `{row.scenario}`",
                "",
                f"- `tau_K={row.tau_k:.4f}`",
                f"- `K_alvo={row.target_k}`",
                f"- `erro_medio_best={row.mean_best_error:.3e}`",
                f"- `erro_medio_K_alvo={row.mean_target_error:.3e}`",
                f"- `distribuicao_K_canonico={row.k_distribution}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Leitura",
            "",
            "- `tau_K` controla a competitividade de erro aceita antes de aumentar a complexidade multicarta.",
            "- o valor foi calibrado por cenario para manter o menor `K` cujo erro medio fique dentro de 5% do melhor erro medio disponivel.",
            "- a escolha final usa o quantil de 90% da perda relativa do `K` alvo frente ao melhor candidato por semente.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    rows = run_calibration()
    write_outputs(base_dir, rows)
    print("Calibracao de tau_K concluida.")
    print(f"Saidas em {base_dir / 'results' / 'tau_k_calibration'}")


if __name__ == "__main__":
    main()
