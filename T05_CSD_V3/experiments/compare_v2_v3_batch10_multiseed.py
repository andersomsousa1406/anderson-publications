from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_v2_v3_batch10 import SCENARIOS, simulate
from gauge_comparison import affine_moments
from quadratic_functional import QuadraticFunctionalWeights, fit_penalized_tensor, jacobian_penalty


@dataclass
class ScenarioSeedComparison:
    scenario: str
    title: str
    seed: int
    v2_mean_temporal_error: float
    v3_mean_temporal_error: float
    v2_final_error: float
    v3_final_error: float
    temporal_gain: float
    final_gain: float
    v3_final_jacobian_penalty: float


@dataclass
class ScenarioAggregate:
    scenario: str
    title: str
    n_seeds: int
    v2_mean_temporal_error: float
    v3_mean_temporal_error: float
    v2_final_error: float
    v3_final_error: float
    temporal_gain_mean: float
    temporal_gain_std: float
    final_gain_mean: float
    final_gain_std: float
    v3_temporal_wins: int
    v3_final_wins: int
    v3_final_jacobian_penalty_mean: float


@dataclass
class GlobalAggregate:
    n_scenarios: int
    n_seeds: int
    total_cases: int
    v2_mean_temporal_error: float
    v3_mean_temporal_error: float
    v2_mean_final_error: float
    v3_mean_final_error: float
    temporal_gain_mean: float
    final_gain_mean: float
    v3_temporal_wins: int
    v3_final_wins: int


def reconstruction_error(pos: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((pos - pred) ** 2, axis=1))))


def evaluate_seed(scenario: dict[str, object], seed: int, weights: QuadraticFunctionalWeights, times: np.ndarray) -> tuple[ScenarioSeedComparison, dict[str, list[float]]]:
    rng = np.random.default_rng(12000 + 997 * seed + int(str(scenario['slug'])[:2]))
    initial = scenario['builder'](rng)
    frames = simulate(initial, times, scenario['velocity'])

    q0, a0 = affine_moments(initial)
    z0 = np.linalg.solve(a0, (initial - q0).T).T
    h = np.column_stack((z0[:, 0] ** 2 - 1.0, z0[:, 0] * z0[:, 1], z0[:, 1] ** 2 - 1.0))

    v2_errors: list[float] = []
    v3_errors: list[float] = []
    v3_jac_penalties: list[float] = []

    for pos in frames:
        qf, af = affine_moments(pos)
        pred_v2 = qf + z0 @ af.T
        v2_errors.append(reconstruction_error(pos, pred_v2))

        target_norm = np.linalg.solve(af, (pos - qf).T).T
        coeffs = fit_penalized_tensor(z0, target_norm, weights)
        pred_v3 = qf + (z0 + h @ coeffs.T) @ af.T
        v3_errors.append(reconstruction_error(pos, pred_v3))
        v3_jac_penalties.append(jacobian_penalty(z0, coeffs))

    row = ScenarioSeedComparison(
        scenario=str(scenario['slug']),
        title=str(scenario['title']),
        seed=seed,
        v2_mean_temporal_error=float(np.mean(v2_errors)),
        v3_mean_temporal_error=float(np.mean(v3_errors)),
        v2_final_error=float(v2_errors[-1]),
        v3_final_error=float(v3_errors[-1]),
        temporal_gain=float(np.mean(v2_errors) / max(np.mean(v3_errors), 1e-12)),
        final_gain=float(v2_errors[-1] / max(v3_errors[-1], 1e-12)),
        v3_final_jacobian_penalty=float(v3_jac_penalties[-1]),
    )
    curves = {'v2': v2_errors, 'v3': v3_errors}
    return row, curves


def aggregate(rows: list[ScenarioSeedComparison], n_seeds: int) -> tuple[list[ScenarioAggregate], GlobalAggregate]:
    by_scenario: dict[str, list[ScenarioSeedComparison]] = {}
    for row in rows:
        by_scenario.setdefault(row.scenario, []).append(row)

    scenario_rows: list[ScenarioAggregate] = []
    for scenario, items in by_scenario.items():
        first = items[0]
        scenario_rows.append(
            ScenarioAggregate(
                scenario=scenario,
                title=first.title,
                n_seeds=n_seeds,
                v2_mean_temporal_error=float(np.mean([x.v2_mean_temporal_error for x in items])),
                v3_mean_temporal_error=float(np.mean([x.v3_mean_temporal_error for x in items])),
                v2_final_error=float(np.mean([x.v2_final_error for x in items])),
                v3_final_error=float(np.mean([x.v3_final_error for x in items])),
                temporal_gain_mean=float(np.mean([x.temporal_gain for x in items])),
                temporal_gain_std=float(np.std([x.temporal_gain for x in items], ddof=0)),
                final_gain_mean=float(np.mean([x.final_gain for x in items])),
                final_gain_std=float(np.std([x.final_gain for x in items], ddof=0)),
                v3_temporal_wins=sum(x.v3_mean_temporal_error < x.v2_mean_temporal_error for x in items),
                v3_final_wins=sum(x.v3_final_error < x.v2_final_error for x in items),
                v3_final_jacobian_penalty_mean=float(np.mean([x.v3_final_jacobian_penalty for x in items])),
            )
        )
    scenario_rows.sort(key=lambda x: x.scenario)

    global_row = GlobalAggregate(
        n_scenarios=len(scenario_rows),
        n_seeds=n_seeds,
        total_cases=len(rows),
        v2_mean_temporal_error=float(np.mean([x.v2_mean_temporal_error for x in rows])),
        v3_mean_temporal_error=float(np.mean([x.v3_mean_temporal_error for x in rows])),
        v2_mean_final_error=float(np.mean([x.v2_final_error for x in rows])),
        v3_mean_final_error=float(np.mean([x.v3_final_error for x in rows])),
        temporal_gain_mean=float(np.mean([x.temporal_gain for x in rows])),
        final_gain_mean=float(np.mean([x.final_gain for x in rows])),
        v3_temporal_wins=sum(x.v3_mean_temporal_error < x.v2_mean_temporal_error for x in rows),
        v3_final_wins=sum(x.v3_final_error < x.v2_final_error for x in rows),
    )
    return scenario_rows, global_row


def write_outputs(
    base_dir: Path,
    seed_rows: list[ScenarioSeedComparison],
    scenario_rows: list[ScenarioAggregate],
    global_row: GlobalAggregate,
    curves: dict[str, dict[str, list[float]]],
    times: np.ndarray,
) -> None:
    out_dir = base_dir / 'results' / 'v2_v3_batch10_multiseed'
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'seed_results.json').write_text(json.dumps([asdict(x) for x in seed_rows], ensure_ascii=False, indent=2), encoding='utf-8')
    (out_dir / 'scenario_summary.json').write_text(json.dumps([asdict(x) for x in scenario_rows], ensure_ascii=False, indent=2), encoding='utf-8')
    (out_dir / 'global_summary.json').write_text(json.dumps(asdict(global_row), ensure_ascii=False, indent=2), encoding='utf-8')
    (out_dir / 'curves.json').write_text(json.dumps(curves, ensure_ascii=False, indent=2), encoding='utf-8')

    lines = ['# Comparacao V2 vs V3 no Lote de 10 Simulacoes com Multiplas Sementes', '', '## Resultado Global', '']
    lines.append(f"- cenarios={global_row.n_scenarios}, sementes_por_cenario={global_row.n_seeds}, casos={global_row.total_cases}")
    lines.append(f"- erro_temporal_medio_v2={global_row.v2_mean_temporal_error:.3e}, erro_temporal_medio_v3={global_row.v3_mean_temporal_error:.3e}, ganho_temporal_medio={global_row.temporal_gain_mean:.3f}x")
    lines.append(f"- erro_final_medio_v2={global_row.v2_mean_final_error:.3e}, erro_final_medio_v3={global_row.v3_mean_final_error:.3e}, ganho_final_medio={global_row.final_gain_mean:.3f}x")
    lines.append(f"- vitorias_v3_temporal={global_row.v3_temporal_wins}/{global_row.total_cases}")
    lines.append(f"- vitorias_v3_final={global_row.v3_final_wins}/{global_row.total_cases}")
    lines.extend(['', '## Resultado por Cenario', ''])
    for row in scenario_rows:
        lines.append(
            f"- `{row.scenario}`: ganho_temporal={row.temporal_gain_mean:.3f}x+-{row.temporal_gain_std:.3f}, ganho_final={row.final_gain_mean:.3f}x+-{row.final_gain_std:.3f}, vitorias_temporais={row.v3_temporal_wins}/{row.n_seeds}, vitorias_finais={row.v3_final_wins}/{row.n_seeds}, jac_v3={row.v3_final_jacobian_penalty_mean:.3e}"
        )
    (out_dir / 'summary.md').write_text("\n".join(lines) + "\n", encoding='utf-8')

    fig, axes = plt.subplots(5, 2, figsize=(14, 18), sharex=True)
    axes = axes.ravel()
    for ax, row in zip(axes, scenario_rows):
        curve = curves[row.scenario]
        ax.plot(times, curve['v2'], color='tab:blue', lw=2.0, label='V2 medio')
        ax.plot(times, curve['v3'], color='tab:red', lw=2.0, label='V3 medio')
        ax.set_title(row.title)
        ax.grid(True, linestyle=':', alpha=0.35)
    axes[0].legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(out_dir / 'temporal_curves_mean.png', dpi=160)
    plt.close(fig)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    n_seeds = 16
    weights = QuadraticFunctionalWeights()
    times = np.linspace(0.0, 10.0, 180)
    seed_rows: list[ScenarioSeedComparison] = []
    curve_acc: dict[str, dict[str, list[np.ndarray]]] = {s['slug']: {'v2': [], 'v3': []} for s in SCENARIOS}

    for scenario in SCENARIOS:
        for seed in range(n_seeds):
            row, curves = evaluate_seed(scenario, seed, weights, times)
            seed_rows.append(row)
            curve_acc[row.scenario]['v2'].append(np.asarray(curves['v2']))
            curve_acc[row.scenario]['v3'].append(np.asarray(curves['v3']))

    scenario_rows, global_row = aggregate(seed_rows, n_seeds)
    mean_curves = {
        slug: {
            'v2': np.mean(bundle['v2'], axis=0).tolist(),
            'v3': np.mean(bundle['v3'], axis=0).tolist(),
        }
        for slug, bundle in curve_acc.items()
    }
    write_outputs(base_dir, seed_rows, scenario_rows, global_row, mean_curves, times)
    print('Comparacao V2 vs V3 multiseed concluida.')
    print(f'Saidas em {base_dir / "results" / "v2_v3_batch10_multiseed"}')


if __name__ == '__main__':
    main()
