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
from robustness_suite import kmeans2, residual_fragmentation_score, v4_transition_criterion


@dataclass
class ScenarioSeedComparison:
    scenario: str
    title: str
    seed: int
    v2_mean_temporal_error: float
    v3_mean_temporal_error: float
    v4_mean_temporal_error: float
    v2_final_error: float
    v3_final_error: float
    v4_final_error: float
    gain_v3_over_v2: float
    gain_v4_over_v3: float
    gain_v4_over_v2: float
    v3_final_jacobian_penalty: float
    v3_residual_fragmentation: float
    transition_flag: int


@dataclass
class ScenarioAggregate:
    scenario: str
    title: str
    n_seeds: int
    gain_v3_over_v2_mean: float
    gain_v4_over_v3_mean: float
    gain_v4_over_v2_mean: float
    gain_v4_over_v3_std: float
    v3_temporal_wins: int
    v4_temporal_wins_over_v3: int
    v4_final_wins_over_v3: int
    transition_rate: float
    v3_residual_fragmentation_mean: float
    v3_final_jacobian_penalty_mean: float


@dataclass
class GlobalAggregate:
    n_scenarios: int
    n_seeds: int
    total_cases: int
    v2_mean_temporal_error: float
    v3_mean_temporal_error: float
    v4_mean_temporal_error: float
    v2_mean_final_error: float
    v3_mean_final_error: float
    v4_mean_final_error: float
    gain_v3_over_v2_mean: float
    gain_v4_over_v3_mean: float
    gain_v4_over_v2_mean: float
    v3_temporal_wins: int
    v4_temporal_wins_over_v3: int
    v4_final_wins_over_v3: int
    transition_rate: float


def reconstruction_error(pos: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((pos - pred) ** 2, axis=1))))


def predict_v3_from_reference(z0: np.ndarray, h: np.ndarray, pos: np.ndarray, weights: QuadraticFunctionalWeights) -> tuple[np.ndarray, np.ndarray, float, float]:
    qf, af = affine_moments(pos)
    target_norm = np.linalg.solve(af, (pos - qf).T).T
    coeffs = fit_penalized_tensor(z0, target_norm, weights)
    pred = qf + (z0 + h @ coeffs.T) @ af.T
    jac = jacobian_penalty(z0, coeffs)
    frag = residual_fragmentation_score(pos - pred, np.random.default_rng(12345))
    return pred, coeffs, jac, frag


def predict_v4_from_reference(initial: np.ndarray, pos: np.ndarray, weights: QuadraticFunctionalWeights, rng: np.random.Generator) -> np.ndarray:
    labels = kmeans2(pos, rng)
    pred = np.empty_like(pos)
    for k in range(2):
        mask = labels == k
        if np.sum(mask) < 8:
            pred[mask] = np.mean(pos, axis=0)
            continue
        q0, a0 = affine_moments(initial[mask])
        z0 = np.linalg.solve(a0, (initial[mask] - q0).T).T
        h0 = np.column_stack((z0[:, 0] ** 2 - 1.0, z0[:, 0] * z0[:, 1], z0[:, 1] ** 2 - 1.0))
        qf, af = affine_moments(pos[mask])
        target_norm = np.linalg.solve(af, (pos[mask] - qf).T).T
        coeffs = fit_penalized_tensor(z0, target_norm, weights)
        pred[mask] = qf + (z0 + h0 @ coeffs.T) @ af.T
    return pred


def evaluate_seed(scenario: dict[str, object], seed: int, weights: QuadraticFunctionalWeights, times: np.ndarray) -> tuple[ScenarioSeedComparison, dict[str, list[float]]]:
    rng = np.random.default_rng(15000 + 997 * seed + int(str(scenario['slug'])[:2]))
    initial = scenario['builder'](rng)
    frames = simulate(initial, times, scenario['velocity'])

    q0, a0 = affine_moments(initial)
    z0 = np.linalg.solve(a0, (initial - q0).T).T
    h = np.column_stack((z0[:, 0] ** 2 - 1.0, z0[:, 0] * z0[:, 1], z0[:, 1] ** 2 - 1.0))

    v2_errors: list[float] = []
    v3_errors: list[float] = []
    v4_errors: list[float] = []
    final_jac = 0.0
    final_frag = 0.0
    final_transition = 0

    for frame_idx, pos in enumerate(frames):
        qf, af = affine_moments(pos)
        pred_v2 = qf + z0 @ af.T
        v2_errors.append(reconstruction_error(pos, pred_v2))

        pred_v3, coeffs, jac, frag = predict_v3_from_reference(z0, h, pos, weights)
        v3_err = reconstruction_error(pos, pred_v3)
        v3_errors.append(v3_err)

        pred_v4 = predict_v4_from_reference(initial, pos, weights, np.random.default_rng(50000 + seed * 1000 + frame_idx))
        v4_err = reconstruction_error(pos, pred_v4)
        v4_errors.append(v4_err)

        if frame_idx == len(frames) - 1:
            final_jac = jac
            final_frag = frag
            gain_v4_over_v3 = (v3_err - v4_err) / max(v3_err, 1e-12)
            final_transition = v4_transition_criterion(gain_v4_over_v3, jac, frag)

    v2_mean = float(np.mean(v2_errors))
    v3_mean = float(np.mean(v3_errors))
    v4_mean = float(np.mean(v4_errors))
    v2_final = float(v2_errors[-1])
    v3_final = float(v3_errors[-1])
    v4_final = float(v4_errors[-1])

    row = ScenarioSeedComparison(
        scenario=str(scenario['slug']),
        title=str(scenario['title']),
        seed=seed,
        v2_mean_temporal_error=v2_mean,
        v3_mean_temporal_error=v3_mean,
        v4_mean_temporal_error=v4_mean,
        v2_final_error=v2_final,
        v3_final_error=v3_final,
        v4_final_error=v4_final,
        gain_v3_over_v2=float(v2_mean / max(v3_mean, 1e-12)),
        gain_v4_over_v3=float(v3_mean / max(v4_mean, 1e-12)),
        gain_v4_over_v2=float(v2_mean / max(v4_mean, 1e-12)),
        v3_final_jacobian_penalty=float(final_jac),
        v3_residual_fragmentation=float(final_frag),
        transition_flag=int(final_transition),
    )
    curves = {'v2': v2_errors, 'v3': v3_errors, 'v4': v4_errors}
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
                gain_v3_over_v2_mean=float(np.mean([x.gain_v3_over_v2 for x in items])),
                gain_v4_over_v3_mean=float(np.mean([x.gain_v4_over_v3 for x in items])),
                gain_v4_over_v2_mean=float(np.mean([x.gain_v4_over_v2 for x in items])),
                gain_v4_over_v3_std=float(np.std([x.gain_v4_over_v3 for x in items], ddof=0)),
                v3_temporal_wins=sum(x.v3_mean_temporal_error < x.v2_mean_temporal_error for x in items),
                v4_temporal_wins_over_v3=sum(x.v4_mean_temporal_error < x.v3_mean_temporal_error for x in items),
                v4_final_wins_over_v3=sum(x.v4_final_error < x.v3_final_error for x in items),
                transition_rate=float(np.mean([x.transition_flag for x in items])),
                v3_residual_fragmentation_mean=float(np.mean([x.v3_residual_fragmentation for x in items])),
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
        v4_mean_temporal_error=float(np.mean([x.v4_mean_temporal_error for x in rows])),
        v2_mean_final_error=float(np.mean([x.v2_final_error for x in rows])),
        v3_mean_final_error=float(np.mean([x.v3_final_error for x in rows])),
        v4_mean_final_error=float(np.mean([x.v4_final_error for x in rows])),
        gain_v3_over_v2_mean=float(np.mean([x.gain_v3_over_v2 for x in rows])),
        gain_v4_over_v3_mean=float(np.mean([x.gain_v4_over_v3 for x in rows])),
        gain_v4_over_v2_mean=float(np.mean([x.gain_v4_over_v2 for x in rows])),
        v3_temporal_wins=sum(x.v3_mean_temporal_error < x.v2_mean_temporal_error for x in rows),
        v4_temporal_wins_over_v3=sum(x.v4_mean_temporal_error < x.v3_mean_temporal_error for x in rows),
        v4_final_wins_over_v3=sum(x.v4_final_error < x.v3_final_error for x in rows),
        transition_rate=float(np.mean([x.transition_flag for x in rows])),
    )
    return scenario_rows, global_row


def write_outputs(base_dir: Path, seed_rows: list[ScenarioSeedComparison], scenario_rows: list[ScenarioAggregate], global_row: GlobalAggregate, curves: dict[str, dict[str, list[float]]], times: np.ndarray) -> None:
    out_dir = base_dir / 'results' / 'v2_v3_v4_batch10_multiseed'
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'seed_results.json').write_text(json.dumps([asdict(x) for x in seed_rows], ensure_ascii=False, indent=2), encoding='utf-8')
    (out_dir / 'scenario_summary.json').write_text(json.dumps([asdict(x) for x in scenario_rows], ensure_ascii=False, indent=2), encoding='utf-8')
    (out_dir / 'global_summary.json').write_text(json.dumps(asdict(global_row), ensure_ascii=False, indent=2), encoding='utf-8')
    (out_dir / 'curves.json').write_text(json.dumps(curves, ensure_ascii=False, indent=2), encoding='utf-8')

    lines = ['# Comparacao V2 vs V3 vs V4 no Lote de 10 Simulacoes com Multiplas Sementes', '', '## Resultado Global', '']
    lines.append(f"- cenarios={global_row.n_scenarios}, sementes_por_cenario={global_row.n_seeds}, casos={global_row.total_cases}")
    lines.append(f"- erro_temporal_medio: V2={global_row.v2_mean_temporal_error:.3e}, V3={global_row.v3_mean_temporal_error:.3e}, V4={global_row.v4_mean_temporal_error:.3e}")
    lines.append(f"- erro_final_medio: V2={global_row.v2_mean_final_error:.3e}, V3={global_row.v3_mean_final_error:.3e}, V4={global_row.v4_mean_final_error:.3e}")
    lines.append(f"- ganho_medio V3/V2={global_row.gain_v3_over_v2_mean:.3f}x, V4/V3={global_row.gain_v4_over_v3_mean:.3f}x, V4/V2={global_row.gain_v4_over_v2_mean:.3f}x")
    lines.append(f"- vitorias_V3_sobre_V2={global_row.v3_temporal_wins}/{global_row.total_cases}")
    lines.append(f"- vitorias_V4_sobre_V3_temporal={global_row.v4_temporal_wins_over_v3}/{global_row.total_cases}")
    lines.append(f"- vitorias_V4_sobre_V3_final={global_row.v4_final_wins_over_v3}/{global_row.total_cases}")
    lines.append(f"- taxa_transicao_V3_para_V4={global_row.transition_rate:.2%}")
    lines.extend(['', '## Resultado por Cenario', ''])
    for row in scenario_rows:
        lines.append(
            f"- `{row.scenario}`: ganho_V3/V2={row.gain_v3_over_v2_mean:.3f}x, ganho_V4/V3={row.gain_v4_over_v3_mean:.3f}x+-{row.gain_v4_over_v3_std:.3f}, ganho_V4/V2={row.gain_v4_over_v2_mean:.3f}x, V4>V3_temporal={row.v4_temporal_wins_over_v3}/{row.n_seeds}, V4>V3_final={row.v4_final_wins_over_v3}/{row.n_seeds}, transicao={row.transition_rate:.2%}, frag_res={row.v3_residual_fragmentation_mean:.3f}, jac_v3={row.v3_final_jacobian_penalty_mean:.3e}"
        )
    (out_dir / 'summary.md').write_text("\n".join(lines) + "\n", encoding='utf-8')

    fig, axes = plt.subplots(5, 2, figsize=(14, 18), sharex=True)
    axes = axes.ravel()
    for ax, row in zip(axes, scenario_rows):
        curve = curves[row.scenario]
        ax.plot(times, curve['v2'], color='tab:blue', lw=2.0, label='V2 medio')
        ax.plot(times, curve['v3'], color='tab:red', lw=2.0, label='V3 medio')
        ax.plot(times, curve['v4'], color='tab:green', lw=2.0, label='V4 medio')
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
    curve_acc: dict[str, dict[str, list[np.ndarray]]] = {s['slug']: {'v2': [], 'v3': [], 'v4': []} for s in SCENARIOS}

    for scenario in SCENARIOS:
        for seed in range(n_seeds):
            row, curves = evaluate_seed(scenario, seed, weights, times)
            seed_rows.append(row)
            curve_acc[row.scenario]['v2'].append(np.asarray(curves['v2']))
            curve_acc[row.scenario]['v3'].append(np.asarray(curves['v3']))
            curve_acc[row.scenario]['v4'].append(np.asarray(curves['v4']))

    scenario_rows, global_row = aggregate(seed_rows, n_seeds)
    mean_curves = {
        slug: {
            'v2': np.mean(bundle['v2'], axis=0).tolist(),
            'v3': np.mean(bundle['v3'], axis=0).tolist(),
            'v4': np.mean(bundle['v4'], axis=0).tolist(),
        }
        for slug, bundle in curve_acc.items()
    }
    write_outputs(base_dir, seed_rows, scenario_rows, global_row, mean_curves, times)
    print('Comparacao V2 vs V3 vs V4 multiseed concluida.')
    print(f'Saidas em {base_dir / "results" / "v2_v3_v4_batch10_multiseed"}')


if __name__ == '__main__':
    main()
