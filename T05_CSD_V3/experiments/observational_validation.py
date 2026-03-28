from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from compare_v2_v3_batch10 import SCENARIOS, simulate
from gauge_comparison import affine_moments
from quadratic_functional import (
    FamilyBasis,
    QuadraticFunctionalWeights,
    build_family_basis,
    fit_penalized_family,
    jacobian_penalty_from_basis,
)


@dataclass
class ObservationalCase:
    scenario: str
    title: str
    seed: int
    v2_mean_temporal_error: float
    v3_quad_mean_temporal_error: float
    v3_cubic_mean_temporal_error: float
    v2_final_error: float
    v3_quad_final_error: float
    v3_cubic_final_error: float
    quad_gain_over_v2: float
    cubic_gain_over_v2: float
    cubic_gain_over_quad: float
    quad_final_jacobian_penalty: float
    cubic_final_jacobian_penalty: float


def reconstruction_error(pos: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((pos - pred) ** 2, axis=1))))


def predict_family(z0: np.ndarray, q: np.ndarray, a: np.ndarray, coeffs: np.ndarray, basis: FamilyBasis) -> np.ndarray:
    return q + (z0 + basis.values @ coeffs.T) @ a.T


def tracked_subset(n: int, rng: np.random.Generator) -> np.ndarray:
    keep = rng.random(n) > 0.28
    if np.sum(keep) < max(40, n // 4):
        keep[rng.choice(n, size=max(40, n // 4), replace=False)] = True
    return np.where(keep)[0]


def degrade_observation(pos: np.ndarray, indices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    obs = pos[indices].copy()
    radial = np.linalg.norm(obs, axis=1)
    noise_scale = 0.035 + 0.012 * np.tanh(radial / 3.0)
    noise = rng.normal(size=obs.shape) * noise_scale[:, None]
    anisotropic = np.column_stack((0.018 * np.sin(obs[:, 1]), -0.015 * np.cos(obs[:, 0])))
    obs += noise + anisotropic

    n_out = max(1, int(0.03 * len(obs)))
    out_idx = rng.choice(len(obs), size=n_out, replace=False)
    obs[out_idx] += rng.normal(scale=0.22, size=(n_out, 2))
    return obs


def evaluate_case(scenario: dict[str, object], seed: int, weights: QuadraticFunctionalWeights, times: np.ndarray) -> ObservationalCase:
    rng = np.random.default_rng(44000 + 991 * seed + int(str(scenario['slug'])[:2]))
    initial_clean = scenario['builder'](rng)
    frames_clean = simulate(initial_clean, times, scenario['velocity'])
    indices = tracked_subset(len(initial_clean), rng)

    initial_obs = degrade_observation(initial_clean, indices, rng)
    q0, a0 = affine_moments(initial_obs)
    z0 = np.linalg.solve(a0, (initial_obs - q0).T).T
    basis_quad = build_family_basis(z0, 'quadratic')
    basis_cubic = build_family_basis(z0, 'cubic')

    v2_errors: list[float] = []
    quad_errors: list[float] = []
    cubic_errors: list[float] = []
    quad_jac: list[float] = []
    cubic_jac: list[float] = []

    for pos_clean in frames_clean:
        pos_obs = degrade_observation(pos_clean, indices, rng)
        qf, af = affine_moments(pos_obs)

        pred_v2 = qf + z0 @ af.T
        v2_errors.append(reconstruction_error(pos_obs, pred_v2))

        target_norm = np.linalg.solve(af, (pos_obs - qf).T).T
        coeffs_quad = fit_penalized_family(z0, target_norm, weights, family='quadratic')
        coeffs_cubic = fit_penalized_family(z0, target_norm, weights, family='cubic')

        pred_quad = predict_family(z0, qf, af, coeffs_quad, basis_quad)
        pred_cubic = predict_family(z0, qf, af, coeffs_cubic, basis_cubic)

        quad_errors.append(reconstruction_error(pos_obs, pred_quad))
        cubic_errors.append(reconstruction_error(pos_obs, pred_cubic))
        quad_jac.append(jacobian_penalty_from_basis(basis_quad, coeffs_quad))
        cubic_jac.append(jacobian_penalty_from_basis(basis_cubic, coeffs_cubic))

    return ObservationalCase(
        scenario=str(scenario['slug']),
        title=str(scenario['title']),
        seed=seed,
        v2_mean_temporal_error=float(np.mean(v2_errors)),
        v3_quad_mean_temporal_error=float(np.mean(quad_errors)),
        v3_cubic_mean_temporal_error=float(np.mean(cubic_errors)),
        v2_final_error=float(v2_errors[-1]),
        v3_quad_final_error=float(quad_errors[-1]),
        v3_cubic_final_error=float(cubic_errors[-1]),
        quad_gain_over_v2=float(np.mean(v2_errors) / max(np.mean(quad_errors), 1e-12)),
        cubic_gain_over_v2=float(np.mean(v2_errors) / max(np.mean(cubic_errors), 1e-12)),
        cubic_gain_over_quad=float(np.mean(quad_errors) / max(np.mean(cubic_errors), 1e-12)),
        quad_final_jacobian_penalty=float(quad_jac[-1]),
        cubic_final_jacobian_penalty=float(cubic_jac[-1]),
    )


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir / 'results' / 'observational_validation'
    out_dir.mkdir(parents=True, exist_ok=True)

    times = np.linspace(0.0, 10.0, 120)
    weights = QuadraticFunctionalWeights()
    n_seeds = 8
    rows: list[ObservationalCase] = []
    for scenario in SCENARIOS:
        for seed in range(n_seeds):
            rows.append(evaluate_case(scenario, seed, weights, times))

    (out_dir / 'cases.json').write_text(json.dumps([asdict(x) for x in rows], ensure_ascii=False, indent=2), encoding='utf-8')

    lines = ['# Validacao Observacional de V3', '', '## Resultado Global', '']
    lines.append(f'- cenarios=10, sementes_por_cenario={n_seeds}, casos={len(rows)}')
    lines.append(f'- erro_temporal_medio_v2={np.mean([x.v2_mean_temporal_error for x in rows]):.3e}')
    lines.append(f'- erro_temporal_medio_v3_quad={np.mean([x.v3_quad_mean_temporal_error for x in rows]):.3e}')
    lines.append(f'- erro_temporal_medio_v3_cubic={np.mean([x.v3_cubic_mean_temporal_error for x in rows]):.3e}')
    lines.append(f'- ganho_quad_sobre_v2={np.mean([x.quad_gain_over_v2 for x in rows]):.3f}x')
    lines.append(f'- ganho_cubic_sobre_v2={np.mean([x.cubic_gain_over_v2 for x in rows]):.3f}x')
    lines.append(f'- ganho_cubic_sobre_quad={np.mean([x.cubic_gain_over_quad for x in rows]):.3f}x')
    lines.append(f'- vitorias_quad_sobre_v2={sum(x.v3_quad_mean_temporal_error < x.v2_mean_temporal_error for x in rows)}/{len(rows)}')
    lines.append(f'- vitorias_cubic_sobre_v2={sum(x.v3_cubic_mean_temporal_error < x.v2_mean_temporal_error for x in rows)}/{len(rows)}')
    lines.append(f'- vitorias_cubic_sobre_quad={sum(x.v3_cubic_mean_temporal_error < x.v3_quad_mean_temporal_error for x in rows)}/{len(rows)}')
    lines.extend(['', '## Resultado por Cenario', ''])

    by_scenario: dict[str, list[ObservationalCase]] = {}
    for row in rows:
        by_scenario.setdefault(row.scenario, []).append(row)
    for slug, items in sorted(by_scenario.items()):
        lines.append(
            f"- `{slug}`: ganho_quad_v2={np.mean([x.quad_gain_over_v2 for x in items]):.3f}x, ganho_cubic_v2={np.mean([x.cubic_gain_over_v2 for x in items]):.3f}x, ganho_cubic_quad={np.mean([x.cubic_gain_over_quad for x in items]):.3f}x, jac_quad={np.mean([x.quad_final_jacobian_penalty for x in items]):.3e}, jac_cubic={np.mean([x.cubic_final_jacobian_penalty for x in items]):.3e}"
        )

    lines.extend([
        '',
        '## Protocolo',
        '',
        '- observacao parcial com descarte aleatorio de aproximadamente 28% dos pontos;',
        '- ruido heterocedastico dependente da posicao;',
        '- viés anisotropico suave de medicao;',
        '- contaminacao por outliers em cerca de 3% das observacoes;',
        '- comparacao entre `V2`, `V3` quadratica e `V3` cubica no mesmo pipeline observacional.',
    ])
    (out_dir / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('Validacao observacional concluida.')
    print(f'Saidas em {out_dir}')


if __name__ == '__main__':
    main()
