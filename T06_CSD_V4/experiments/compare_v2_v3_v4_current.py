from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(r'c:\Projetos\CSD')
V3_ROOT = ROOT / 'V3'
V4_ROOT = ROOT / 'V4'
for p in (V3_ROOT, V4_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from compare_v2_v3_batch10 import SCENARIOS, simulate
from gauge_comparison import affine_moments
from quadratic_functional import QuadraticFunctionalWeights, build_family_basis, fit_penalized_family, jacobian_penalty_from_basis
from robustness_suite import add_noise, add_outliers, irregular_sample, progressive_multimodality, residual_fragmentation_score, v4_transition_criterion
from v4_solver import solve_v4_theoretical


@dataclass
class Row:
    benchmark: str
    n_cases: int
    v2_error: float
    v3q_error: float
    v3c_error: float
    v4_error: float
    v3q_wins_over_v2: int
    v3c_wins_over_v2: int
    v4_wins_over_v3q: int
    v4_wins_over_v3c: int
    trans_q: float
    trans_c: float


def reconstruction_error(pos: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((pos - pred) ** 2, axis=1))))


def predict_v2(z0: np.ndarray, qf: np.ndarray, af: np.ndarray) -> np.ndarray:
    return qf + z0 @ af.T


def predict_v3_family(z0: np.ndarray, qf: np.ndarray, af: np.ndarray, target_norm: np.ndarray, weights: QuadraticFunctionalWeights, family: str):
    basis = build_family_basis(z0, family)
    coeffs = fit_penalized_family(z0, target_norm, weights, family=family)
    pred = qf + (z0 + basis.values @ coeffs.T) @ af.T
    jac = float(jacobian_penalty_from_basis(basis, coeffs))
    return pred, jac


def eval_case(initial: np.ndarray, final: np.ndarray, weights: QuadraticFunctionalWeights, scenario_key: str, seed: int):
    q0, a0 = affine_moments(initial)
    z0 = np.linalg.solve(a0, (initial - q0).T).T
    qf, af = affine_moments(final)
    target_norm = np.linalg.solve(af, (final - qf).T).T
    pred_v2 = predict_v2(z0, qf, af)
    pred_v3q, jac_q = predict_v3_family(z0, qf, af, target_norm, weights, 'quadratic')
    pred_v3c, jac_c = predict_v3_family(z0, qf, af, target_norm, weights, 'cubic')
    result_v4 = solve_v4_theoretical(initial, final, scenario_key=scenario_key, weights_v3=weights, seed=seed)
    pred_v4 = result_v4.prediction
    err_v2 = reconstruction_error(final, pred_v2)
    err_v3q = reconstruction_error(final, pred_v3q)
    err_v3c = reconstruction_error(final, pred_v3c)
    err_v4 = reconstruction_error(final, pred_v4)
    frag_q = residual_fragmentation_score(final - pred_v3q, np.random.default_rng(seed + 31))
    frag_c = residual_fragmentation_score(final - pred_v3c, np.random.default_rng(seed + 63))
    gain_q = (err_v3q - err_v4) / max(err_v3q, 1e-12)
    gain_c = (err_v3c - err_v4) / max(err_v3c, 1e-12)
    return {
        'v2': err_v2,
        'v3q': err_v3q,
        'v3c': err_v3c,
        'v4': err_v4,
        'trans_q': float(v4_transition_criterion(gain_q, jac_q, frag_q)),
        'trans_c': float(v4_transition_criterion(gain_c, jac_c, frag_c)),
    }


def aggregate(name: str, cases: list[dict[str, float]]) -> Row:
    return Row(
        benchmark=name,
        n_cases=len(cases),
        v2_error=float(np.mean([c['v2'] for c in cases])),
        v3q_error=float(np.mean([c['v3q'] for c in cases])),
        v3c_error=float(np.mean([c['v3c'] for c in cases])),
        v4_error=float(np.mean([c['v4'] for c in cases])),
        v3q_wins_over_v2=sum(c['v3q'] < c['v2'] for c in cases),
        v3c_wins_over_v2=sum(c['v3c'] < c['v2'] for c in cases),
        v4_wins_over_v3q=sum(c['v4'] < c['v3q'] for c in cases),
        v4_wins_over_v3c=sum(c['v4'] < c['v3c'] for c in cases),
        trans_q=float(np.mean([c['trans_q'] for c in cases])),
        trans_c=float(np.mean([c['trans_c'] for c in cases])),
    )


def run():
    weights = QuadraticFunctionalWeights()
    times_clean = np.linspace(0.0, 10.0, 60)
    times_obs = np.linspace(0.0, 10.0, 45)
    rows = []

    clean_cases = []
    for scenario in SCENARIOS:
        for seed in range(3):
            rng = np.random.default_rng(1000 + 97 * seed + int(str(scenario['slug'])[:2]))
            initial = scenario['builder'](rng)
            final = simulate(initial, times_clean, scenario['velocity'])[-1]
            clean_cases.append(eval_case(initial, final, weights, scenario['slug'], 10000 + seed))
    rows.append(aggregate('clean_final_multiseed', clean_cases))

    obs_cases = []
    for scenario in SCENARIOS:
        for seed in range(2):
            rng = np.random.default_rng(2000 + 97 * seed + int(str(scenario['slug'])[:2]))
            initial_clean = scenario['builder'](rng)
            final_clean = simulate(initial_clean, times_obs, scenario['velocity'])[-1]
            keep = rng.random(len(initial_clean)) > 0.28
            if np.sum(keep) < max(40, len(initial_clean)//4):
                keep[rng.choice(len(initial_clean), size=max(40, len(initial_clean)//4), replace=False)] = True
            idx = np.where(keep)[0]
            initial = initial_clean[idx].copy()
            final = final_clean[idx].copy()
            radial = np.linalg.norm(final, axis=1)
            noise_scale = 0.035 + 0.012 * np.tanh(radial / 3.0)
            final += rng.normal(size=final.shape) * noise_scale[:, None]
            final += np.column_stack((0.018 * np.sin(final[:, 1]), -0.015 * np.cos(final[:, 0])))
            n_out = max(1, int(0.03 * len(final)))
            out_idx = rng.choice(len(final), size=n_out, replace=False)
            final[out_idx] += rng.normal(scale=0.22, size=(n_out, 2))
            obs_cases.append(eval_case(initial, final, weights, scenario['slug'], 20000 + seed))
    rows.append(aggregate('observational_degraded_final', obs_cases))

    stress_cases = []
    selected_stress = [
        ('ruido', 0.10),
        ('amostragem_irregular', 0.35),
        ('outliers', 0.08),
        ('multimodalidade_progressiva', 0.60),
    ]
    for scenario in SCENARIOS:
        for stress_name, level in selected_stress:
            rng = np.random.default_rng(3000 + int(str(scenario['slug'])[:2]) + int(level * 100))
            initial = scenario['builder'](rng)
            final_clean = simulate(initial, times_clean, scenario['velocity'])[-1]
            if stress_name == 'ruido':
                stressed_initial, stressed_final = initial, add_noise(final_clean, level, rng)
            elif stress_name == 'amostragem_irregular':
                stressed_initial, stressed_final = irregular_sample(initial, final_clean, level, rng)
            elif stress_name == 'outliers':
                stressed_initial, stressed_final = initial, add_outliers(final_clean, level, rng)
            else:
                stressed_initial, stressed_final = initial, progressive_multimodality(final_clean, level)
            stress_cases.append(eval_case(stressed_initial, stressed_final, weights, scenario['slug'], 30000 + int(level * 100)))
    rows.append(aggregate('stress_mixed_final', stress_cases))

    out_dir = ROOT / 'V4' / 'results' / 'current_v2_v3_v4_compare'
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'summary.json').write_text(json.dumps([asdict(r) for r in rows], ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# Comparativo Atual V2 V3 V4', '']
    for row in rows:
        lines += [f"## `{row.benchmark}`", '', f"- casos={row.n_cases}", f"- erro medio: V2={row.v2_error:.3e}, V3_quad={row.v3q_error:.3e}, V3_cubic={row.v3c_error:.3e}, V4={row.v4_error:.3e}", f"- vitorias: V3_quad>V2={row.v3q_wins_over_v2}/{row.n_cases}, V3_cubic>V2={row.v3c_wins_over_v2}/{row.n_cases}, V4>V3_quad={row.v4_wins_over_v3q}/{row.n_cases}, V4>V3_cubic={row.v4_wins_over_v3c}/{row.n_cases}", f"- transicao: V3_quad->V4={row.trans_q:.2%}, V3_cubic->V4={row.trans_c:.2%}", '']
    (out_dir / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('Comparativo atual concluido.')
    print(out_dir)


if __name__ == '__main__':
    run()
