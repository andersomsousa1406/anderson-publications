from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from gauge_comparison import (
    SCENARIOS,
    affine_moments,
    mean_cov_errors,
    predict_moment_like,
    quadratic_features,
    simulate,
)
from quadratic_functional import (
    QuadraticFunctionalWeights,
    fit_penalized_tensor,
    jacobian_penalty,
)


TRANSITION_GAIN_STRONG = 0.20
TRANSITION_GAIN_JAC = 0.12
TRANSITION_JAC_THRESHOLD = 0.015
TRANSITION_FRAGMENTATION_STRONG = 1.50
TRANSITION_FRAGMENTATION_COUPLED = 1.20


@dataclass
class RobustnessCaseResult:
    scenario: str
    stress: str
    level: float
    seed: int
    v3_error: float
    v3_covariance_error: float
    v3_jacobian_penalty: float
    v3_tensor_norm: float
    residual_fragmentation_score: float
    v4_error: float
    v4_covariance_error: float
    transition_gain: float
    transition_flag: int


@dataclass
class WeightSensitivityResult:
    scenario: str
    seed: int
    perturbation: str
    coeff_delta: float
    error_delta: float
    jacobian_delta: float


def kmeans2(points: np.ndarray, rng: np.random.Generator, n_iter: int = 20) -> np.ndarray:
    idx = rng.choice(len(points), size=2, replace=False)
    centers = points[idx].copy()
    labels = np.zeros(len(points), dtype=int)
    for _ in range(n_iter):
        dist0 = np.sum((points - centers[0]) ** 2, axis=1)
        dist1 = np.sum((points - centers[1]) ** 2, axis=1)
        new_labels = (dist1 < dist0).astype(int)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for k in range(2):
            mask = labels == k
            if np.any(mask):
                centers[k] = np.mean(points[mask], axis=0)
    return labels


def residual_fragmentation_score(residual: np.ndarray, rng: np.random.Generator) -> float:
    if len(residual) < 8:
        return 0.0
    labels = kmeans2(residual, rng)
    if len(set(labels.tolist())) < 2:
        return 0.0
    score_parts: list[float] = []
    for k in range(2):
        if not np.any(labels == k):
            return 0.0
    c0 = np.mean(residual[labels == 0], axis=0)
    c1 = np.mean(residual[labels == 1], axis=0)
    sep = float(np.linalg.norm(c0 - c1))
    var0 = float(np.mean(np.sum((residual[labels == 0] - c0) ** 2, axis=1)))
    var1 = float(np.mean(np.sum((residual[labels == 1] - c1) ** 2, axis=1)))
    pooled = float(np.sqrt(max(0.5 * (var0 + var1), 1e-12)))
    balance = min(np.mean(labels == 0), np.mean(labels == 1)) / max(max(np.mean(labels == 0), np.mean(labels == 1)), 1e-12)
    score_parts.append((sep / pooled) * balance)
    return float(score_parts[0])


def predict_v4_mixture(initial: np.ndarray, final: np.ndarray, weights: QuadraticFunctionalWeights, rng: np.random.Generator) -> np.ndarray:
    labels = kmeans2(final, rng)
    pred = np.empty_like(final)
    for k in range(2):
        mask = labels == k
        if np.sum(mask) < 8:
            pred[mask] = np.mean(final, axis=0)
            continue
        q0, a0 = affine_moments(initial[mask])
        z0 = np.linalg.solve(a0, (initial[mask] - q0).T).T
        qf, af = affine_moments(final[mask])
        target_norm = np.linalg.solve(af, (final[mask] - qf).T).T
        c = fit_penalized_tensor(z0, target_norm, weights)
        pred[mask] = predict_moment_like(z0, qf, af, c)
    return pred


def fit_predict_v3(initial: np.ndarray, final: np.ndarray, weights: QuadraticFunctionalWeights) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q0, a0 = affine_moments(initial)
    z0 = np.linalg.solve(a0, (initial - q0).T).T
    qf, af = affine_moments(final)
    target_norm = np.linalg.solve(af, (final - qf).T).T
    c = fit_penalized_tensor(z0, target_norm, weights)
    pred = predict_moment_like(z0, qf, af, c)
    return pred, c, z0, af


def add_noise(final: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    scale = np.std(final, axis=0, ddof=0).mean()
    return final + rng.normal(scale=level * scale, size=final.shape)


def irregular_sample(initial: np.ndarray, final: np.ndarray, level: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    radius = np.linalg.norm(initial, axis=1)
    prob = 0.35 + 0.65 * np.exp(-level * radius)
    mask = rng.uniform(size=len(initial)) < prob
    if np.sum(mask) < max(20, len(initial) // 6):
        keep = rng.choice(len(initial), size=max(20, len(initial) // 4), replace=False)
        mask = np.zeros(len(initial), dtype=bool)
        mask[keep] = True
    return initial[mask], final[mask]


def add_outliers(final: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    out = final.copy()
    n_out = max(1, int(level * len(final)))
    idx = rng.choice(len(final), size=n_out, replace=False)
    scale = np.std(final, axis=0, ddof=0).mean()
    out[idx] += rng.normal(loc=0.0, scale=4.0 * scale, size=(n_out, 2))
    return out


def progressive_multimodality(final: np.ndarray, level: float) -> np.ndarray:
    out = final.copy()
    shift = 0.8 * level
    mask = out[:, 0] >= np.median(out[:, 0])
    out[mask, 0] += shift
    out[~mask, 0] -= shift
    return out


def v4_transition_criterion(transition_gain: float, jacobian_penalty_v3: float, fragmentation_score: float) -> int:
    strong_gain = transition_gain >= TRANSITION_GAIN_STRONG and fragmentation_score >= TRANSITION_FRAGMENTATION_STRONG
    coupled_gain = (
        transition_gain >= TRANSITION_GAIN_JAC
        and jacobian_penalty_v3 > TRANSITION_JAC_THRESHOLD
        and fragmentation_score >= TRANSITION_FRAGMENTATION_COUPLED
    )
    return int(strong_gain or coupled_gain)


def evaluate_case(initial: np.ndarray, final: np.ndarray, scenario: str, stress: str, level: float, seed: int, weights: QuadraticFunctionalWeights) -> RobustnessCaseResult:
    rng = np.random.default_rng(70000 + 97 * seed + int(1000 * level) + sum(ord(c) for c in stress))
    pred_v3, c, z0, _af = fit_predict_v3(initial, final, weights)
    v3_err = float(np.sqrt(np.mean(np.sum((final - pred_v3) ** 2, axis=1))))
    _mean_v3, cov_v3 = mean_cov_errors(final, pred_v3)
    jac_v3 = jacobian_penalty(z0, c)
    frag_v3 = residual_fragmentation_score(final - pred_v3, rng)

    pred_v4 = predict_v4_mixture(initial, final, weights, rng)
    v4_err = float(np.sqrt(np.mean(np.sum((final - pred_v4) ** 2, axis=1))))
    _mean_v4, cov_v4 = mean_cov_errors(final, pred_v4)

    gain = (v3_err - v4_err) / max(v3_err, 1e-8)
    transition_flag = v4_transition_criterion(gain, jac_v3, frag_v3)
    return RobustnessCaseResult(
        scenario=scenario,
        stress=stress,
        level=float(level),
        seed=seed,
        v3_error=v3_err,
        v3_covariance_error=float(cov_v3),
        v3_jacobian_penalty=float(jac_v3),
        v3_tensor_norm=float(np.linalg.norm(c)),
        residual_fragmentation_score=float(frag_v3),
        v4_error=v4_err,
        v4_covariance_error=float(cov_v4),
        transition_gain=float(gain),
        transition_flag=transition_flag,
    )


def weight_sensitivity(initial: np.ndarray, final: np.ndarray, scenario: str, seed: int, base: QuadraticFunctionalWeights) -> list[WeightSensitivityResult]:
    pred_base, c_base, z0, _af = fit_predict_v3(initial, final, base)
    base_err = float(np.sqrt(np.mean(np.sum((final - pred_base) ** 2, axis=1))))
    base_jac = jacobian_penalty(z0, c_base)

    perturbations = {
        'rec_up': QuadraticFunctionalWeights(1.2 * base.reconstruction, base.moment3, base.affine_orthogonality, base.jacobian, base.regularization, base.jacobian_buffer, base.jacobian_hardness),
        'm3_up': QuadraticFunctionalWeights(base.reconstruction, 1.2 * base.moment3, base.affine_orthogonality, base.jacobian, base.regularization, base.jacobian_buffer, base.jacobian_hardness),
        'aff_up': QuadraticFunctionalWeights(base.reconstruction, base.moment3, 1.2 * base.affine_orthogonality, base.jacobian, base.regularization, base.jacobian_buffer, base.jacobian_hardness),
        'jac_up': QuadraticFunctionalWeights(base.reconstruction, base.moment3, base.affine_orthogonality, 1.2 * base.jacobian, base.regularization, base.jacobian_buffer, base.jacobian_hardness),
        'reg_up': QuadraticFunctionalWeights(base.reconstruction, base.moment3, base.affine_orthogonality, base.jacobian, 1.2 * base.regularization, base.jacobian_buffer, base.jacobian_hardness),
    }

    rows: list[WeightSensitivityResult] = []
    for name, weights in perturbations.items():
        pred, c, z0_loc, _af_loc = fit_predict_v3(initial, final, weights)
        err = float(np.sqrt(np.mean(np.sum((final - pred) ** 2, axis=1))))
        jac = jacobian_penalty(z0_loc, c)
        rows.append(
            WeightSensitivityResult(
                scenario=scenario,
                seed=seed,
                perturbation=name,
                coeff_delta=float(np.linalg.norm(c - c_base)),
                error_delta=float(abs(err - base_err)),
                jacobian_delta=float(abs(jac - base_jac)),
            )
        )
    return rows


STRESS_LEVELS = {
    'ruido': [0.02, 0.05, 0.10],
    'amostragem_irregular': [0.6, 1.1, 1.8],
    'outliers': [0.03, 0.06, 0.10],
    'multimodalidade_progressiva': [0.15, 0.35, 0.60],
}


def run_robustness_suite(n_seeds: int = 12) -> tuple[list[RobustnessCaseResult], list[WeightSensitivityResult]]:
    weights = QuadraticFunctionalWeights()
    times = np.linspace(0.0, 10.0, 180)
    case_rows: list[RobustnessCaseResult] = []
    weight_rows: list[WeightSensitivityResult] = []
    for scenario in SCENARIOS:
        for seed in range(n_seeds):
            rng = np.random.default_rng(9100 + 101 * seed)
            initial = scenario['builder'](rng)
            frames = simulate(initial, times, scenario['velocity'])
            final_clean = frames[-1]

            weight_rows.extend(weight_sensitivity(initial, final_clean, scenario['slug'], seed, weights))

            for level in STRESS_LEVELS['ruido']:
                stressed = add_noise(final_clean, level, rng)
                case_rows.append(evaluate_case(initial, stressed, scenario['slug'], 'ruido', level, seed, weights))

            for level in STRESS_LEVELS['amostragem_irregular']:
                init_sub, final_sub = irregular_sample(initial, final_clean, level, rng)
                case_rows.append(evaluate_case(init_sub, final_sub, scenario['slug'], 'amostragem_irregular', level, seed, weights))

            for level in STRESS_LEVELS['outliers']:
                stressed = add_outliers(final_clean, level, rng)
                case_rows.append(evaluate_case(initial, stressed, scenario['slug'], 'outliers', level, seed, weights))

            for level in STRESS_LEVELS['multimodalidade_progressiva']:
                stressed = progressive_multimodality(final_clean, level)
                case_rows.append(evaluate_case(initial, stressed, scenario['slug'], 'multimodalidade_progressiva', level, seed, weights))
    return case_rows, weight_rows


def write_outputs(base_dir: Path, case_rows: list[RobustnessCaseResult], weight_rows: list[WeightSensitivityResult]) -> None:
    out_dir = base_dir / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'robustness_suite.json').write_text(json.dumps([asdict(x) for x in case_rows], ensure_ascii=False, indent=2), encoding='utf-8')
    (out_dir / 'weight_sensitivity.json').write_text(json.dumps([asdict(x) for x in weight_rows], ensure_ascii=False, indent=2), encoding='utf-8')

    by_stress: dict[str, list[RobustnessCaseResult]] = {}
    for row in case_rows:
        by_stress.setdefault(row.stress, []).append(row)

    lines = ['# Robustez Dura de V3', '', '## Resultado Global', '']
    for stress, items in by_stress.items():
        lines.append(
            f"- `{stress}`: v3_erro={np.mean([x.v3_error for x in items]):.3e}, v4_erro={np.mean([x.v4_error for x in items]):.3e}, ganho_transicao={np.mean([x.transition_gain for x in items]):.2%}, frag_res={np.mean([x.residual_fragmentation_score for x in items]):.3f}, sinal_v4={np.mean([x.transition_flag for x in items]):.2%}, jac_v3={np.mean([x.v3_jacobian_penalty for x in items]):.3e}"
        )

    lines.extend(['', '## Resultado por Estresse e Nivel', ''])
    for stress in STRESS_LEVELS:
        lines.append(f"### `{stress}`")
        lines.append('')
        for level in STRESS_LEVELS[stress]:
            items = [x for x in case_rows if x.stress == stress and abs(x.level - level) < 1e-12]
            lines.append(
                f"- nivel={level:.2f}: v3_erro={np.mean([x.v3_error for x in items]):.3e}, v4_erro={np.mean([x.v4_error for x in items]):.3e}, ganho={np.mean([x.transition_gain for x in items]):.2%}, frag_res={np.mean([x.residual_fragmentation_score for x in items]):.3f}, sinal_v4={np.mean([x.transition_flag for x in items]):.2%}, jac_v3={np.mean([x.v3_jacobian_penalty for x in items]):.3e}"
            )
        lines.append('')

    lines.extend(['## Sensibilidade aos Pesos', ''])
    for perturbation in ['rec_up', 'm3_up', 'aff_up', 'jac_up', 'reg_up']:
        items = [x for x in weight_rows if x.perturbation == perturbation]
        lines.append(
            f"- `{perturbation}`: delta_c={np.mean([x.coeff_delta for x in items]):.3e}, delta_erro={np.mean([x.error_delta for x in items]):.3e}, delta_jac={np.mean([x.jacobian_delta for x in items]):.3e}"
        )

    lines.extend([
        '',
        '## Criterio de Transicao V3 -> V4',
        '',
        f'- ganho forte: `ganho_transicao >= {TRANSITION_GAIN_STRONG:.2f}` e `frag_res >= {TRANSITION_FRAGMENTATION_STRONG:.2f}`.',
        f'- ganho acoplado: `ganho_transicao >= {TRANSITION_GAIN_JAC:.2f}`, `jac_v3 > {TRANSITION_JAC_THRESHOLD:.3f}` e `frag_res >= {TRANSITION_FRAGMENTATION_COUPLED:.2f}`.',
        '- `sinal_v4 = 1` quando pelo menos uma dessas duas condicoes e satisfeita.',
        '',
        '## Leitura',
        '',
        '- `frag_res` mede fragmentacao explicita do residual apos o melhor ajuste de `V3`.',
        '- `sinal_v4` agora representa um criterio formal de passagem experimental `V3 -> V4`, e nao apenas um marcador descritivo.',
        '- `ruido` mede estabilidade estatistica da gauge.',
        '- `amostragem_irregular` mede dependencia de densidade espacial e cobertura.',
        '- `outliers` mede fragilidade a contaminacao observacional.',
        '- `multimodalidade_progressiva` mede quando a carta unica comeca a perder linguagem para uma estrutura de multiplos ramos.',
    ])

    (out_dir / 'robustness_suite.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    case_rows, weight_rows = run_robustness_suite()
    write_outputs(base_dir, case_rows, weight_rows)
    print('Robustez dura de V3 concluida.')
    print(f'Saidas em {base_dir / "results"}')


if __name__ == '__main__':
    main()
