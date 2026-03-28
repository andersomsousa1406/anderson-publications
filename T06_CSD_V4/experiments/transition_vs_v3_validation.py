from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
V3_ROOT = ROOT / "V3"
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

from gauge_comparison import SCENARIOS, affine_moments, simulate
from quadratic_functional import (
    QuadraticFunctionalWeights,
    build_family_basis,
    fit_penalized_family,
    jacobian_penalty_from_basis,
)
from robustness_suite import progressive_multimodality, residual_fragmentation_score, v4_transition_criterion
from v4_solver import solve_v4_theoretical


ONTOLOGY_SCENARIOS = (
    "multimodal_leve",
    "cisao_em_quatro",
    "anel_filamentado",
    "sela_torcida",
    "dobramento_senoidal",
)

MULTIMODAL_LEVELS = (0.15, 0.35, 0.60, 0.85)


@dataclass
class TransitionRow:
    scenario: str
    level: float
    seed: int
    v3_quadratic_error: float
    v3_cubic_error: float
    v4_error: float
    v3_quadratic_jacobian: float
    v3_cubic_jacobian: float
    v3_quadratic_fragmentation: float
    v3_cubic_fragmentation: float
    transition_from_v3_quadratic: int
    transition_from_v3_cubic: int
    v4_k: int
    v4_converged: int
    v4_quasi_converged: int
    v4_iterative_stability: float
    v4_fallback_to_v3: int
    v4_gauge_status: str
    v4_objective: float


def predict_v3_family(
    initial: np.ndarray,
    final: np.ndarray,
    weights: QuadraticFunctionalWeights,
    family: str,
) -> tuple[np.ndarray, np.ndarray]:
    q0, a0 = affine_moments(initial)
    z0 = np.linalg.solve(a0, (initial - q0).T).T
    qf, af = affine_moments(final)
    target_norm = np.linalg.solve(af, (final - qf).T).T
    basis = build_family_basis(z0, family)
    coeffs = fit_penalized_family(z0, target_norm, weights, family=family)
    pred = qf + (z0 + basis.values @ coeffs.T) @ af.T
    jac = float(jacobian_penalty_from_basis(basis, coeffs))
    return pred, np.array([jac], dtype=float)


def reconstruction_error(final: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((final - pred) ** 2, axis=1))))


def evaluate_case(
    initial: np.ndarray,
    final: np.ndarray,
    scenario: str,
    level: float,
    seed: int,
    weights: QuadraticFunctionalWeights,
) -> TransitionRow:
    pred_q, jac_q_vec = predict_v3_family(initial, final, weights, "quadratic")
    pred_c, jac_c_vec = predict_v3_family(initial, final, weights, "cubic")
    err_q = reconstruction_error(final, pred_q)
    err_c = reconstruction_error(final, pred_c)
    jac_q = float(jac_q_vec[0])
    jac_c = float(jac_c_vec[0])
    frag_q = float(residual_fragmentation_score(final - pred_q, np.random.default_rng(31000 + seed)))
    frag_c = float(residual_fragmentation_score(final - pred_c, np.random.default_rng(41000 + seed)))

    result_v4 = solve_v4_theoretical(
        initial,
        final,
        scenario_key=scenario,
        weights_v3=weights,
        seed=51000 + seed,
    )
    err_v4 = reconstruction_error(final, result_v4.prediction)

    gain_q = (err_q - err_v4) / max(err_q, 1e-12)
    gain_c = (err_c - err_v4) / max(err_c, 1e-12)
    trans_q = int(v4_transition_criterion(gain_q, jac_q, frag_q))
    trans_c = int(v4_transition_criterion(gain_c, jac_c, frag_c))

    return TransitionRow(
        scenario=scenario,
        level=float(level),
        seed=seed,
        v3_quadratic_error=err_q,
        v3_cubic_error=err_c,
        v4_error=err_v4,
        v3_quadratic_jacobian=jac_q,
        v3_cubic_jacobian=jac_c,
        v3_quadratic_fragmentation=frag_q,
        v3_cubic_fragmentation=frag_c,
        transition_from_v3_quadratic=trans_q,
        transition_from_v3_cubic=trans_c,
        v4_k=result_v4.n_components,
        v4_converged=int(result_v4.converged),
        v4_quasi_converged=int(result_v4.quasi_converged),
        v4_iterative_stability=float(result_v4.iterative_stability),
        v4_fallback_to_v3=int(result_v4.fallback_to_v3),
        v4_gauge_status=result_v4.gauge_status,
        v4_objective=float(result_v4.objective),
    )


def run_suite(n_seeds: int = 12) -> list[TransitionRow]:
    weights = QuadraticFunctionalWeights()
    times = np.linspace(0.0, 10.0, 180)
    rows: list[TransitionRow] = []
    scenario_map = {scenario["slug"]: scenario for scenario in SCENARIOS if scenario["slug"] in ONTOLOGY_SCENARIOS}
    for slug in ONTOLOGY_SCENARIOS:
        scenario = scenario_map[slug]
        for seed in range(n_seeds):
            rng = np.random.default_rng(9100 + 101 * seed)
            initial = scenario["builder"](rng)
            final_clean = simulate(initial, times, scenario["velocity"])[-1]
            for level in MULTIMODAL_LEVELS:
                final = progressive_multimodality(final_clean, level)
                rows.append(evaluate_case(initial, final, slug, level, seed, weights))
    return rows


def write_outputs(base_dir: Path, rows: list[TransitionRow]) -> None:
    out_dir = base_dir / "results" / "transition_vs_v3"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rows.json").write_text(
        json.dumps([asdict(row) for row in rows], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    by_level: dict[float, list[TransitionRow]] = {}
    by_scenario: dict[str, list[TransitionRow]] = {}
    for row in rows:
        by_level.setdefault(row.level, []).append(row)
        by_scenario.setdefault(row.scenario, []).append(row)

    lines = [
        "# Validacao de Transicao Ontologica: V4 vs V3",
        "",
        "## Resultado Global",
        "",
        f"- casos={len(rows)}",
        f"- erro_medio V3_quadratica={np.mean([row.v3_quadratic_error for row in rows]):.3e}",
        f"- erro_medio V3_cubica={np.mean([row.v3_cubic_error for row in rows]):.3e}",
        f"- erro_medio V4={np.mean([row.v4_error for row in rows]):.3e}",
        f"- V4 vence V3_quadratica={sum(row.v4_error < row.v3_quadratic_error for row in rows)}/{len(rows)}",
        f"- V4 vence V3_cubica={sum(row.v4_error < row.v3_cubic_error for row in rows)}/{len(rows)}",
        f"- transicao a partir de V3_quadratica={np.mean([row.transition_from_v3_quadratic for row in rows]):.2%}",
        f"- transicao a partir de V3_cubica={np.mean([row.transition_from_v3_cubic for row in rows]):.2%}",
        f"- convergencia V4={np.mean([row.v4_converged for row in rows]):.2%}",
        f"- quase_convergencia V4={np.mean([row.v4_quasi_converged for row in rows]):.2%}",
        f"- estabilidade_iterativa V4={np.mean([row.v4_iterative_stability for row in rows]):.3f}",
        f"- fallback_V4_para_V3={np.mean([row.v4_fallback_to_v3 for row in rows]):.2%}",
        "",
        "## Resultado por Nivel de Multimodalidade",
        "",
    ]

    for level in MULTIMODAL_LEVELS:
        items = by_level[level]
        lines.extend(
            [
                f"### `nivel={level:.2f}`",
                "",
                f"- erro V3_quadratica={np.mean([row.v3_quadratic_error for row in items]):.3e}",
                f"- erro V3_cubica={np.mean([row.v3_cubic_error for row in items]):.3e}",
                f"- erro V4={np.mean([row.v4_error for row in items]):.3e}",
                f"- V4 vence V3_quadratica={sum(row.v4_error < row.v3_quadratic_error for row in items)}/{len(items)}",
                f"- V4 vence V3_cubica={sum(row.v4_error < row.v3_cubic_error for row in items)}/{len(items)}",
                f"- transicao de V3_quadratica={np.mean([row.transition_from_v3_quadratic for row in items]):.2%}",
                f"- transicao de V3_cubica={np.mean([row.transition_from_v3_cubic for row in items]):.2%}",
                f"- K medio de V4={np.mean([row.v4_k for row in items]):.2f}",
                f"- quase_convergencia V4={np.mean([row.v4_quasi_converged for row in items]):.2%}",
                f"- fallback_V4_para_V3={np.mean([row.v4_fallback_to_v3 for row in items]):.2%}",
                "",
            ]
        )

    lines.extend(["## Resultado por Cenario", ""])
    for scenario, items in by_scenario.items():
        lines.extend(
            [
                f"### `{scenario}`",
                "",
                f"- erro V3_quadratica={np.mean([row.v3_quadratic_error for row in items]):.3e}",
                f"- erro V3_cubica={np.mean([row.v3_cubic_error for row in items]):.3e}",
                f"- erro V4={np.mean([row.v4_error for row in items]):.3e}",
                f"- V4 vence V3_quadratica={sum(row.v4_error < row.v3_quadratic_error for row in items)}/{len(items)}",
                f"- V4 vence V3_cubica={sum(row.v4_error < row.v3_cubic_error for row in items)}/{len(items)}",
                f"- transicao de V3_quadratica={np.mean([row.transition_from_v3_quadratic for row in items]):.2%}",
                f"- transicao de V3_cubica={np.mean([row.transition_from_v3_cubic for row in items]):.2%}",
                f"- K medio de V4={np.mean([row.v4_k for row in items]):.2f}",
                f"- quase_convergencia V4={np.mean([row.v4_quasi_converged for row in items]):.2%}",
                f"- estabilidade_iterativa V4={np.mean([row.v4_iterative_stability for row in items]):.3f}",
                f"- fallback_V4_para_V3={np.mean([row.v4_fallback_to_v3 for row in items]):.2%}",
                "",
            ]
        )

    lines.extend(
        [
            "## Leitura",
            "",
            "- `V3_quadratica` representa a camada estavel oficial de V3.",
            "- `V3_cubica` representa a extensao mais expressiva e mais recente de V3.",
            "- `V4` representa o solver teorico multicarta com selecao de `K`.",
            "- quando a fusao canônica colapsa a decomposicao para `K=1`, a leitura correta passa a ser retorno formal ao regime de carta unica (`V4 -> V3`).",
            "- este teste nao mede apenas erro: ele tambem mede quando o criterio de transicao ontologica passa a indicar que carta unica deixou de ser suficiente.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    rows = run_suite()
    write_outputs(base_dir, rows)
    print("Validacao V4 vs V3 em cenarios de transicao ontologica concluida.")
    print(f"Saidas em {base_dir / 'results' / 'transition_vs_v3'}")


if __name__ == "__main__":
    main()
