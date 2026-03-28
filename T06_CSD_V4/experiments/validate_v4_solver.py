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
from robustness_suite import predict_v4_mixture
from v4_solver import solve_v4_theoretical


@dataclass
class ValidationRow:
    scenario: str
    seed: int
    proxy_error: float
    solver_error: float
    gain_over_proxy: float
    chosen_k: int
    converged: bool
    quasi_converged: bool
    iterative_stability: float
    fallback_to_v3: bool
    gauge_status: str
    objective: float


def run_validation(n_seeds: int = 12) -> list[ValidationRow]:
    weights = QuadraticFunctionalWeights()
    times = np.linspace(0.0, 10.0, 180)
    rows: list[ValidationRow] = []
    for scenario in SCENARIOS:
        for seed in range(n_seeds):
            rng = np.random.default_rng(9100 + 101 * seed)
            initial = scenario["builder"](rng)
            final = simulate(initial, times, scenario["velocity"])[-1]
            pred_proxy = predict_v4_mixture(initial, final, weights, rng=np.random.default_rng(4000 + seed))
            proxy_error = float(np.sqrt(np.mean(np.sum((final - pred_proxy) ** 2, axis=1))))

            result = solve_v4_theoretical(
                initial,
                final,
                scenario_key=scenario["slug"],
                weights_v3=weights,
                seed=8000 + seed,
            )
            solver_error = float(np.sqrt(np.mean(np.sum((final - result.prediction) ** 2, axis=1))))
            gain = (proxy_error - solver_error) / max(proxy_error, 1e-12)
            rows.append(
                ValidationRow(
                    scenario=scenario["slug"],
                    seed=seed,
                    proxy_error=proxy_error,
                    solver_error=solver_error,
                    gain_over_proxy=float(gain),
                    chosen_k=result.n_components,
                    converged=result.converged,
                    quasi_converged=result.quasi_converged,
                    iterative_stability=float(result.iterative_stability),
                    fallback_to_v3=bool(result.fallback_to_v3),
                    gauge_status=result.gauge_status,
                    objective=float(result.objective),
                )
            )
    return rows


def write_outputs(base_dir: Path, rows: list[ValidationRow]) -> None:
    out_dir = base_dir / "results" / "solver_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rows.json").write_text(
        json.dumps([asdict(row) for row in rows], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    by_scenario: dict[str, list[ValidationRow]] = {}
    for row in rows:
        by_scenario.setdefault(row.scenario, []).append(row)

    global_wins = sum(row.solver_error < row.proxy_error for row in rows)
    k_counter = Counter(row.chosen_k for row in rows)
    conv_rate = np.mean([row.converged for row in rows]) if rows else 0.0
    quasi_rate = np.mean([row.quasi_converged for row in rows]) if rows else 0.0
    fallback_rate = np.mean([row.fallback_to_v3 for row in rows]) if rows else 0.0

    lines = [
        "# Validacao do Solver Teorico de V4",
        "",
        "## Resultado Global",
        "",
        f"- casos={len(rows)}",
        f"- proxy_erro={np.mean([row.proxy_error for row in rows]):.3e}",
        f"- solver_erro={np.mean([row.solver_error for row in rows]):.3e}",
        f"- ganho_medio={np.mean([row.gain_over_proxy for row in rows]):.2%}",
        f"- vitorias_solver={global_wins}/{len(rows)}",
        f"- convergencia={conv_rate:.2%}",
        f"- quase_convergencia={quasi_rate:.2%}",
        f"- estabilidade_iterativa={np.mean([row.iterative_stability for row in rows]):.3f}",
        f"- fallback_V4_para_V3={fallback_rate:.2%}",
        f"- distribuicao_K={dict(sorted(k_counter.items()))}",
        "",
        "## Resultado por Cenario",
        "",
    ]

    for scenario, items in by_scenario.items():
        local_counter = Counter(row.chosen_k for row in items)
        lines.extend(
            [
                f"### `{scenario}`",
                "",
                f"- proxy_erro={np.mean([row.proxy_error for row in items]):.3e}",
                f"- solver_erro={np.mean([row.solver_error for row in items]):.3e}",
                f"- ganho_medio={np.mean([row.gain_over_proxy for row in items]):.2%}",
                f"- vitorias_solver={sum(row.solver_error < row.proxy_error for row in items)}/{len(items)}",
                f"- convergencia={np.mean([row.converged for row in items]):.2%}",
                f"- quase_convergencia={np.mean([row.quasi_converged for row in items]):.2%}",
                f"- estabilidade_iterativa={np.mean([row.iterative_stability for row in items]):.3f}",
                f"- fallback_V4_para_V3={np.mean([row.fallback_to_v3 for row in items]):.2%}",
                f"- distribuicao_K={dict(sorted(local_counter.items()))}",
                "",
            ]
        )

    lines.extend(
        [
            "## Leitura",
            "",
            "- o solver teorico de V4 escolhe `K` por minimizacao do funcional multicarta com penalidade de complexidade.",
            "- a proxy antiga de V4 corresponde a uma mistura fixa de 2 componentes obtida por `k-means` e ajuste local tipo V3.",
            "- quando o solver escolhe `K>2`, isso indica regime estrutural mais rico do que a proxy capturava.",
            "- quando a fusao canonica colapsa para `K=1`, a gauge passa a sinalizar retorno formal ao regime de carta unica (`fallback_v3`).",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    rows = run_validation()
    write_outputs(base_dir, rows)
    print("Validacao do solver teorico de V4 concluida.")
    print(f"Saidas em {base_dir / 'results' / 'solver_validation'}")


if __name__ == "__main__":
    main()
