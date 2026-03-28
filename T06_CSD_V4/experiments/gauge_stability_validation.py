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

from gauge_comparison import SCENARIOS, simulate
from quadratic_functional import QuadraticFunctionalWeights
from v4_solver import solve_v4_theoretical


@dataclass
class StabilityRow:
    scenario: str
    mode: str
    seed: int
    mean_error: float
    error_std: float
    mean_objective: float
    objective_std: float
    mean_k: float
    std_k: float
    mode_share_k: float
    coassignment_agreement: float
    convergence_rate: float
    quasi_convergence_rate: float
    iterative_stability: float
    fallback_rate: float


def reconstruction_error(final: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((final - pred) ** 2, axis=1))))


def coassignment_agreement(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    n = len(labels_a)
    if n != len(labels_b) or n <= 1:
        return 1.0
    same_a = labels_a[:, None] == labels_a[None, :]
    same_b = labels_b[:, None] == labels_b[None, :]
    mask = ~np.eye(n, dtype=bool)
    return float(np.mean(same_a[mask] == same_b[mask]))


def perturb_final(final: np.ndarray, rng: np.random.Generator, level: float = 0.015) -> np.ndarray:
    scale = np.std(final, axis=0, ddof=0).mean()
    return final + rng.normal(scale=level * scale, size=final.shape)


def evaluate_case(
    initial: np.ndarray,
    final: np.ndarray,
    scenario_key: str,
    seed: int,
    mode: str,
    repeats: int,
    weights: QuadraticFunctionalWeights,
) -> StabilityRow:
    labels_list: list[np.ndarray] = []
    errors: list[float] = []
    objectives: list[float] = []
    ks: list[int] = []
    converged: list[int] = []
    quasi_converged: list[int] = []
    iterative_stability: list[float] = []
    fallback_flags: list[int] = []

    for rep in range(repeats):
        if mode == "seed":
            final_rep = final
        elif mode == "perturb":
            final_rep = perturb_final(final, np.random.default_rng(70000 + seed * 97 + rep))
        else:
            raise ValueError(mode)

        result = solve_v4_theoretical(
            initial,
            final_rep,
            scenario_key=scenario_key,
            weights_v3=weights,
            seed=10000 + seed * 211 + rep * 997,
        )
        labels_list.append(result.labels)
        errors.append(reconstruction_error(final_rep, result.prediction))
        objectives.append(float(result.objective))
        ks.append(int(result.n_components))
        converged.append(int(result.converged))
        quasi_converged.append(int(result.quasi_converged))
        iterative_stability.append(float(result.iterative_stability))
        fallback_flags.append(int(result.fallback_to_v3))

    agreements = []
    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            agreements.append(coassignment_agreement(labels_list[i], labels_list[j]))

    unique_k, counts_k = np.unique(np.asarray(ks, dtype=int), return_counts=True)
    mode_share = float(np.max(counts_k) / max(len(ks), 1))

    return StabilityRow(
        scenario=scenario_key,
        mode=mode,
        seed=seed,
        mean_error=float(np.mean(errors)),
        error_std=float(np.std(errors)),
        mean_objective=float(np.mean(objectives)),
        objective_std=float(np.std(objectives)),
        mean_k=float(np.mean(ks)),
        std_k=float(np.std(ks)),
        mode_share_k=mode_share,
        coassignment_agreement=float(np.mean(agreements) if agreements else 1.0),
        convergence_rate=float(np.mean(converged)),
        quasi_convergence_rate=float(np.mean(quasi_converged)),
        iterative_stability=float(np.mean(iterative_stability)),
        fallback_rate=float(np.mean(fallback_flags)),
    )


def run_suite(n_seeds: int = 6, repeats: int = 4) -> list[StabilityRow]:
    weights = QuadraticFunctionalWeights()
    times = np.linspace(0.0, 10.0, 180)
    rows: list[StabilityRow] = []
    for scenario in SCENARIOS:
        for seed in range(n_seeds):
            rng = np.random.default_rng(9100 + 101 * seed)
            initial = scenario["builder"](rng)
            final = simulate(initial, times, scenario["velocity"])[-1]
            rows.append(evaluate_case(initial, final, scenario["slug"], seed, "seed", repeats, weights))
            rows.append(evaluate_case(initial, final, scenario["slug"], seed, "perturb", repeats, weights))
    return rows


def write_outputs(base_dir: Path, rows: list[StabilityRow]) -> None:
    out_dir = base_dir / "results" / "gauge_stability"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rows.json").write_text(
        json.dumps([asdict(row) for row in rows], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    by_mode: dict[str, list[StabilityRow]] = {}
    by_scenario_mode: dict[tuple[str, str], list[StabilityRow]] = {}
    for row in rows:
        by_mode.setdefault(row.mode, []).append(row)
        by_scenario_mode.setdefault((row.scenario, row.mode), []).append(row)

    lines = [
        "# Estabilidade da Gauge Multicarta de V4",
        "",
        "## Resultado Global por Modo",
        "",
    ]

    for mode, items in by_mode.items():
        lines.extend(
            [
                f"### `{mode}`",
                "",
                f"- erro_std={np.mean([row.error_std for row in items]):.3e}",
                f"- objective_std={np.mean([row.objective_std for row in items]):.3e}",
                f"- std_K={np.mean([row.std_k for row in items]):.3f}",
                f"- modo_share_K={np.mean([row.mode_share_k for row in items]):.2%}",
                f"- coassignment={np.mean([row.coassignment_agreement for row in items]):.2%}",
                f"- convergencia={np.mean([row.convergence_rate for row in items]):.2%}",
                f"- quase_convergencia={np.mean([row.quasi_convergence_rate for row in items]):.2%}",
                f"- estabilidade_iterativa={np.mean([row.iterative_stability for row in items]):.3f}",
                f"- fallback_V4_para_V3={np.mean([row.fallback_rate for row in items]):.2%}",
                "",
            ]
        )

    lines.extend(["## Resultado por Cenario e Modo", ""])
    for (scenario, mode), items in sorted(by_scenario_mode.items()):
        lines.extend(
            [
                f"### `{scenario}` / `{mode}`",
                "",
                f"- erro_std={np.mean([row.error_std for row in items]):.3e}",
                f"- objective_std={np.mean([row.objective_std for row in items]):.3e}",
                f"- std_K={np.mean([row.std_k for row in items]):.3f}",
                f"- modo_share_K={np.mean([row.mode_share_k for row in items]):.2%}",
                f"- coassignment={np.mean([row.coassignment_agreement for row in items]):.2%}",
                f"- convergencia={np.mean([row.convergence_rate for row in items]):.2%}",
                f"- quase_convergencia={np.mean([row.quasi_convergence_rate for row in items]):.2%}",
                f"- estabilidade_iterativa={np.mean([row.iterative_stability for row in items]):.3f}",
                f"- fallback_V4_para_V3={np.mean([row.fallback_rate for row in items]):.2%}",
                "",
            ]
        )

    lines.extend(
        [
            "## Leitura",
            "",
            "- `seed` mede estabilidade da gauge sob mudanca de inicializacao do solver.",
            "- `perturb` mede estabilidade da gauge sob pequenas perturbacoes nos dados observados.",
            "- `coassignment` e invariante a permutacao de rotulos e mede estabilidade real da decomposicao multicarta.",
            "- `fallback_V4_para_V3` mede quantas vezes a fusao canonica colapsa a decomposicao ate carta unica.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    rows = run_suite()
    write_outputs(base_dir, rows)
    print("Validacao de estabilidade da gauge multicarta concluida.")
    print(f"Saidas em {base_dir / 'results' / 'gauge_stability'}")


if __name__ == "__main__":
    main()
