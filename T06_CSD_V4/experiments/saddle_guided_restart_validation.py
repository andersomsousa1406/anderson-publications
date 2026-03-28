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
class SaddleRow:
    mode: str
    seed: int
    error: float
    objective: float
    k: int
    converged: int
    quasi_converged: int
    iterative_stability: float


def reconstruction_error(final: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((final - pred) ** 2, axis=1))))


def run_suite(n_seeds: int = 24) -> list[SaddleRow]:
    weights = QuadraticFunctionalWeights()
    scenario = next(s for s in SCENARIOS if s["slug"] == "sela_torcida")
    times = np.linspace(0.0, 10.0, 180)
    rows: list[SaddleRow] = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(9100 + 101 * seed)
        initial = scenario["builder"](rng)
        final = simulate(initial, times, scenario["velocity"])[-1]

        # Baseline without scenario-guided initialization.
        baseline = solve_v4_theoretical(
            initial,
            final,
            scenario_key=None,
            weights_v3=weights,
            seed=50000 + seed,
        )
        rows.append(
            SaddleRow(
                mode="baseline",
                seed=seed,
                error=reconstruction_error(final, baseline.prediction),
                objective=float(baseline.objective),
                k=int(baseline.n_components),
                converged=int(baseline.converged),
                quasi_converged=int(baseline.quasi_converged),
                iterative_stability=float(baseline.iterative_stability),
            )
        )

        guided = solve_v4_theoretical(
            initial,
            final,
            scenario_key="sela_torcida",
            weights_v3=weights,
            seed=50000 + seed,
        )
        rows.append(
            SaddleRow(
                mode="guided",
                seed=seed,
                error=reconstruction_error(final, guided.prediction),
                objective=float(guided.objective),
                k=int(guided.n_components),
                converged=int(guided.converged),
                quasi_converged=int(guided.quasi_converged),
                iterative_stability=float(guided.iterative_stability),
            )
        )
    return rows


def write_outputs(base_dir: Path, rows: list[SaddleRow]) -> None:
    out_dir = base_dir / "results" / "saddle_guided_restart"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rows.json").write_text(
        json.dumps([asdict(row) for row in rows], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    baseline = [row for row in rows if row.mode == "baseline"]
    guided = [row for row in rows if row.mode == "guided"]

    lines = [
        "# Validacao de Reinicializacao Guiada em sela_torcida",
        "",
        "## Resultado",
        "",
        f"- baseline_erro={np.mean([row.error for row in baseline]):.3e}",
        f"- guided_erro={np.mean([row.error for row in guided]):.3e}",
        f"- baseline_obj={np.mean([row.objective for row in baseline]):.3e}",
        f"- guided_obj={np.mean([row.objective for row in guided]):.3e}",
        f"- baseline_conv={np.mean([row.converged for row in baseline]):.2%}",
        f"- guided_conv={np.mean([row.converged for row in guided]):.2%}",
        f"- baseline_quase={np.mean([row.quasi_converged for row in baseline]):.2%}",
        f"- guided_quase={np.mean([row.quasi_converged for row in guided]):.2%}",
        f"- baseline_estab={np.mean([row.iterative_stability for row in baseline]):.3f}",
        f"- guided_estab={np.mean([row.iterative_stability for row in guided]):.3f}",
        f"- baseline_K={np.mean([row.k for row in baseline]):.2f}",
        f"- guided_K={np.mean([row.k for row in guided]):.2f}",
        f"- guided_melhor_erro={sum(g.error < b.error for b, g in zip(baseline, guided))}/{len(guided)}",
        f"- guided_melhor_obj={sum(g.objective < b.objective for b, g in zip(baseline, guided))}/{len(guided)}",
        "",
        "## Leitura",
        "",
        "- `baseline` usa apenas inicializacoes genericas.",
        "- `guided` ativa inicializacoes geometricas orientadas por sela.",
        "- a comparacao mede se a geometria local melhora convergencia e qualidade do solver em `sela_torcida`.",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    rows = run_suite()
    write_outputs(base_dir, rows)
    print("Validacao de reinicializacao guiada em sela_torcida concluida.")
    print(f"Saidas em {base_dir / 'results' / 'saddle_guided_restart'}")


if __name__ == "__main__":
    main()
