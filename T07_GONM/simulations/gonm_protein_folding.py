import json
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[1]
LIBRARY_ROOT = None
for parent in SCRIPT_PATH.parents:
    candidate = parent / "publications" / "library"
    if candidate.exists():
        LIBRARY_ROOT = candidate
        break
if LIBRARY_ROOT is None:
    raise RuntimeError("Could not locate publications/library for the anderson package.")
if str(LIBRARY_ROOT) not in sys.path:
    sys.path.insert(0, str(LIBRARY_ROOT))

import matplotlib

if "--save" in sys.argv or "--no-show" in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from anderson.noise import CoolingNoise
from anderson.optimizers.gonm import GONMConfig, GONMOptimizer
from anderson.problems import ProteinFoldingChain
from csd_plot_utils import finish_figure


SEQUENCE = "HHPPHHPPHHPPHHPPHHPPHH"
SEED = 23


def _make_problem() -> ProteinFoldingChain:
    return ProteinFoldingChain(sequence=SEQUENCE)


def _make_optimizer(problem: ProteinFoldingChain) -> GONMOptimizer:
    return GONMOptimizer(
        problem=problem,
        noise=CoolingNoise(start_std=0.08, mid_std=0.04, late_std=0.015, final_std=0.0),
        config=GONMConfig(
            population=56,
            phase1_keep=8,
            phase2_steps=140,
            phase3_steps=56,
            box_scale=2.8,
            thermal_scale=0.02,
            momentum=0.86,
            step_size=0.03,
            terminal_radii=(0.18, 0.10, 0.05, 0.025),
        ),
    )


def _draw_chain(ax, positions: np.ndarray, sequence: str, title: str) -> None:
    colors = ["#c2410c" if residue == "H" else "#0f766e" for residue in sequence]
    pos = np.asarray(positions, dtype=float)
    ax.set_title(title)
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color="#334155", lw=2.2, alpha=0.8)
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=70, c=colors, edgecolors="black", linewidths=0.4)

    for i in range(len(pos)):
        for j in range(i + 3, len(pos)):
            if sequence[i] == sequence[j] == "H" and np.linalg.norm(pos[i] - pos[j]) <= 1.75:
                ax.plot(
                    [pos[i, 0], pos[j, 0]],
                    [pos[i, 1], pos[j, 1]],
                    [pos[i, 2], pos[j, 2]],
                    color="#dc2626",
                    alpha=0.28,
                    lw=1.1,
                )

    lim = max(3.6, float(np.max(np.abs(pos))) + 0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def _sequence_text(sequence: str) -> str:
    chunks = [sequence[i : i + 11] for i in range(0, len(sequence), 11)]
    return " ".join(chunks)


def render_result(problem: ProteinFoldingChain, result) -> plt.Figure:
    phase1 = np.asarray(result.phase1_best_positions, dtype=float)
    final = np.asarray(result.final_best_positions, dtype=float)
    trace = np.asarray(result.trace_true_energy, dtype=float)
    phase1_vec = phase1.reshape(-1)
    final_vec = final.reshape(-1)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1.15])
    ax_phase1 = fig.add_subplot(gs[0, 0], projection="3d")
    ax_final = fig.add_subplot(gs[0, 1], projection="3d")
    ax_trace = fig.add_subplot(gs[1, 0])
    ax_text = fig.add_subplot(gs[1, 1])

    _draw_chain(ax_phase1, phase1, problem.sequence, "Estrutura multibacia antes do fechamento")
    _draw_chain(ax_final, final, problem.sequence, "Conformacao final dobrada com GONM")

    ax_trace.plot(trace, color="#2563eb", lw=2.0)
    ax_trace.set_title("Trajetoria da energia")
    ax_trace.set_xlabel("iteracao agregada")
    ax_trace.set_ylabel("energia")
    ax_trace.grid(True, linestyle=":", alpha=0.5)

    ax_text.axis("off")
    ax_text.text(
        0.0,
        1.0,
        "\n".join(
            [
                "GONM | protein folding coarse-grained",
                "",
                f"sequencia = {_sequence_text(problem.sequence)}",
                f"residuos = {problem.atom_count}",
                f"dimensoes = {problem.dimensions}",
                f"avaliacoes = {result.evals}",
                f"tempo = {result.runtime_ms:.1f} ms",
                "",
                f"energia fase 1 = {result.phase1_best_energy:.4f}",
                f"energia final = {result.final_true_energy:.4f}",
                f"ganho final = {result.energy_gain_vs_phase1:.4f}",
                "",
                f"R_g fase 1 = {problem.radius_of_gyration(phase1_vec):.4f}",
                f"R_g final = {problem.radius_of_gyration(final_vec):.4f}",
                f"HH contatos fase 1 = {problem.hydrophobic_contacts(phase1_vec)}",
                f"HH contatos finais = {problem.hydrophobic_contacts(final_vec)}",
                "",
                "cores: H = laranja | P = verde",
            ]
        ),
        ha="left",
        va="top",
        fontsize=11,
        family="monospace",
    )

    fig.suptitle("GONM | Simulacao visual de protein folding em cadeia 3D", fontsize=16)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    return fig


def save_outputs(problem: ProteinFoldingChain, result) -> tuple[Path, Path]:
    out_dir = Path(__file__).resolve().parent / "experiment_results" / "gonm_protein_folding"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "seed": result.seed,
        "sequence": problem.sequence,
        "residues": problem.atom_count,
        "dimensions": problem.dimensions,
        "evals": result.evals,
        "runtime_ms": result.runtime_ms,
        "initial_best_energy": result.initial_best_energy,
        "phase1_best_energy": result.phase1_best_energy,
        "final_true_energy": result.final_true_energy,
        "energy_gain_vs_phase1": result.energy_gain_vs_phase1,
        "phase1_radius_of_gyration": problem.radius_of_gyration(result.phase1_best_positions.reshape(-1)),
        "final_radius_of_gyration": problem.radius_of_gyration(result.final_best_positions.reshape(-1)),
        "phase1_hh_contacts": problem.hydrophobic_contacts(result.phase1_best_positions.reshape(-1)),
        "final_hh_contacts": problem.hydrophobic_contacts(result.final_best_positions.reshape(-1)),
        "phase1_best_positions": np.asarray(result.phase1_best_positions, dtype=float).tolist(),
        "final_best_positions": np.asarray(result.final_best_positions, dtype=float).tolist(),
        "trace_true_energy": [float(value) for value in result.trace_true_energy],
    }
    json_path = out_dir / "summary.json"
    md_path = out_dir / "summary.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# GONM | Protein Folding",
                "",
                f"- sequence: `{problem.sequence}`",
                f"- residues: `{problem.atom_count}`",
                f"- dimensions: `{problem.dimensions}`",
                f"- evaluations: `{result.evals}`",
                f"- runtime: `{result.runtime_ms:.1f} ms`",
                "",
                "## Energy",
                "",
                f"- phase 1 best: `{result.phase1_best_energy:.4f}`",
                f"- final best: `{result.final_true_energy:.4f}`",
                f"- gain vs phase 1: `{result.energy_gain_vs_phase1:.4f}`",
                "",
                "## Folding indicators",
                "",
                f"- phase 1 radius of gyration: `{summary['phase1_radius_of_gyration']:.4f}`",
                f"- final radius of gyration: `{summary['final_radius_of_gyration']:.4f}`",
                f"- phase 1 hydrophobic contacts: `{summary['phase1_hh_contacts']}`",
                f"- final hydrophobic contacts: `{summary['final_hh_contacts']}`",
                "",
                "## Files",
                "",
                "- image: `simulacao/prints/gonm_protein_folding.png`",
                "- summary: `simulacao/experiment_results/gonm_protein_folding/summary.json`",
            ]
        ),
        encoding="utf-8",
    )
    return json_path, md_path


def main() -> None:
    problem = _make_problem()
    optimizer = _make_optimizer(problem)
    result = optimizer.optimize(seed=SEED)
    json_path, md_path = save_outputs(problem, result)
    print(f"Resumo salvo em {json_path}")
    print(f"Resumo legivel salvo em {md_path}")
    fig = render_result(problem, result)
    finish_figure(fig, __file__, "gonm_protein_folding")


if __name__ == "__main__":
    main()
