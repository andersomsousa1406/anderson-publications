import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
ROOT = None
LIBRARY_ROOT = None
for parent in SCRIPT_PATH.parents:
    candidate_library = parent / "publications" / "library"
    if candidate_library.exists():
        LIBRARY_ROOT = candidate_library
        ROOT = parent
        break
if LIBRARY_ROOT is None or ROOT is None:
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
from anderson.problems.molecular import MolecularCluster


SEED = 13
ATOM_COUNT = 13


def make_problem() -> MolecularCluster:
    return MolecularCluster(atom_count=ATOM_COUNT, bounds=(-1.6, 1.6))


def make_optimizer(problem: MolecularCluster) -> GONMOptimizer:
    return GONMOptimizer(
        problem=problem,
        noise=CoolingNoise(start_std=0.08, mid_std=0.04, late_std=0.015, final_std=0.0),
        config=GONMConfig(
            population=96,
            phase1_keep=12,
            phase2_steps=220,
            phase3_steps=96,
            box_scale=1.8,
            thermal_scale=0.012,
            momentum=0.90,
            step_size=0.018,
            terminal_radii=(0.08, 0.05, 0.03, 0.015),
        ),
    )


def pairwise_distances(positions: np.ndarray) -> np.ndarray:
    dists = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dists.append(np.linalg.norm(positions[i] - positions[j]))
    return np.asarray(dists, dtype=float)


def coordination_numbers(positions: np.ndarray, threshold: float = 1.45) -> list[int]:
    counts: list[int] = []
    for i in range(len(positions)):
        count = 0
        for j in range(len(positions)):
            if i == j:
                continue
            if np.linalg.norm(positions[i] - positions[j]) <= threshold:
                count += 1
        counts.append(count)
    return counts


def radial_shell_signature(positions: np.ndarray) -> list[float]:
    radii = np.linalg.norm(np.asarray(positions, dtype=float), axis=1)
    return [float(value) for value in np.sort(radii)]


def draw_cluster(ax, positions: np.ndarray, title: str, threshold: float = 1.45) -> None:
    pos = np.asarray(positions, dtype=float)
    ax.set_title(title)
    radii = np.linalg.norm(pos, axis=1)
    ax.scatter(
        pos[:, 0],
        pos[:, 1],
        pos[:, 2],
        s=82,
        c=radii,
        cmap="plasma",
        edgecolors="black",
        linewidths=0.45,
    )
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            dist = np.linalg.norm(pos[i] - pos[j])
            if dist <= threshold:
                ax.plot(
                    [pos[i, 0], pos[j, 0]],
                    [pos[i, 1], pos[j, 1]],
                    [pos[i, 2], pos[j, 2]],
                    color="#475569",
                    alpha=0.42,
                    lw=1.2,
                )
    lim = max(2.6, float(np.max(np.abs(pos))) + 0.7)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def render_result(result: dict) -> plt.Figure:
    phase1_pos = np.asarray(result["phase1_best_positions"], dtype=float)
    final_pos = np.asarray(result["final_best_positions"], dtype=float)
    trace = np.asarray(result["trace_true_energy"], dtype=float)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1.15])
    ax_phase1 = fig.add_subplot(gs[0, 0], projection="3d")
    ax_final = fig.add_subplot(gs[0, 1], projection="3d")
    ax_trace = fig.add_subplot(gs[1, 0])
    ax_text = fig.add_subplot(gs[1, 1])

    draw_cluster(ax_phase1, phase1_pos, "Melhor estrutura apos a fase estrutural")
    draw_cluster(ax_final, final_pos, "Estrutura atomica final LJ-13 com GONM")

    ax_trace.plot(trace, color="#2563eb", lw=2.0)
    ax_trace.set_title("Trajetoria da energia verdadeira")
    ax_trace.set_xlabel("iteracao agregada")
    ax_trace.set_ylabel("energia")
    ax_trace.grid(True, linestyle=":", alpha=0.5)

    ax_text.axis("off")
    ax_text.text(
        0.0,
        1.0,
        "\n".join(
            [
                "GONM | atomic structure calculation",
                "",
                f"atomos = {result['atom_count']}",
                f"dimensoes = {result['dimensions']}",
                f"avaliacoes = {result['evals']}",
                f"tempo = {result['runtime_ms']:.1f} ms",
                "",
                f"energia inicial = {result['initial_best_energy']:.4f}",
                f"energia fase 1 = {result['phase1_best_energy']:.4f}",
                f"energia final = {result['final_best_energy']:.4f}",
                f"ganho vs fase 1 = {result['energy_gain_vs_phase1']:.4f}",
                "",
                f"distancia media = {result['final_pair_distance_mean']:.4f}",
                f"distancia minima = {result['final_pair_distance_min']:.4f}",
                f"distancia maxima = {result['final_pair_distance_max']:.4f}",
                "",
                f"coordenacao maxima = {max(result['final_coordination_numbers'])}",
                f"coordenacao media = {np.mean(result['final_coordination_numbers']):.2f}",
            ]
        ),
        ha="left",
        va="top",
        fontsize=11,
        family="monospace",
    )

    fig.suptitle("GONM | Calculo visual de estrutura atomica LJ-13", fontsize=16)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    return fig


def save_outputs(result: dict, fig: plt.Figure) -> tuple[Path, Path, Path]:
    out_dir = ROOT / "publications" / "T07_GONM" / "results" / "gonm_atomic_structure_lj13"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "summary.json"
    md_path = out_dir / "summary.md"
    image_path = out_dir / "gonm_atomic_structure_lj13.png"

    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# GONM | Atomic Structure LJ-13",
                "",
                f"- atoms: `{result['atom_count']}`",
                f"- dimensions: `{result['dimensions']}`",
                f"- evaluations: `{result['evals']}`",
                f"- runtime: `{result['runtime_ms']:.1f} ms`",
                "",
                "## Energy",
                "",
                f"- initial best: `{result['initial_best_energy']:.4f}`",
                f"- phase 1 best: `{result['phase1_best_energy']:.4f}`",
                f"- final best: `{result['final_best_energy']:.4f}`",
                f"- gain vs phase 1: `{result['energy_gain_vs_phase1']:.4f}`",
                "",
                "## Geometric indicators",
                "",
                f"- mean pair distance: `{result['final_pair_distance_mean']:.4f}`",
                f"- min pair distance: `{result['final_pair_distance_min']:.4f}`",
                f"- max pair distance: `{result['final_pair_distance_max']:.4f}`",
                f"- coordination numbers: `{result['final_coordination_numbers']}`",
                "",
                "## Files",
                "",
                "- image: `publications/T07_GONM/results/gonm_atomic_structure_lj13/gonm_atomic_structure_lj13.png`",
                "- summary: `publications/T07_GONM/results/gonm_atomic_structure_lj13/summary.json`",
            ]
        ),
        encoding="utf-8",
    )
    fig.savefig(image_path, dpi=170)
    return json_path, md_path, image_path


def main() -> None:
    problem = make_problem()
    optimizer = make_optimizer(problem)
    run = optimizer.optimize(seed=SEED)
    final_pos = np.asarray(run.final_best_positions, dtype=float)
    final_dists = pairwise_distances(final_pos)

    result = {
        "seed": run.seed,
        "atom_count": problem.atom_count,
        "dimensions": problem.dimensions,
        "evals": run.evals,
        "runtime_ms": run.runtime_ms,
        "initial_best_energy": run.initial_best_energy,
        "phase1_best_energy": run.phase1_best_energy,
        "final_best_energy": run.final_true_energy,
        "energy_gain_vs_phase1": run.energy_gain_vs_phase1,
        "phase1_best_positions": np.asarray(run.phase1_best_positions, dtype=float).tolist(),
        "final_best_positions": final_pos.tolist(),
        "trace_true_energy": [float(value) for value in run.trace_true_energy],
        "final_pair_distance_mean": float(np.mean(final_dists)),
        "final_pair_distance_min": float(np.min(final_dists)),
        "final_pair_distance_max": float(np.max(final_dists)),
        "final_coordination_numbers": coordination_numbers(final_pos),
        "final_radial_shell_signature": radial_shell_signature(final_pos),
    }

    fig = render_result(result)
    json_path, md_path, image_path = save_outputs(result, fig)
    print(f"Resumo salvo em {json_path}")
    print(f"Resumo legivel salvo em {md_path}")
    print(f"Imagem salva em {image_path}")
    if "--no-show" in sys.argv:
        plt.close(fig)
    elif "--save" not in sys.argv:
        plt.show()


if __name__ == "__main__":
    main()
