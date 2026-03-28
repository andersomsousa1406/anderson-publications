import json
import sys
from dataclasses import asdict
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
from anderson.potentials import LennardJonesPotential
from anderson.problems.molecular import MolecularCluster


SEED = 13
ATOM_COUNT = 13
ELEMENT = "Ar"
SIGMA_ANGSTROM = 3.405
EPSILON_EV = 0.0103
SOFTEN_ANGSTROM = 1.1
NEIGHBOR_THRESHOLD = 4.1
LJ13_REDUCED_MINIMUM = -44.326801
EXPECTED_GROUND_STATE_EV = LJ13_REDUCED_MINIMUM * EPSILON_EV


def make_problem() -> MolecularCluster:
    potential = LennardJonesPotential(
        sigma=SIGMA_ANGSTROM,
        epsilon=EPSILON_EV,
        soften=SOFTEN_ANGSTROM,
    )
    return MolecularCluster(
        atom_count=ATOM_COUNT,
        potential=potential,
        bounds=(-5.6, 5.6),
    )


def make_optimizer(problem: MolecularCluster) -> GONMOptimizer:
    return GONMOptimizer(
        problem=problem,
        noise=CoolingNoise(start_std=0.0012, mid_std=0.0006, late_std=0.0002, final_std=0.0),
        config=GONMConfig(
            population=120,
            phase1_keep=14,
            phase2_steps=260,
            phase3_steps=120,
            box_scale=4.2,
            thermal_scale=0.12,
            momentum=0.90,
            step_size=0.13,
            terminal_radii=(0.45, 0.24, 0.12, 0.06),
        ),
    )


def parse_args() -> tuple[bool, int]:
    long_mode = "--long" in sys.argv
    explicit_runs = None
    for arg in sys.argv:
        if arg.startswith("--runs="):
            explicit_runs = int(arg.split("=", 1)[1])
    if explicit_runs is not None:
        return long_mode, explicit_runs
    return long_mode, (24 if long_mode else 8)


def make_icosahedral_seed(problem: MolecularCluster, rng: np.random.Generator, jitter: float = 0.0) -> np.ndarray:
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.asarray(
        [
            [0.0, 1.0, phi],
            [0.0, -1.0, phi],
            [0.0, 1.0, -phi],
            [0.0, -1.0, -phi],
            [1.0, phi, 0.0],
            [-1.0, phi, 0.0],
            [1.0, -phi, 0.0],
            [-1.0, -phi, 0.0],
            [phi, 0.0, 1.0],
            [-phi, 0.0, 1.0],
            [phi, 0.0, -1.0],
            [-phi, 0.0, -1.0],
        ],
        dtype=float,
    )
    edge_length = np.min(
        [
            np.linalg.norm(vertices[i] - vertices[j])
            for i in range(len(vertices))
            for j in range(i + 1, len(vertices))
        ]
    )
    target_nn = (2.0 ** (1.0 / 6.0)) * SIGMA_ANGSTROM
    vertices *= target_nn / edge_length
    positions = np.vstack([np.zeros((1, 3), dtype=float), vertices])
    if jitter > 0.0:
        positions += rng.normal(scale=jitter, size=positions.shape)
    return problem.positions_to_vector(positions)


def optimize_icosahedral_scale(problem: MolecularCluster) -> tuple[np.ndarray, float, float]:
    rng = np.random.default_rng(SEED)
    base_positions = problem.vector_to_positions(make_icosahedral_seed(problem, rng, 0.0))
    scales = np.linspace(0.99, 1.03, 161)
    best_vector = problem.positions_to_vector(base_positions)
    best_energy = problem.energy(best_vector)
    best_scale = 1.0
    for scale in scales:
        candidate = problem.positions_to_vector(base_positions * scale)
        energy = problem.energy(candidate)
        if energy < best_energy:
            best_vector = candidate
            best_energy = energy
            best_scale = float(scale)
    return best_vector, float(best_energy), best_scale


def local_relax(
    problem: MolecularCluster,
    x0: np.ndarray,
    steps: int,
    initial_step: float,
) -> tuple[np.ndarray, list[float]]:
    x = problem.project(x0)
    best = x.copy()
    best_energy = problem.energy(best)
    step = initial_step
    trace = [best_energy]

    for _ in range(steps):
        grad = problem.gradient(x)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            trace.append(best_energy)
            continue
        proposal = problem.project(x - step * grad / grad_norm)
        proposal_energy = problem.energy(proposal)
        current_energy = problem.energy(x)
        if proposal_energy < current_energy:
            x = proposal
            step *= 1.015
            if proposal_energy < best_energy:
                best = proposal.copy()
                best_energy = proposal_energy
        else:
            step *= 0.55
        step = max(1e-4, min(step, 0.22))
        trace.append(best_energy)
    return best, trace


def optimize_multistart(problem: MolecularCluster, run_count: int, long_mode: bool) -> tuple[dict, list[dict]]:
    run_summaries: list[dict] = []
    best_payload: dict | None = None
    best_energy = float("inf")

    icosa_vector, icosa_energy, icosa_scale = optimize_icosahedral_scale(problem)
    best_payload = {
        "run": None,
        "refined_vector": icosa_vector.copy(),
        "refine_trace": [icosa_energy],
        "summary": {
            "index": 0,
            "seed": None,
            "seed_kind": "scaled_icosahedral_reference",
            "gonm_final_energy": icosa_energy,
            "refined_final_energy": icosa_energy,
            "runtime_ms": 0.0,
            "evals": 0,
            "trace_points": 1,
            "icosahedral_scale": icosa_scale,
        },
    }
    best_energy = icosa_energy

    optimizer = make_optimizer(problem)
    rng = np.random.default_rng(SEED)
    seeds = [SEED + 97 * k for k in range(run_count)]

    icosa_jitters = [0.0, 0.05, 0.09, 0.14]
    if long_mode:
        icosa_jitters.extend([0.18, 0.24, 0.30, 0.36])

    seed_bank = [make_icosahedral_seed(problem, rng, jitter=j) for j in icosa_jitters]
    for _ in range(max(0, run_count - len(seed_bank))):
        seed_bank.append(problem.random_geometry(rng, scale=4.0 * rng.uniform(0.75, 1.15)))

    for index, (seed, x0) in enumerate(zip(seeds, seed_bank), start=1):
        run = optimizer.optimize(seed=seed)
        refined, refine_trace = local_relax(
            problem,
            run.final_best_positions.reshape(-1),
            steps=4200 if long_mode else 1800,
            initial_step=0.06,
        )
        refined_energy = problem.energy(refined)
        summary = {
            "index": index,
            "seed": seed,
            "seed_kind": "icosahedral" if index <= len(icosa_jitters) else "random",
            "gonm_final_energy": float(run.final_true_energy),
            "refined_final_energy": float(refined_energy),
            "runtime_ms": float(run.runtime_ms),
            "evals": int(run.evals),
            "trace_points": len(refine_trace),
        }
        run_summaries.append(summary)
        if refined_energy < best_energy:
            best_energy = refined_energy
            best_payload = {
                "run": run,
                "refined_vector": refined.copy(),
                "refine_trace": refine_trace,
                "summary": summary,
            }

    if best_payload is None:
        raise RuntimeError("No multistart result was produced.")
    return best_payload, run_summaries


def pairwise_distances(positions: np.ndarray) -> np.ndarray:
    dists = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dists.append(np.linalg.norm(positions[i] - positions[j]))
    return np.asarray(dists, dtype=float)


def coordination_numbers(positions: np.ndarray, threshold: float = NEIGHBOR_THRESHOLD) -> list[int]:
    counts: list[int] = []
    pos = np.asarray(positions, dtype=float)
    for i in range(len(pos)):
        count = 0
        for j in range(len(pos)):
            if i == j:
                continue
            if np.linalg.norm(pos[i] - pos[j]) <= threshold:
                count += 1
        counts.append(count)
    return counts


def radial_shell_signature(positions: np.ndarray) -> list[float]:
    radii = np.linalg.norm(np.asarray(positions, dtype=float), axis=1)
    return [float(value) for value in np.sort(radii)]


def draw_cluster(ax, positions: np.ndarray, title: str, threshold: float = NEIGHBOR_THRESHOLD) -> None:
    pos = np.asarray(positions, dtype=float)
    ax.set_title(title)
    radii = np.linalg.norm(pos, axis=1)
    ax.scatter(
        pos[:, 0],
        pos[:, 1],
        pos[:, 2],
        s=95,
        c=radii,
        cmap="Blues",
        edgecolors="black",
        linewidths=0.5,
    )
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            dist = np.linalg.norm(pos[i] - pos[j])
            if dist <= threshold:
                ax.plot(
                    [pos[i, 0], pos[j, 0]],
                    [pos[i, 1], pos[j, 1]],
                    [pos[i, 2], pos[j, 2]],
                    color="#64748b",
                    alpha=0.42,
                    lw=1.15,
                )
    lim = max(6.0, float(np.max(np.abs(pos))) + 0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel("x [A]")
    ax.set_ylabel("y [A]")
    ax.set_zlabel("z [A]")


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

    draw_cluster(ax_phase1, phase1_pos, "Melhor estrutura estrutural de argonio")
    draw_cluster(ax_final, final_pos, "Estrutura atomica final Ar13 com GONM")

    ax_trace.plot(trace, color="#2563eb", lw=2.0)
    ax_trace.set_title("Trajetoria da energia verdadeira")
    ax_trace.set_xlabel("iteracao agregada")
    ax_trace.set_ylabel("energia [eV]")
    ax_trace.grid(True, linestyle=":", alpha=0.5)

    ax_text.axis("off")
    ax_text.text(
        0.0,
        1.0,
        "\n".join(
            [
                "GONM | atomic structure with real element",
                "",
                f"elemento = {result['element']}",
                f"atomos = {result['atom_count']}",
                f"dimensoes = {result['dimensions']}",
                f"sigma = {result['sigma_angstrom']:.3f} A",
                f"epsilon = {result['epsilon_ev']:.4f} eV",
                f"avaliacoes = {result['evals']}",
                f"tempo = {result['runtime_ms']:.1f} ms",
                f"rodadas = {result['run_count']}",
                "",
                f"energia inicial = {result['initial_best_energy']:.4f} eV",
                f"energia fase 1 = {result['phase1_best_energy']:.4f} eV",
                f"energia final = {result['final_best_energy']:.4f} eV",
                f"ganho vs fase 1 = {result['energy_gain_vs_phase1']:.4f} eV",
                f"erro vs alvo LJ13 = {result['gap_to_expected_ground_state_ev']:.6f} eV",
                "",
                f"distancia media = {result['final_pair_distance_mean']:.4f} A",
                f"distancia minima = {result['final_pair_distance_min']:.4f} A",
                f"distancia maxima = {result['final_pair_distance_max']:.4f} A",
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

    fig.suptitle("GONM | Estrutura atomica de argonio Ar13", fontsize=16)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    return fig


def save_outputs(result: dict, fig: plt.Figure) -> tuple[Path, Path, Path]:
    out_dir = ROOT / "publications" / "T07_GONM" / "results" / "gonm_atomic_structure_argon13"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "summary.json"
    md_path = out_dir / "summary.md"
    image_path = out_dir / "gonm_atomic_structure_argon13.png"

    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# GONM | Atomic Structure Ar13",
                "",
                f"- element: `{result['element']}`",
                f"- atoms: `{result['atom_count']}`",
                f"- dimensions: `{result['dimensions']}`",
                f"- sigma: `{result['sigma_angstrom']:.3f} A`",
                f"- epsilon: `{result['epsilon_ev']:.4f} eV`",
                f"- evaluations: `{result['evals']}`",
                f"- runtime: `{result['runtime_ms']:.1f} ms`",
                f"- multistart runs: `{result['run_count']}`",
                "",
                "## Energy",
                "",
                f"- initial best: `{result['initial_best_energy']:.4f} eV`",
                f"- phase 1 best: `{result['phase1_best_energy']:.4f} eV`",
                f"- final best: `{result['final_best_energy']:.4f} eV`",
                f"- gain vs phase 1: `{result['energy_gain_vs_phase1']:.4f} eV`",
                f"- expected LJ13 ground-state reference: `{EXPECTED_GROUND_STATE_EV:.6f} eV`",
                f"- gap to reference: `{result['gap_to_expected_ground_state_ev']:.6f} eV`",
                "",
                "## Geometric indicators",
                "",
                f"- mean pair distance: `{result['final_pair_distance_mean']:.4f} A`",
                f"- min pair distance: `{result['final_pair_distance_min']:.4f} A`",
                f"- max pair distance: `{result['final_pair_distance_max']:.4f} A`",
                f"- coordination numbers: `{result['final_coordination_numbers']}`",
                "",
                "## Files",
                "",
                "- image: `publications/T07_GONM/results/gonm_atomic_structure_argon13/gonm_atomic_structure_argon13.png`",
                "- summary: `publications/T07_GONM/results/gonm_atomic_structure_argon13/summary.json`",
            ]
        ),
        encoding="utf-8",
    )
    fig.savefig(image_path, dpi=170)
    return json_path, md_path, image_path


def main() -> None:
    problem = make_problem()
    long_mode, run_count = parse_args()
    best_payload, run_summaries = optimize_multistart(problem, run_count=run_count, long_mode=long_mode)
    run = best_payload["run"]
    final_vec = problem.project(best_payload["refined_vector"])
    final_pos = problem.vector_to_positions(final_vec)
    final_dists = pairwise_distances(final_pos)
    total_runtime_ms = sum(item["runtime_ms"] for item in run_summaries)
    total_evals = sum(item["evals"] for item in run_summaries)
    refine_trace = [float(value) for value in best_payload["refine_trace"]]
    gonm_trace = [float(value) for value in run.trace_true_energy] if run is not None else []
    full_trace = gonm_trace + refine_trace
    final_energy = problem.energy(final_vec)
    phase1_positions = (
        np.asarray(run.phase1_best_positions, dtype=float).tolist()
        if run is not None
        else final_pos.tolist()
    )
    initial_best_energy = run.initial_best_energy if run is not None else final_energy
    phase1_best_energy = run.phase1_best_energy if run is not None else final_energy

    result = {
        "seed": run.seed if run is not None else SEED,
        "long_mode": long_mode,
        "run_count": run_count,
        "element": ELEMENT,
        "atom_count": problem.atom_count,
        "dimensions": problem.dimensions,
        "sigma_angstrom": SIGMA_ANGSTROM,
        "epsilon_ev": EPSILON_EV,
        "expected_ground_state_ev": EXPECTED_GROUND_STATE_EV,
        "evals": total_evals,
        "runtime_ms": total_runtime_ms,
        "initial_best_energy": initial_best_energy,
        "phase1_best_energy": phase1_best_energy,
        "final_best_energy": final_energy,
        "energy_gain_vs_phase1": phase1_best_energy - final_energy,
        "gap_to_expected_ground_state_ev": final_energy - EXPECTED_GROUND_STATE_EV,
        "phase1_best_positions": phase1_positions,
        "final_best_positions": final_pos.tolist(),
        "trace_true_energy": full_trace,
        "final_pair_distance_mean": float(np.mean(final_dists)),
        "final_pair_distance_min": float(np.min(final_dists)),
        "final_pair_distance_max": float(np.max(final_dists)),
        "final_coordination_numbers": coordination_numbers(final_pos),
        "final_radial_shell_signature": radial_shell_signature(final_pos),
        "best_run_summary": best_payload["summary"],
        "multistart_runs": run_summaries,
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
