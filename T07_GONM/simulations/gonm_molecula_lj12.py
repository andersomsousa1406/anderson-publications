import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import matplotlib

if "--save" in sys.argv or "--no-show" in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from csd_plot_utils import finish_figure


ATOM_COUNT = 12
DIM = ATOM_COUNT * 3
SIGMA = 1.0
EPSILON = 1.0
NOISE_STD = 0.08
SOFTEN = 0.04
POPULATION = 64
PHASE1_KEEP = 8
PHASE2_STEPS = 140
PHASE3_STEPS = 60
BOX_SCALE = 2.8
SEED = 17


def vector_to_positions(x: np.ndarray) -> np.ndarray:
    pos = np.asarray(x, dtype=float).reshape(ATOM_COUNT, 3)
    pos = pos - np.mean(pos, axis=0, keepdims=True)
    return pos


def positions_to_vector(pos: np.ndarray) -> np.ndarray:
    pos = np.asarray(pos, dtype=float)
    pos = pos - np.mean(pos, axis=0, keepdims=True)
    return pos.reshape(-1)


def radial_signature(pos: np.ndarray) -> np.ndarray:
    radii = np.linalg.norm(pos, axis=1)
    return np.sort(radii)


def pairwise_distances(pos: np.ndarray) -> np.ndarray:
    dists = []
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            dists.append(np.linalg.norm(pos[i] - pos[j]))
    return np.asarray(dists, dtype=float)


def lennard_jones_energy(pos: np.ndarray) -> float:
    energy = 0.0
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            diff = pos[i] - pos[j]
            r = max(np.linalg.norm(diff), SOFTEN)
            sr6 = (SIGMA / r) ** 6
            energy += 4.0 * EPSILON * (sr6 * sr6 - sr6)
    return float(energy)


def lennard_jones_gradient(pos: np.ndarray) -> np.ndarray:
    grad = np.zeros_like(pos, dtype=float)
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            diff = pos[i] - pos[j]
            r = max(np.linalg.norm(diff), SOFTEN)
            inv_r = 1.0 / r
            sr6 = (SIGMA * inv_r) ** 6
            coeff = 24.0 * EPSILON * (2.0 * sr6 * sr6 - sr6) * inv_r * inv_r
            force = coeff * diff
            grad[i] += force
            grad[j] -= force
    grad -= np.mean(grad, axis=0, keepdims=True)
    return grad


class NoisyOracle:
    def __init__(self, noise_std: float, rng: np.random.Generator):
        self.noise_std = noise_std
        self.rng = rng
        self.evals = 0

    def true(self, x: np.ndarray) -> float:
        return lennard_jones_energy(vector_to_positions(x))

    def noisy(self, x: np.ndarray) -> float:
        self.evals += 1
        return self.true(x) + float(self.rng.normal(scale=self.noise_std))

    def averaged(self, x: np.ndarray, n_samples: int = 3) -> float:
        vals = [self.noisy(x) for _ in range(n_samples)]
        return float(np.mean(vals))


def random_geometry(rng: np.random.Generator, scale: float) -> np.ndarray:
    pos = rng.normal(size=(ATOM_COUNT, 3))
    pos -= np.mean(pos, axis=0, keepdims=True)
    rms = np.sqrt(np.mean(np.sum(pos * pos, axis=1)))
    pos *= scale / max(rms, 1e-9)
    return positions_to_vector(pos)


def cluster_candidates(candidates: list[np.ndarray], tol: float = 0.55) -> list[np.ndarray]:
    clusters: list[np.ndarray] = []
    signatures: list[np.ndarray] = []
    for x in candidates:
        pos = vector_to_positions(x)
        sig = radial_signature(pos)
        if any(np.linalg.norm(sig - other) <= tol for other in signatures):
            continue
        signatures.append(sig)
        clusters.append(x)
    return clusters


def local_refine(oracle: NoisyOracle, x0: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, list[float]]:
    x = x0.copy()
    velocity = np.zeros_like(x)
    best_x = x.copy()
    best_score = oracle.averaged(x, 5)
    energy_trace = [oracle.true(x)]
    step = 0.026
    momentum = 0.84

    for t in range(PHASE2_STEPS):
        pos = vector_to_positions(x)
        grad = positions_to_vector(lennard_jones_gradient(pos))
        grad_norm = np.linalg.norm(grad) + 1e-9
        thermal = rng.normal(size=(ATOM_COUNT, 3))
        thermal -= np.mean(thermal, axis=0, keepdims=True)
        thermal = thermal.reshape(-1)
        thermal *= 0.012 * np.exp(-t / 45.0)

        velocity = momentum * velocity - step * grad / grad_norm + thermal
        proposal = x + velocity
        proposal = positions_to_vector(vector_to_positions(proposal))

        old_score = oracle.averaged(x, 3)
        new_score = oracle.averaged(proposal, 3)
        if new_score < old_score or rng.random() < np.exp(-(new_score - old_score) / (0.12 + 0.02 * np.exp(-t / 35.0))):
            x = proposal
            if new_score < best_score:
                best_score = new_score
                best_x = x.copy()
        else:
            velocity *= 0.45
            step *= 0.995

        energy_trace.append(oracle.true(best_x))

    return best_x, energy_trace


def robust_terminal_search(oracle: NoisyOracle, x0: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, list[float]]:
    x = x0.copy()
    best_x = x.copy()
    best_score = oracle.averaged(x, 7)
    trace = [oracle.true(x)]
    radii = [0.18, 0.11, 0.07, 0.04]

    for radius in radii:
        for _ in range(PHASE3_STEPS // len(radii)):
            trial_dirs = [rng.normal(size=x.shape) for _ in range(9)]
            candidates = []
            for direction in trial_dirs:
                direction = direction.reshape(ATOM_COUNT, 3)
                direction -= np.mean(direction, axis=0, keepdims=True)
                direction = direction.reshape(-1)
                direction /= max(np.linalg.norm(direction), 1e-9)
                proposal = x + radius * direction
                proposal = positions_to_vector(vector_to_positions(proposal))
                candidates.append(proposal)

            local_best = x
            local_score = best_score
            for proposal in candidates:
                score = oracle.averaged(proposal, 5)
                if score < local_score:
                    local_score = score
                    local_best = proposal

            if local_score < best_score:
                x = local_best
                best_x = local_best.copy()
                best_score = local_score
            trace.append(oracle.true(best_x))

    return best_x, trace


def run_gonm_lj12(seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    oracle = NoisyOracle(NOISE_STD, rng)
    start = perf_counter()

    initial_population = [random_geometry(rng, BOX_SCALE * rng.uniform(0.75, 1.25)) for _ in range(POPULATION)]
    scored = []
    for x in initial_population:
        scored.append((oracle.averaged(x, 3), x))
    scored.sort(key=lambda item: item[0])

    elites = [x for _, x in scored[: PHASE1_KEEP * 2]]
    clustered = cluster_candidates(elites, tol=0.48)
    if len(clustered) < PHASE1_KEEP:
        clustered = elites[:PHASE1_KEEP]
    else:
        clustered = clustered[:PHASE1_KEEP]

    phase1_best = min(clustered, key=oracle.true)
    phase1_best_energy = oracle.true(phase1_best)

    refined_runs = []
    for x in clustered:
        refined_x, phase2_trace = local_refine(oracle, x, rng)
        terminal_x, phase3_trace = robust_terminal_search(oracle, refined_x, rng)
        refined_runs.append((oracle.true(terminal_x), x, refined_x, terminal_x, phase2_trace + phase3_trace))

    refined_runs.sort(key=lambda item: item[0])
    best_true_energy, best_seed, best_phase2, best_final, best_trace = refined_runs[0]
    runtime_ms = (perf_counter() - start) * 1000.0

    initial_best = min(initial_population, key=oracle.true)
    initial_best_energy = oracle.true(initial_best)

    return {
        "seed": seed,
        "atom_count": ATOM_COUNT,
        "dimensions": DIM,
        "noise_std": NOISE_STD,
        "population": POPULATION,
        "phase1_keep": len(clustered),
        "phase2_steps": PHASE2_STEPS,
        "phase3_steps": PHASE3_STEPS,
        "evals": oracle.evals,
        "runtime_ms": runtime_ms,
        "initial_best_energy": initial_best_energy,
        "phase1_best_energy": phase1_best_energy,
        "final_best_energy": best_true_energy,
        "energy_gain_vs_phase1": phase1_best_energy - best_true_energy,
        "energy_gain_vs_initial": initial_best_energy - best_true_energy,
        "initial_best_positions": vector_to_positions(initial_best).tolist(),
        "phase1_best_positions": vector_to_positions(phase1_best).tolist(),
        "final_best_positions": vector_to_positions(best_final).tolist(),
        "phase2_best_positions": vector_to_positions(best_phase2).tolist(),
        "trace_true_energy": best_trace,
        "final_pair_distance_mean": float(np.mean(pairwise_distances(vector_to_positions(best_final)))),
        "final_pair_distance_min": float(np.min(pairwise_distances(vector_to_positions(best_final)))),
        "final_pair_distance_max": float(np.max(pairwise_distances(vector_to_positions(best_final)))),
    }


def draw_bonds(ax, pos: np.ndarray, threshold: float = 1.55) -> None:
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            dist = np.linalg.norm(pos[i] - pos[j])
            if dist <= threshold:
                xs = [pos[i, 0], pos[j, 0]]
                ys = [pos[i, 1], pos[j, 1]]
                zs = [pos[i, 2], pos[j, 2]]
                ax.plot(xs, ys, zs, color="gray", alpha=0.45, lw=1.2)


def setup_3d_axis(ax, title: str, pos: np.ndarray) -> None:
    ax.set_title(title)
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=60, c=np.linspace(0.1, 0.9, len(pos)), cmap="viridis", edgecolors="black")
    draw_bonds(ax, pos)
    lim = max(2.8, float(np.max(np.abs(pos))) + 0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def render_result(result: dict) -> plt.Figure:
    initial_pos = np.asarray(result["phase1_best_positions"], dtype=float)
    final_pos = np.asarray(result["final_best_positions"], dtype=float)
    trace = np.asarray(result["trace_true_energy"], dtype=float)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1.2])
    ax_init = fig.add_subplot(gs[0, 0], projection="3d")
    ax_final = fig.add_subplot(gs[0, 1], projection="3d")
    ax_trace = fig.add_subplot(gs[1, 0])
    ax_text = fig.add_subplot(gs[1, 1])

    setup_3d_axis(ax_init, "Melhor geometria estrutural (fase 1)", initial_pos)
    setup_3d_axis(ax_final, "Geometria final GONM", final_pos)

    ax_trace.plot(trace, color="tab:blue", lw=2.0)
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
                "GONM como solucionador de potencial fisico",
                "",
                f"atomos = {result['atom_count']}",
                f"dimensoes = {result['dimensions']}",
                f"ruido = {result['noise_std']:.3f}",
                f"avaliacoes = {result['evals']}",
                f"tempo = {result['runtime_ms']:.1f} ms",
                "",
                f"energia inicial = {result['initial_best_energy']:.4f}",
                f"energia fase 1 = {result['phase1_best_energy']:.4f}",
                f"energia final = {result['final_best_energy']:.4f}",
                f"ganho vs fase 1 = {result['energy_gain_vs_phase1']:.4f}",
                "",
                f"distancia media final = {result['final_pair_distance_mean']:.4f}",
                f"distancia minima final = {result['final_pair_distance_min']:.4f}",
                f"distancia maxima final = {result['final_pair_distance_max']:.4f}",
            ]
        ),
        ha="left",
        va="top",
        fontsize=11,
        family="monospace",
    )

    fig.suptitle("GONM | Molecula sintetica LJ-12 em 36D", fontsize=16)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    return fig


def save_result_json(result: dict) -> Path:
    out_dir = Path(__file__).resolve().parent / "experiment_results" / "gonm_lj12"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    result = run_gonm_lj12()
    json_path = save_result_json(result)
    print(f"Resumo salvo em {json_path}")
    fig = render_result(result)
    finish_figure(fig, __file__, "gonm_lj12")


if __name__ == "__main__":
    main()
