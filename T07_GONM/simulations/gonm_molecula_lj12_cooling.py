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
BASE_NOISE_STD = 0.08
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


def pairwise_distances(pos: np.ndarray) -> np.ndarray:
    dists = []
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            dists.append(np.linalg.norm(pos[i] - pos[j]))
    return np.asarray(dists, dtype=float)


def radial_signature(pos: np.ndarray) -> np.ndarray:
    return np.sort(np.linalg.norm(pos, axis=1))


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
    def __init__(self, base_noise_std: float, rng: np.random.Generator, mode: str):
        self.base_noise_std = base_noise_std
        self.rng = rng
        self.mode = mode
        self.evals = 0

    def true(self, x: np.ndarray) -> float:
        return lennard_jones_energy(vector_to_positions(x))

    def noise_scale(self, stage: str, step: int) -> float:
        if self.mode == "fixed":
            return self.base_noise_std
        if stage == "phase1":
            return self.base_noise_std
        if stage == "phase2":
            frac = min(max(step / max(PHASE2_STEPS - 1, 1), 0.0), 1.0)
            if frac < 0.35:
                return self.base_noise_std
            if frac < 0.65:
                return 0.04
            if frac < 0.88:
                return 0.01
            return 0.0
        if stage == "phase3":
            frac = min(max(step / max(PHASE3_STEPS - 1, 1), 0.0), 1.0)
            return 0.01 * (1.0 - frac)
        return self.base_noise_std

    def noisy(self, x: np.ndarray, stage: str = "phase1", step: int = 0) -> float:
        self.evals += 1
        scale = self.noise_scale(stage, step)
        return self.true(x) + float(self.rng.normal(scale=scale))

    def averaged(self, x: np.ndarray, n_samples: int = 3, stage: str = "phase1", step: int = 0) -> float:
        vals = [self.noisy(x, stage=stage, step=step) for _ in range(n_samples)]
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
        sig = radial_signature(vector_to_positions(x))
        if any(np.linalg.norm(sig - other) <= tol for other in signatures):
            continue
        signatures.append(sig)
        clusters.append(x)
    return clusters


def local_refine(oracle: NoisyOracle, x0: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, list[float]]:
    x = x0.copy()
    velocity = np.zeros_like(x)
    best_x = x.copy()
    best_score = oracle.averaged(x, 5, stage="phase2", step=0)
    trace = [oracle.true(x)]
    step_size = 0.026
    momentum = 0.84

    for t in range(PHASE2_STEPS):
        pos = vector_to_positions(x)
        grad = positions_to_vector(lennard_jones_gradient(pos))
        grad_norm = np.linalg.norm(grad) + 1e-9
        thermal = rng.normal(size=(ATOM_COUNT, 3))
        thermal -= np.mean(thermal, axis=0, keepdims=True)
        thermal = thermal.reshape(-1)
        thermal *= 0.012 * np.exp(-t / 45.0)

        velocity = momentum * velocity - step_size * grad / grad_norm + thermal
        proposal = positions_to_vector(vector_to_positions(x + velocity))

        old_score = oracle.averaged(x, 3, stage="phase2", step=t)
        new_score = oracle.averaged(proposal, 3, stage="phase2", step=t)
        temperature = 0.12 + 0.02 * np.exp(-t / 35.0)
        if new_score < old_score or rng.random() < np.exp(-(new_score - old_score) / temperature):
            x = proposal
            if new_score < best_score:
                best_score = new_score
                best_x = x.copy()
        else:
            velocity *= 0.45
            step_size *= 0.995

        trace.append(oracle.true(best_x))

    return best_x, trace


def robust_terminal_search(oracle: NoisyOracle, x0: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, list[float]]:
    x = x0.copy()
    best_x = x.copy()
    best_score = oracle.averaged(x, 7, stage="phase3", step=0)
    trace = [oracle.true(x)]
    radii = [0.18, 0.11, 0.07, 0.04]
    total_steps = 0

    for radius in radii:
        for _ in range(PHASE3_STEPS // len(radii)):
            candidates = []
            for _ in range(9):
                direction = rng.normal(size=(ATOM_COUNT, 3))
                direction -= np.mean(direction, axis=0, keepdims=True)
                direction = direction.reshape(-1)
                direction /= max(np.linalg.norm(direction), 1e-9)
                proposal = positions_to_vector(vector_to_positions(x + radius * direction))
                candidates.append(proposal)

            local_best = x
            local_score = best_score
            for proposal in candidates:
                score = oracle.averaged(proposal, 5, stage="phase3", step=total_steps)
                if score < local_score:
                    local_score = score
                    local_best = proposal

            if local_score < best_score:
                x = local_best
                best_x = local_best.copy()
                best_score = local_score
            trace.append(oracle.true(best_x))
            total_steps += 1

    return best_x, trace


def run_variant(mode: str, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    oracle = NoisyOracle(BASE_NOISE_STD, rng, mode=mode)
    start = perf_counter()

    initial_population = [random_geometry(rng, BOX_SCALE * rng.uniform(0.75, 1.25)) for _ in range(POPULATION)]
    scored = [(oracle.averaged(x, 3, stage="phase1", step=0), x) for x in initial_population]
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
        refined_runs.append((oracle.true(terminal_x), terminal_x, phase2_trace + phase3_trace))

    refined_runs.sort(key=lambda item: item[0])
    best_true_energy, best_final, best_trace = refined_runs[0]
    final_pos = vector_to_positions(best_final)
    final_dists = pairwise_distances(final_pos)
    runtime_ms = (perf_counter() - start) * 1000.0
    initial_best = min(initial_population, key=oracle.true)

    return {
        "mode": mode,
        "seed": seed,
        "noise_std_initial": BASE_NOISE_STD,
        "evals": oracle.evals,
        "runtime_ms": runtime_ms,
        "initial_best_energy": oracle.true(initial_best),
        "phase1_best_energy": phase1_best_energy,
        "final_best_energy": best_true_energy,
        "energy_gain_vs_phase1": phase1_best_energy - best_true_energy,
        "phase1_best_positions": vector_to_positions(phase1_best).tolist(),
        "final_best_positions": final_pos.tolist(),
        "trace_true_energy": best_trace,
        "final_pair_distance_mean": float(np.mean(final_dists)),
        "final_pair_distance_min": float(np.min(final_dists)),
        "final_pair_distance_max": float(np.max(final_dists)),
    }


def draw_bonds(ax, pos: np.ndarray, threshold: float = 1.55) -> None:
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            if np.linalg.norm(pos[i] - pos[j]) <= threshold:
                ax.plot(
                    [pos[i, 0], pos[j, 0]],
                    [pos[i, 1], pos[j, 1]],
                    [pos[i, 2], pos[j, 2]],
                    color="gray",
                    alpha=0.45,
                    lw=1.1,
                )


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


def render_comparison(fixed_result: dict, cooled_result: dict) -> plt.Figure:
    fixed_pos = np.asarray(fixed_result["final_best_positions"], dtype=float)
    cooled_pos = np.asarray(cooled_result["final_best_positions"], dtype=float)
    fixed_trace = np.asarray(fixed_result["trace_true_energy"], dtype=float)
    cooled_trace = np.asarray(cooled_result["trace_true_energy"], dtype=float)

    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1.2])
    ax_fixed = fig.add_subplot(gs[0, 0], projection="3d")
    ax_cooled = fig.add_subplot(gs[0, 1], projection="3d")
    ax_trace = fig.add_subplot(gs[1, 0])
    ax_text = fig.add_subplot(gs[1, 1])

    setup_3d_axis(ax_fixed, "LJ-12 | ruido fixo", fixed_pos)
    setup_3d_axis(ax_cooled, "LJ-12 | resfriamento termico", cooled_pos)

    ax_trace.plot(fixed_trace, color="tab:red", lw=2.0, label="ruido fixo")
    ax_trace.plot(cooled_trace, color="tab:blue", lw=2.0, label="cooling")
    ax_trace.set_title("Comparacao da trajetoria de energia")
    ax_trace.set_xlabel("iteracao agregada")
    ax_trace.set_ylabel("energia")
    ax_trace.grid(True, linestyle=":", alpha=0.5)
    ax_trace.legend()

    ax_text.axis("off")
    lines = [
        "LJ-12 | quente vs resfriada",
        "",
        f"ruido inicial = {BASE_NOISE_STD:.3f}",
        f"avaliacoes fixo = {fixed_result['evals']}",
        f"avaliacoes cooling = {cooled_result['evals']}",
        "",
        f"energia final fixo = {fixed_result['final_best_energy']:.4f}",
        f"energia final cooling = {cooled_result['final_best_energy']:.4f}",
        f"delta = {fixed_result['final_best_energy'] - cooled_result['final_best_energy']:.4f}",
        "",
        f"ganho fixo vs fase 1 = {fixed_result['energy_gain_vs_phase1']:.4f}",
        f"ganho cooling vs fase 1 = {cooled_result['energy_gain_vs_phase1']:.4f}",
        "",
        f"distancia minima fixa = {fixed_result['final_pair_distance_min']:.4f}",
        f"distancia minima cooling = {cooled_result['final_pair_distance_min']:.4f}",
    ]
    ax_text.text(0.0, 1.0, "\n".join(lines), ha="left", va="top", fontsize=11, family="monospace")

    fig.suptitle("GONM | Lennard-Jones 12 atomos com resfriamento termico", fontsize=16)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    return fig


def save_outputs(payload: dict) -> tuple[Path, Path]:
    out_dir = Path(__file__).resolve().parent / "experiment_results" / "gonm_lj12_cooling"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "summary.json"
    md_path = out_dir / "summary.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fixed = payload["fixed"]
    cooled = payload["cooling"]
    md_text = "\n".join(
        [
            "# GONM | LJ-12 com resfriamento termico",
            "",
            f"- energia final ruido fixo: `{fixed['final_best_energy']:.4f}`",
            f"- energia final cooling: `{cooled['final_best_energy']:.4f}`",
            f"- delta: `{fixed['final_best_energy'] - cooled['final_best_energy']:.4f}`",
            "",
            "## Ruido fixo",
            f"- avaliacoes: `{fixed['evals']}`",
            f"- ganho vs fase 1: `{fixed['energy_gain_vs_phase1']:.4f}`",
            f"- distancia minima final: `{fixed['final_pair_distance_min']:.4f}`",
            "",
            "## Cooling",
            f"- avaliacoes: `{cooled['evals']}`",
            f"- ganho vs fase 1: `{cooled['energy_gain_vs_phase1']:.4f}`",
            f"- distancia minima final: `{cooled['final_pair_distance_min']:.4f}`",
            "",
            "## Arquivos",
            "- imagem: `simulacao/prints/gonm_lj12_cooling.png`",
            "- resumo bruto: `simulacao/experiment_results/gonm_lj12_cooling/summary.json`",
        ]
    )
    md_path.write_text(md_text, encoding="utf-8")
    return json_path, md_path


def main() -> None:
    fixed = run_variant("fixed")
    cooling = run_variant("cooling")
    payload = {"fixed": fixed, "cooling": cooling}
    json_path, md_path = save_outputs(payload)
    print(f"Resumo salvo em {json_path}")
    print(f"Resumo legivel salvo em {md_path}")
    fig = render_comparison(fixed, cooling)
    finish_figure(fig, __file__, "gonm_lj12_cooling")


if __name__ == "__main__":
    main()
