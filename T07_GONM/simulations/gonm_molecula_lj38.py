import json
import sys
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import matplotlib

if "--save" in sys.argv or "--no-show" in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from csd_plot_utils import finish_figure


ATOM_COUNT = 38
DIM = ATOM_COUNT * 3
SIGMA = 1.0
EPSILON = 1.0
NOISE_STD = 0.05
SOFTEN = 0.04
POPULATION = 72
PHASE1_KEEP = 10
PHASE2_STEPS = 120
PHASE3_STEPS = 48
BOX_SCALE = 4.6
SEED = 38
BOND_THRESHOLD = 1.42
BOUND_LIMIT = None


def vector_to_positions(x: np.ndarray) -> np.ndarray:
    pos = np.asarray(x, dtype=float).reshape(ATOM_COUNT, 3)
    pos = pos - np.mean(pos, axis=0, keepdims=True)
    if BOUND_LIMIT is not None:
        pos = np.clip(pos, -BOUND_LIMIT, BOUND_LIMIT)
    return pos


def positions_to_vector(pos: np.ndarray) -> np.ndarray:
    pos = np.asarray(pos, dtype=float)
    pos = pos - np.mean(pos, axis=0, keepdims=True)
    if BOUND_LIMIT is not None:
        pos = np.clip(pos, -BOUND_LIMIT, BOUND_LIMIT)
    return pos.reshape(-1)


def parse_runtime_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--population", type=int, default=POPULATION)
    parser.add_argument("--phase1-keep", type=int, default=PHASE1_KEEP)
    parser.add_argument("--phase2-steps", type=int, default=PHASE2_STEPS)
    parser.add_argument("--phase3-steps", type=int, default=PHASE3_STEPS)
    parser.add_argument("--noise-std", type=float, default=NOISE_STD)
    parser.add_argument("--box-scale", type=float, default=BOX_SCALE)
    parser.add_argument("--bounds", type=float, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--resume-from", type=str, default="")
    parser.add_argument("--resume-copies", type=int, default=8)
    args, _ = parser.parse_known_args()
    return args


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


def load_resume_state(path_str: str) -> tuple[np.ndarray, dict] | tuple[None, None]:
    if not path_str:
        return None, None
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de resume nao encontrado: {path}")

    if path.suffix.lower() == ".npz":
        data = np.load(path)
        x = np.asarray(data["final_positions"], dtype=float).reshape(-1)
        meta = {"source": str(path), "kind": "npz"}
        return positions_to_vector(x.reshape(ATOM_COUNT, 3)), meta

    payload = json.loads(path.read_text(encoding="utf-8"))
    if "final_best_positions" in payload:
        pos = np.asarray(payload["final_best_positions"], dtype=float)
        return positions_to_vector(pos), {"source": str(path), "kind": "summary_json"}
    raise ValueError(f"Arquivo de resume sem final_best_positions: {path}")


def cluster_candidates(candidates: list[np.ndarray], tol: float = 0.95) -> list[np.ndarray]:
    clusters: list[np.ndarray] = []
    signatures: list[np.ndarray] = []
    for x in candidates:
        sig = radial_signature(vector_to_positions(x))
        if any(np.linalg.norm(sig - s) <= tol for s in signatures):
            continue
        signatures.append(sig)
        clusters.append(x)
    return clusters


def local_refine(oracle: NoisyOracle, x0: np.ndarray, rng: np.random.Generator, phase2_steps: int) -> tuple[np.ndarray, list[float]]:
    x = x0.copy()
    velocity = np.zeros_like(x)
    best_x = x.copy()
    best_score = oracle.averaged(x, 5)
    trace = [oracle.true(x)]
    step = 0.018
    momentum = 0.86

    for t in range(phase2_steps):
        pos = vector_to_positions(x)
        grad = positions_to_vector(lennard_jones_gradient(pos))
        grad_norm = np.linalg.norm(grad) + 1e-9

        thermal = rng.normal(size=(ATOM_COUNT, 3))
        thermal -= np.mean(thermal, axis=0, keepdims=True)
        thermal = thermal.reshape(-1)
        thermal *= 0.010 * np.exp(-t / 55.0)

        velocity = momentum * velocity - step * grad / grad_norm + thermal
        proposal = positions_to_vector(vector_to_positions(x + velocity))

        old_score = oracle.averaged(x, 3)
        new_score = oracle.averaged(proposal, 3)
        temperature = 0.09 + 0.02 * np.exp(-t / 50.0)
        if new_score < old_score or rng.random() < np.exp(-(new_score - old_score) / temperature):
            x = proposal
            if new_score < best_score:
                best_score = new_score
                best_x = x.copy()
        else:
            velocity *= 0.4
            step *= 0.996

        trace.append(oracle.true(best_x))

    return best_x, trace


def robust_terminal_search(oracle: NoisyOracle, x0: np.ndarray, rng: np.random.Generator, phase3_steps: int) -> tuple[np.ndarray, list[float]]:
    x = x0.copy()
    best_x = x.copy()
    best_score = oracle.averaged(x, 7)
    trace = [oracle.true(x)]
    radii = [0.12, 0.08, 0.05, 0.03]

    for radius in radii:
        for _ in range(max(1, phase3_steps // len(radii))):
            candidates = []
            for _ in range(12):
                direction = rng.normal(size=(ATOM_COUNT, 3))
                direction -= np.mean(direction, axis=0, keepdims=True)
                direction = direction.reshape(-1)
                direction /= max(np.linalg.norm(direction), 1e-9)
                proposal = positions_to_vector(vector_to_positions(x + radius * direction))
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


def run_gonm_lj38(
    seed: int = SEED,
    population: int = POPULATION,
    phase1_keep: int = PHASE1_KEEP,
    phase2_steps: int = PHASE2_STEPS,
    phase3_steps: int = PHASE3_STEPS,
    noise_std: float = NOISE_STD,
    box_scale: float = BOX_SCALE,
    bounds: float | None = None,
    resume_from: str = "",
    resume_copies: int = 8,
) -> dict:
    global BOUND_LIMIT
    BOUND_LIMIT = bounds
    rng = np.random.default_rng(seed)
    oracle = NoisyOracle(noise_std, rng)
    start = perf_counter()

    resume_x, resume_meta = load_resume_state(resume_from)

    initial_population = [random_geometry(rng, box_scale * rng.uniform(0.75, 1.30)) for _ in range(population)]
    if resume_x is not None:
        n_inject = min(max(resume_copies, 1), population)
        injected = [resume_x.copy()]
        for k in range(n_inject - 1):
            jitter_scale = 0.03 + 0.01 * k
            jitter = rng.normal(scale=jitter_scale, size=resume_x.shape)
            injected.append(positions_to_vector(vector_to_positions(resume_x + jitter)))
        initial_population[:n_inject] = injected

    scored = [(oracle.averaged(x, 3), x) for x in initial_population]
    scored.sort(key=lambda item: item[0])

    elites = [x for _, x in scored[: phase1_keep * 2]]
    clustered = cluster_candidates(elites, tol=0.90)
    if len(clustered) < phase1_keep:
        clustered = elites[:phase1_keep]
    else:
        clustered = clustered[:phase1_keep]

    phase1_best = min(clustered, key=oracle.true)
    phase1_best_energy = oracle.true(phase1_best)

    refined_runs = []
    for x in clustered:
        refined_x, phase2_trace = local_refine(oracle, x, rng, phase2_steps)
        terminal_x, phase3_trace = robust_terminal_search(oracle, refined_x, rng, phase3_steps)
        refined_runs.append((oracle.true(terminal_x), x, refined_x, terminal_x, phase2_trace + phase3_trace))

    refined_runs.sort(key=lambda item: item[0])
    best_true_energy, best_seed, best_phase2, best_final, best_trace = refined_runs[0]
    runtime_ms = (perf_counter() - start) * 1000.0

    initial_best = min(initial_population, key=oracle.true)
    initial_best_energy = oracle.true(initial_best)
    final_pos = vector_to_positions(best_final)
    final_dists = pairwise_distances(final_pos)

    return {
        "seed": seed,
        "atom_count": ATOM_COUNT,
        "dimensions": DIM,
        "noise_std": noise_std,
        "bounds": bounds,
        "population": population,
        "phase1_keep": len(clustered),
        "phase2_steps": phase2_steps,
        "phase3_steps": phase3_steps,
        "evals": oracle.evals,
        "runtime_ms": runtime_ms,
        "initial_best_energy": initial_best_energy,
        "phase1_best_energy": phase1_best_energy,
        "final_best_energy": best_true_energy,
        "energy_gain_vs_phase1": phase1_best_energy - best_true_energy,
        "energy_gain_vs_initial": initial_best_energy - best_true_energy,
        "phase1_best_positions": vector_to_positions(phase1_best).tolist(),
        "phase2_best_positions": vector_to_positions(best_phase2).tolist(),
        "final_best_positions": final_pos.tolist(),
        "trace_true_energy": best_trace,
        "final_pair_distance_mean": float(np.mean(final_dists)),
        "final_pair_distance_min": float(np.min(final_dists)),
        "final_pair_distance_max": float(np.max(final_dists)),
        "resume_from": resume_meta,
    }


def draw_bonds(ax, pos: np.ndarray, threshold: float = BOND_THRESHOLD) -> None:
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            dist = np.linalg.norm(pos[i] - pos[j])
            if dist <= threshold:
                ax.plot(
                    [pos[i, 0], pos[j, 0]],
                    [pos[i, 1], pos[j, 1]],
                    [pos[i, 2], pos[j, 2]],
                    color="gray",
                    alpha=0.32,
                    lw=0.9,
                )


def setup_3d_axis(ax, title: str, pos: np.ndarray) -> None:
    ax.set_title(title)
    ax.scatter(
        pos[:, 0],
        pos[:, 1],
        pos[:, 2],
        s=34,
        c=np.linspace(0.05, 0.95, len(pos)),
        cmap="plasma",
        edgecolors="black",
        linewidths=0.35,
    )
    draw_bonds(ax, pos)
    lim = max(3.8, float(np.max(np.abs(pos))) + 0.8)
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
    gs = fig.add_gridspec(2, 2, height_ratios=[2.3, 1.2])
    ax_phase1 = fig.add_subplot(gs[0, 0], projection="3d")
    ax_final = fig.add_subplot(gs[0, 1], projection="3d")
    ax_trace = fig.add_subplot(gs[1, 0])
    ax_text = fig.add_subplot(gs[1, 1])

    setup_3d_axis(ax_phase1, "Melhor geometria estrutural (fase 1)", phase1_pos)
    setup_3d_axis(ax_final, "Geometria final GONM | LJ-38", final_pos)

    ax_trace.plot(trace, color="tab:blue", lw=2.0)
    ax_trace.set_title("Trajetoria da energia verdadeira")
    ax_trace.set_xlabel("iteracao agregada")
    ax_trace.set_ylabel("energia")
    ax_trace.grid(True, linestyle=":", alpha=0.5)

    ax_text.axis("off")
    resume_label = result["resume_from"]["source"] if result.get("resume_from") else "nenhum"
    ax_text.text(
        0.0,
        1.0,
        "\n".join(
            [
                "LJ-38 | desafio fisico real",
                "",
                f"atomos = {result['atom_count']}",
                f"dimensoes = {result['dimensions']}",
                f"ruido termico = {result['noise_std']:.3f}",
                f"bounds = {result['bounds']}",
                f"resume = {resume_label}",
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
            ]
        ),
        ha="left",
        va="top",
        fontsize=11,
        family="monospace",
    )

    fig.suptitle("GONM | Cluster molecular Lennard-Jones 38 atomos", fontsize=16)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    return fig


def save_outputs(result: dict, tag: str) -> tuple[Path, Path, Path]:
    stem = "gonm_lj38" if not tag else f"gonm_lj38_{tag}"
    out_dir = Path(__file__).resolve().parent / "experiment_results" / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "summary.json"
    md_path = out_dir / "summary.md"
    state_path = out_dir / "state.npz"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(state_path, final_positions=np.asarray(result["final_best_positions"], dtype=float))

    resume_label = result["resume_from"]["source"] if result.get("resume_from") else "nenhum"
    md_text = "\n".join(
        [
            "# GONM | Cluster molecular LJ-38",
            "",
            f"- atomos: `{result['atom_count']}`",
            f"- dimensoes: `{result['dimensions']}`",
            f"- ruido termico: `{result['noise_std']}`",
            f"- bounds: `{result['bounds']}`",
            f"- resume: `{resume_label}`",
            f"- avaliacoes: `{result['evals']}`",
            f"- tempo: `{result['runtime_ms']:.1f} ms`",
            "",
            "## Energia",
            "",
            f"- melhor energia inicial: `{result['initial_best_energy']:.4f}`",
            f"- melhor energia apos fase estrutural: `{result['phase1_best_energy']:.4f}`",
            f"- melhor energia final: `{result['final_best_energy']:.4f}`",
            f"- ganho sobre fase 1: `{result['energy_gain_vs_phase1']:.4f}`",
            "",
            "## Geometria final",
            "",
            f"- distancia media entre pares: `{result['final_pair_distance_mean']:.4f}`",
            f"- distancia minima entre pares: `{result['final_pair_distance_min']:.4f}`",
            f"- distancia maxima entre pares: `{result['final_pair_distance_max']:.4f}`",
            "",
            "## Arquivos",
            "",
            f"- imagem: `simulacao/prints/{stem}.png`",
            f"- resumo bruto: `simulacao/experiment_results/{stem}/summary.json`",
            f"- estado final: `simulacao/experiment_results/{stem}/state.npz`",
        ]
    )
    md_path.write_text(md_text, encoding="utf-8")
    return json_path, md_path, state_path


def main() -> None:
    args = parse_runtime_args()
    result = run_gonm_lj38(
        seed=args.seed,
        population=args.population,
        phase1_keep=args.phase1_keep,
        phase2_steps=args.phase2_steps,
        phase3_steps=args.phase3_steps,
        noise_std=args.noise_std,
        box_scale=args.box_scale,
        bounds=args.bounds,
        resume_from=args.resume_from,
        resume_copies=args.resume_copies,
    )
    json_path, md_path, state_path = save_outputs(result, args.tag)
    print(f"Resumo salvo em {json_path}")
    print(f"Resumo legivel salvo em {md_path}")
    print(f"Estado salvo em {state_path}")
    fig = render_result(result)
    stem = "gonm_lj38" if not args.tag else f"gonm_lj38_{args.tag}"
    finish_figure(fig, __file__, stem)


if __name__ == "__main__":
    main()
