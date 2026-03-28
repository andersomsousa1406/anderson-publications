import json
import sys
from dataclasses import dataclass
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


SEED = 59
RESULT_DIR = ROOT / "publications" / "T07_GONM" / "results" / "gonm_cosmology_nbody"
FIGURE_PATH = RESULT_DIR / "gonm_cosmology_nbody.png"
SUMMARY_JSON_PATH = RESULT_DIR / "summary.json"
SUMMARY_MD_PATH = RESULT_DIR / "summary.md"


def pairwise_statistics(positions: np.ndarray) -> dict:
    dists = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dists.append(float(np.linalg.norm(positions[i] - positions[j])))
    arr = np.asarray(dists, dtype=float)
    return {
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
    }


@dataclass(slots=True)
class CosmologyClusteringProblem:
    body_count: int = 18
    box_radius: float = 14.0
    softening: float = 0.38
    barrier_scale: float = 0.85
    target_shell: float = 4.4

    @property
    def dimensions(self) -> int:
        return self.body_count * 2

    def project(self, x: np.ndarray) -> np.ndarray:
        pos = np.asarray(x, dtype=float).reshape(self.body_count, 2)
        radii = np.linalg.norm(pos, axis=1, keepdims=True)
        scale = np.minimum(1.0, self.box_radius / np.maximum(radii, 1e-9))
        pos = pos * scale
        pos -= np.mean(pos, axis=0, keepdims=True)
        return pos.reshape(-1)

    def random_geometry(self, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
        angles = rng.uniform(0.0, 2.0 * np.pi, size=self.body_count)
        radii = rng.uniform(3.2, self.box_radius * min(scale / 2.2, 1.0), size=self.body_count)
        pos = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        pos += rng.normal(scale=0.55, size=pos.shape)
        return self.project(pos.reshape(-1))

    def random_direction(self, rng: np.random.Generator) -> np.ndarray:
        direction = rng.normal(size=(self.body_count, 2))
        direction -= np.mean(direction, axis=0, keepdims=True)
        direction = direction.reshape(-1)
        return direction / max(np.linalg.norm(direction), 1e-9)

    def thermal_noise(self, rng: np.random.Generator, scale: float) -> np.ndarray:
        noise = rng.normal(size=(self.body_count, 2))
        noise -= np.mean(noise, axis=0, keepdims=True)
        radial = np.linspace(0.8, 1.2, self.body_count).reshape(-1, 1)
        return (noise * radial * scale).reshape(-1)

    def radial_signature(self, x: np.ndarray) -> np.ndarray:
        pos = np.asarray(self.project(x)).reshape(self.body_count, 2)
        return np.sort(np.linalg.norm(pos, axis=1))

    def vector_to_positions(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.project(x)).reshape(self.body_count, 2)

    def energy_terms(self, x: np.ndarray) -> dict:
        pos = self.vector_to_positions(x)
        gravitational = 0.0
        barrier = 0.0
        dists = []
        for i in range(self.body_count):
            for j in range(i + 1, self.body_count):
                diff = pos[i] - pos[j]
                r2 = float(np.dot(diff, diff))
                softened = np.sqrt(r2 + self.softening**2)
                gravitational += -1.0 / softened
                dists.append(np.sqrt(r2))
                barrier += self.barrier_scale / (r2 + 0.14)

        radii = np.linalg.norm(pos, axis=1)
        shell_penalty = 0.030 * float(np.mean((radii - self.target_shell) ** 2))
        centroid_penalty = 0.12 * float(np.sum(np.mean(pos, axis=0) ** 2))
        radial_spread = 0.018 * float(np.var(radii))
        action_like = gravitational + barrier + shell_penalty + centroid_penalty + radial_spread

        dists_arr = np.asarray(dists, dtype=float)
        return {
            "action_like": float(action_like),
            "gravitational": float(gravitational),
            "barrier": float(barrier),
            "shell_penalty": float(shell_penalty),
            "radial_spread": float(radial_spread),
            "min_pair_distance": float(np.min(dists_arr)),
            "mean_pair_distance": float(np.mean(dists_arr)),
        }

    def energy(self, x: np.ndarray) -> float:
        return self.energy_terms(x)["action_like"]

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = self.project(x)
        eps = 3e-4
        grad = np.zeros_like(x)
        for idx in range(len(x)):
            delta = np.zeros_like(x)
            delta[idx] = eps
            grad[idx] = (self.energy(self.project(x + delta)) - self.energy(self.project(x - delta))) / (2.0 * eps)
        return grad


def make_optimizer(problem: CosmologyClusteringProblem) -> GONMOptimizer:
    return GONMOptimizer(
        problem=problem,
        noise=CoolingNoise(start_std=0.020, mid_std=0.007, late_std=0.002, final_std=0.0),
        config=GONMConfig(
            population=28,
            phase1_keep=6,
            phase2_steps=54,
            phase3_steps=24,
            box_scale=2.2,
            thermal_scale=0.20,
            momentum=0.89,
            step_size=0.060,
            terminal_radii=(0.30, 0.18, 0.09, 0.045),
        ),
    )


def simulate_naive_nbody(
    positions0: np.ndarray,
    steps: int = 120,
    dt: float = 0.016,
    softening: float = 0.025,
) -> dict:
    pos = np.asarray(positions0, dtype=float).copy()
    vel = np.zeros_like(pos)
    min_distance_trace = []
    kinetic_trace = []
    unstable_step = None

    for step in range(steps):
        acc = np.zeros_like(pos)
        min_dist = 1e9
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                diff = pos[j] - pos[i]
                r2 = float(np.dot(diff, diff))
                min_dist = min(min_dist, np.sqrt(max(r2, 0.0)))
                denom = (r2 + softening**2) ** 1.5
                force = diff / max(denom, 1e-12)
                acc[i] += force
                acc[j] -= force
        vel += dt * acc
        pos += dt * vel
        min_distance_trace.append(float(min_dist))
        kinetic_trace.append(float(0.5 * np.mean(np.sum(vel * vel, axis=1))))
        if min_dist < 0.18 and unstable_step is None:
            unstable_step = step + 1

    return {
        "positions": pos,
        "min_distance_trace": min_distance_trace,
        "kinetic_trace": kinetic_trace,
        "unstable_step": unstable_step,
    }


def adam_baseline(
    problem: CosmologyClusteringProblem,
    start: np.ndarray,
    steps: int = 54,
    learning_rate: float = 0.045,
) -> tuple[np.ndarray, list[float]]:
    x = problem.project(start)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    beta1 = 0.9
    beta2 = 0.995
    eps = 1e-8
    losses = [problem.energy(x)]
    best_x = x.copy()
    best_loss = losses[0]
    for step in range(1, steps + 1):
        grad = problem.gradient(x)
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad**2)
        m_hat = m / (1.0 - beta1**step)
        v_hat = v / (1.0 - beta2**step)
        x = problem.project(x - learning_rate * m_hat / (np.sqrt(v_hat) + eps))
        loss = problem.energy(x)
        losses.append(loss)
        if loss < best_loss:
            best_loss = loss
            best_x = x.copy()
    return best_x, losses


def render_result(
    initial_positions: np.ndarray,
    naive: dict,
    problem: CosmologyClusteringProblem,
    baseline_best: np.ndarray,
    baseline_losses: list[float],
    gonm_result,
) -> plt.Figure:
    gonm_final = problem.vector_to_positions(gonm_result.final_best_positions.reshape(-1))
    baseline_pos = problem.vector_to_positions(baseline_best)
    initial = np.asarray(initial_positions, dtype=float)
    naive_final = np.asarray(naive["positions"], dtype=float)

    baseline_terms = problem.energy_terms(baseline_best)
    gonm_terms = problem.energy_terms(gonm_result.final_best_positions.reshape(-1))

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.7, 1.1])
    ax_cloud = fig.add_subplot(gs[0, 0])
    ax_trace = fig.add_subplot(gs[0, 1])
    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_text = fig.add_subplot(gs[1, 1])

    ax_cloud.scatter(initial[:, 0], initial[:, 1], s=24, color="#94a3b8", alpha=0.45, label="initial field")
    ax_cloud.scatter(naive_final[:, 0], naive_final[:, 1], s=22, color="#ef4444", alpha=0.55, label="naive N-body")
    ax_cloud.scatter(gonm_final[:, 0], gonm_final[:, 1], s=22, color="#2563eb", alpha=0.85, label="GONM clustered")
    ax_cloud.set_title("Agrupamento cosmologico reduzido")
    ax_cloud.set_xlabel("x")
    ax_cloud.set_ylabel("y")
    ax_cloud.set_aspect("equal", adjustable="box")
    ax_cloud.grid(True, linestyle=":", alpha=0.35)
    ax_cloud.legend(loc="best")

    ax_trace.plot(naive["min_distance_trace"], color="#dc2626", linewidth=2.0, label="naive min pair distance")
    ax_trace.plot(gonm_result.trace_true_energy, color="#2563eb", linewidth=2.0, label="GONM action-like")
    ax_trace.set_title("UV-collapse vs busca estruturada")
    ax_trace.set_xlabel("iteracao")
    ax_trace.grid(True, linestyle=":", alpha=0.5)
    ax_trace.legend(loc="best")

    baseline_stats = pairwise_statistics(baseline_pos)
    gonm_stats = pairwise_statistics(gonm_final)
    labels = ["baseline local", "GONM"]
    mins = [baseline_stats["min"], gonm_stats["min"]]
    means = [baseline_stats["mean"], gonm_stats["mean"]]
    x = np.arange(len(labels))
    width = 0.36
    ax_scatter.bar(x - width / 2, mins, width=width, color="#f87171", label="min pair distance")
    ax_scatter.bar(x + width / 2, means, width=width, color="#60a5fa", label="mean pair distance")
    ax_scatter.set_xticks(x)
    ax_scatter.set_xticklabels(labels)
    ax_scatter.set_title("Escalas de separacao")
    ax_scatter.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax_scatter.legend(loc="best")

    gain = min(baseline_losses) - gonm_terms["action_like"]
    unstable_line = "naive unstable step = none" if naive["unstable_step"] is None else f"naive unstable step = {naive['unstable_step']}"
    ax_text.axis("off")
    ax_text.text(
        0.0,
        1.0,
        "\n".join(
            [
                "GONM | cosmological N-body clustering",
                "",
                f"bodies = {problem.body_count}",
                f"naive final min pair = {min(naive['min_distance_trace']):.6f}",
                unstable_line,
                "",
                f"baseline local best = {min(baseline_losses):.6f}",
                f"GONM final = {gonm_terms['action_like']:.6f}",
                f"ganho GONM vs baseline = {gain:.6f}",
                "",
                f"GONM gravitational term = {gonm_terms['gravitational']:.6f}",
                f"GONM barrier term = {gonm_terms['barrier']:.6f}",
                f"GONM min pair distance = {gonm_terms['min_pair_distance']:.6f}",
                "",
                f"baseline barrier term = {baseline_terms['barrier']:.6f}",
                f"baseline min pair distance = {baseline_terms['min_pair_distance']:.6f}",
                "",
                "Leitura:",
                "a evolucao ingênua quase entra em colisao dura,",
                "enquanto o GONM produz agrupamento compacto",
                "sem catastrofe ultravioleta numerica.",
            ]
        ),
        ha="left",
        va="top",
        fontsize=11,
        family="monospace",
    )

    fig.suptitle("GONM | Large-Scale N-Body Stability", fontsize=16, y=0.98)
    fig.tight_layout()
    return fig


def write_summary(problem: CosmologyClusteringProblem, naive: dict, baseline_best: np.ndarray, baseline_losses: list[float], gonm_result) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    baseline_terms = problem.energy_terms(baseline_best)
    gonm_terms = problem.energy_terms(gonm_result.final_best_positions.reshape(-1))
    summary = {
        "seed": SEED,
        "problem": "cosmology_nbody_stability",
        "interpretation": "Reduced large-scale gravitational clustering with a naive near-singular evolution compared against GONM structural regularization.",
        "body_count": problem.body_count,
        "naive_nbody": {
            "min_pair_distance": float(min(naive["min_distance_trace"])),
            "unstable_step": naive["unstable_step"],
            "final_kinetic_mean": float(naive["kinetic_trace"][-1]),
        },
        "baseline_local": {
            "best_action_like": float(min(baseline_losses)),
            "terms": baseline_terms,
        },
        "gonm": {
            "phase1_best_action_like": float(gonm_result.phase1_best_energy),
            "final_action_like": float(gonm_terms["action_like"]),
            "terms": gonm_terms,
            "evals": int(gonm_result.evals),
            "runtime_ms": float(gonm_result.runtime_ms),
        },
        "comparison": {
            "gonm_minus_baseline": float(gonm_terms["action_like"] - min(baseline_losses)),
            "baseline_start_action_like": float(baseline_losses[0]),
        },
    }
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = f"""# GONM | Cosmological N-Body Stability

This simulation compares a naive gravitational N-body evolution against a layered GONM clustering search.

The gravitational setting is intentionally reduced, but it preserves the main difficulty:

- long-range attraction;
- numerical near-collisions at short distance;
- the need to keep clustering compact without collapsing into an ultraviolet singularity.

## Recorded outcome

- naive minimum pair distance: `{min(naive["min_distance_trace"]):.6f}`
- local baseline best functional: `{min(baseline_losses):.6f}`
- GONM final functional: `{gonm_terms["action_like"]:.6f}`
- GONM gain versus baseline: `{min(baseline_losses) - gonm_terms["action_like"]:.6f}`

## Interpretation

This is not a million-body production cosmology code. The narrower demonstration is still meaningful: the GONM-style structural search can form a compact clustered configuration while maintaining a finite minimum separation, whereas a naive near-singular evolution rapidly approaches the ultraviolet catastrophe.
"""
    SUMMARY_MD_PATH.write_text(md, encoding="utf-8")


def main() -> None:
    rng = np.random.default_rng(SEED)
    problem = CosmologyClusteringProblem()
    initial_vector = problem.random_geometry(rng, scale=2.0)
    initial_positions = problem.vector_to_positions(initial_vector)
    naive = simulate_naive_nbody(initial_positions)

    optimizer = make_optimizer(problem)
    gonm_result = optimizer.optimize(seed=SEED)

    baseline_start = initial_vector
    baseline_best, baseline_losses = adam_baseline(problem, baseline_start)

    fig = render_result(initial_positions, naive, problem, baseline_best, baseline_losses, gonm_result)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    if "--no-show" not in sys.argv and "--save" not in sys.argv:
        plt.show()
    plt.close(fig)

    write_summary(problem, naive, baseline_best, baseline_losses, gonm_result)


if __name__ == "__main__":
    main()
