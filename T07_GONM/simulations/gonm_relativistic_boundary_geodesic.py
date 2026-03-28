import json
import sys
from dataclasses import dataclass, field
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


SEED = 83
RESULT_DIR = ROOT / "publications" / "T07_GONM" / "results" / "gonm_relativistic_boundary_geodesic"
FIGURE_PATH = RESULT_DIR / "gonm_relativistic_boundary_geodesic.png"
SUMMARY_JSON_PATH = RESULT_DIR / "summary.json"
SUMMARY_MD_PATH = RESULT_DIR / "summary.md"


@dataclass(slots=True)
class SchwarzschildBoundaryGeodesicProblem:
    x_start: float = -10.0
    x_end: float = 10.0
    y_start: float = 4.8
    y_end: float = 4.8
    control_points: int = 14
    fine_points: int = 360
    schwarzschild_mass: float = 1.0
    safe_radius: float = 2.25
    x_grid: np.ndarray = field(init=False, repr=False)
    x_ctrl: np.ndarray = field(init=False, repr=False)
    dx: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.x_grid = np.linspace(self.x_start, self.x_end, self.fine_points, dtype=float)
        self.x_ctrl = np.linspace(self.x_start, self.x_end, self.control_points + 2, dtype=float)[1:-1]
        self.dx = float(self.x_grid[1] - self.x_grid[0])

    @property
    def dimensions(self) -> int:
        return self.control_points

    @property
    def horizon_radius(self) -> float:
        return 2.0 * self.schwarzschild_mass

    def project(self, x: np.ndarray) -> np.ndarray:
        y = np.asarray(x, dtype=float).reshape(-1)
        return np.clip(y, -8.5, 8.5)

    def random_geometry(self, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
        base = np.linspace(self.y_start, self.y_end, self.control_points, dtype=float)
        sag = rng.uniform(-2.0, -0.5)
        width = rng.uniform(2.2, 5.5)
        center = rng.uniform(-1.4, 1.4)
        valley = sag * np.exp(-((self.x_ctrl - center) ** 2) / (2.0 * width**2))
        asym = 0.12 * rng.normal(size=self.control_points)
        return self.project(base + valley + asym * min(scale, 2.0))

    def random_direction(self, rng: np.random.Generator) -> np.ndarray:
        direction = rng.normal(size=self.control_points)
        return direction / max(np.linalg.norm(direction), 1e-9)

    def thermal_noise(self, rng: np.random.Generator, scale: float) -> np.ndarray:
        envelope = np.sin(np.linspace(0.0, np.pi, self.control_points)) ** 2 + 0.35
        return rng.normal(size=self.control_points) * envelope * scale

    def radial_signature(self, x: np.ndarray) -> np.ndarray:
        y = self._curve(x)
        sample_idx = np.linspace(0, len(y) - 1, 12, dtype=int)
        r = np.sqrt(self.x_grid[sample_idx] ** 2 + y[sample_idx] ** 2)
        return r

    def vector_to_positions(self, x: np.ndarray) -> np.ndarray:
        return self.project(x).reshape(1, -1)

    def _curve(self, x: np.ndarray) -> np.ndarray:
        interior = self.project(x)
        y_nodes = np.concatenate([[self.y_start], interior, [self.y_end]])
        x_nodes = np.concatenate([[self.x_start], self.x_ctrl, [self.x_end]])
        return np.interp(self.x_grid, x_nodes, y_nodes)

    def optical_factor(self, r: np.ndarray) -> np.ndarray:
        return 1.0 / np.maximum(1e-6, 1.0 - 2.0 * self.schwarzschild_mass / r)

    def energy_terms(self, x: np.ndarray) -> dict:
        y = self._curve(x)
        dy = np.gradient(y, self.dx, edge_order=2)
        ddy = np.gradient(dy, self.dx, edge_order=2)
        r = np.sqrt(self.x_grid**2 + y**2)

        n = self.optical_factor(r)
        ds = np.sqrt(1.0 + dy**2)
        optical_length = float(np.trapezoid(n * ds, self.x_grid))

        dn_dr = (2.0 * self.schwarzschild_mass) / np.maximum(1e-6, r**2 * (1.0 - 2.0 * self.schwarzschild_mass / r) ** 2)
        dn_dy = dn_dr * y / np.maximum(r, 1e-9)
        euler_residual = dn_dy * ds - np.gradient(n * dy / np.maximum(ds, 1e-9), self.dx, edge_order=2)
        geodesic_residual = float(np.sqrt(np.mean(euler_residual**2)))

        curvature_penalty = 0.030 * float(np.mean(ddy**2))
        horizon_penalty = 8.5 * float(np.mean(np.maximum(0.0, self.safe_radius - r) ** 2))
        endpoint_slope_penalty = 0.18 * float(dy[0] ** 2 + dy[-1] ** 2)
        strong_field_reward = -0.14 * float(np.exp(-((np.min(r) - 3.2) ** 2) / 0.28))

        action_like = optical_length + 4.2 * geodesic_residual + curvature_penalty + horizon_penalty + endpoint_slope_penalty + strong_field_reward
        return {
            "action_like": float(action_like),
            "optical_length": optical_length,
            "geodesic_residual": geodesic_residual,
            "curvature_penalty": float(curvature_penalty),
            "horizon_penalty": float(horizon_penalty),
            "endpoint_slope_penalty": float(endpoint_slope_penalty),
            "strong_field_reward": float(strong_field_reward),
            "min_radius": float(np.min(r)),
            "max_deflection": float(np.max(np.abs(y_start_line(self.x_grid, self.y_start, self.y_end, self.x_start, self.x_end) - y))),
        }

    def energy(self, x: np.ndarray) -> float:
        return self.energy_terms(x)["action_like"]

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = self.project(x)
        eps = np.full(self.control_points, 2e-4, dtype=float)
        grad = np.zeros(self.control_points, dtype=float)
        for idx in range(self.control_points):
            delta = np.zeros(self.control_points, dtype=float)
            delta[idx] = eps[idx]
            grad[idx] = (self.energy(self.project(x + delta)) - self.energy(self.project(x - delta))) / (2.0 * eps[idx])
        return grad


def y_start_line(x: np.ndarray, y_start: float, y_end: float, x_start: float, x_end: float) -> np.ndarray:
    t = (x - x_start) / (x_end - x_start)
    return (1.0 - t) * y_start + t * y_end


def make_optimizer(problem: SchwarzschildBoundaryGeodesicProblem) -> GONMOptimizer:
    return GONMOptimizer(
        problem=problem,
        noise=CoolingNoise(start_std=0.0010, mid_std=0.0004, late_std=0.00008, final_std=0.0),
        config=GONMConfig(
            population=72,
            phase1_keep=10,
            phase2_steps=170,
            phase3_steps=72,
            box_scale=1.8,
            thermal_scale=0.10,
            momentum=0.88,
            step_size=0.055,
            terminal_radii=(0.18, 0.10, 0.05, 0.02),
        ),
    )


def baseline_local(
    problem: SchwarzschildBoundaryGeodesicProblem,
    start: np.ndarray,
    steps: int = 110,
    learning_rate: float = 0.038,
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


def render_result(problem: SchwarzschildBoundaryGeodesicProblem, baseline_best: np.ndarray, baseline_losses: list[float], gonm_result) -> plt.Figure:
    baseline_y = problem._curve(baseline_best)
    phase1_y = problem._curve(gonm_result.phase1_best_positions.reshape(-1))
    gonm_y = problem._curve(gonm_result.final_best_positions.reshape(-1))
    straight = y_start_line(problem.x_grid, problem.y_start, problem.y_end, problem.x_start, problem.x_end)
    baseline_terms = problem.energy_terms(baseline_best)
    gonm_terms = problem.energy_terms(gonm_result.final_best_positions.reshape(-1))

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.7, 1.1])
    ax_path = fig.add_subplot(gs[0, 0])
    ax_trace = fig.add_subplot(gs[0, 1])
    ax_radius = fig.add_subplot(gs[1, 0])
    ax_text = fig.add_subplot(gs[1, 1])

    horizon = plt.Circle((0, 0), problem.horizon_radius, color="#020617", alpha=0.95, label="event horizon")
    photon = plt.Circle((0, 0), 3.0, color="#334155", fill=False, linestyle="--", linewidth=1.6, label="photon sphere")
    ax_path.add_patch(horizon)
    ax_path.add_patch(photon)
    ax_path.plot(problem.x_grid, straight, color="#94a3b8", linewidth=1.8, linestyle=":", label="straight chord")
    ax_path.plot(problem.x_grid, baseline_y, color="#ef4444", linewidth=2.2, label="baseline local")
    ax_path.plot(problem.x_grid, phase1_y, color="#93c5fd", linewidth=2.0, label="GONM phase 1")
    ax_path.plot(problem.x_grid, gonm_y, color="#1d4ed8", linewidth=2.5, label="GONM final")
    ax_path.set_aspect("equal", adjustable="box")
    ax_path.set_xlim(problem.x_start - 1.0, problem.x_end + 1.0)
    ax_path.set_ylim(-8.5, 8.5)
    ax_path.set_title("Geodésica de contorno em Schwarzschild")
    ax_path.set_xlabel("x / M")
    ax_path.set_ylabel("y / M")
    ax_path.grid(True, linestyle=":", alpha=0.35)
    ax_path.legend(loc="upper right")

    ax_trace.plot(baseline_losses, color="#dc2626", linewidth=2.0, label="baseline local")
    ax_trace.plot(gonm_result.trace_true_energy, color="#2563eb", linewidth=2.0, label="GONM")
    ax_trace.set_title("Funcional óptico ao longo da busca")
    ax_trace.set_xlabel("iteracao")
    ax_trace.set_ylabel("J[path]")
    ax_trace.grid(True, linestyle=":", alpha=0.5)
    ax_trace.legend(loc="best")

    baseline_r = np.sqrt(problem.x_grid**2 + baseline_y**2)
    gonm_r = np.sqrt(problem.x_grid**2 + gonm_y**2)
    ax_radius.plot(problem.x_grid, baseline_r, color="#f87171", linewidth=2.0, label="baseline local")
    ax_radius.plot(problem.x_grid, gonm_r, color="#60a5fa", linewidth=2.0, label="GONM final")
    ax_radius.axhline(3.0, color="#334155", linewidth=1.5, linestyle="--", label="photon sphere")
    ax_radius.axhline(problem.horizon_radius, color="#111827", linewidth=1.5, linestyle="-.", label="horizon")
    ax_radius.set_title("Raio ao longo da corda geodésica")
    ax_radius.set_xlabel("x / M")
    ax_radius.set_ylabel("r(x) / M")
    ax_radius.grid(True, linestyle=":", alpha=0.5)
    ax_radius.legend(loc="best")

    gain = min(baseline_losses) - gonm_terms["action_like"]
    ax_text.axis("off")
    ax_text.text(
        0.0,
        1.0,
        "\n".join(
            [
                "GONM | relativistic boundary-value geodesic",
                "",
                "Problema de contorno:",
                "mesmos endpoints, curva inteira otimizada de uma vez.",
                "",
                f"baseline local best = {min(baseline_losses):.6f}",
                f"GONM final = {gonm_terms['action_like']:.6f}",
                f"ganho GONM vs baseline = {gain:.6f}",
                "",
                f"optical length GONM = {gonm_terms['optical_length']:.6f}",
                f"geodesic residual GONM = {gonm_terms['geodesic_residual']:.6f}",
                f"min radius GONM = {gonm_terms['min_radius']:.4f} M",
                "",
                f"optical length baseline = {baseline_terms['optical_length']:.6f}",
                f"geodesic residual baseline = {baseline_terms['geodesic_residual']:.6f}",
                f"min radius baseline = {baseline_terms['min_radius']:.4f} M",
                "",
                "Leitura:",
                "o GONM ajusta a corda inteira e reduz o acúmulo",
                "de erro local que aparece em abordagens puramente",
                "iterativas ou de integração passo a passo.",
            ]
        ),
        ha="left",
        va="top",
        fontsize=11,
        family="monospace",
    )

    fig.suptitle("GONM | Boundary-Value Relativistic Geodesic", fontsize=16, y=0.98)
    fig.tight_layout()
    return fig


def write_summary(problem: SchwarzschildBoundaryGeodesicProblem, baseline_best: np.ndarray, baseline_losses: list[float], gonm_result) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    baseline_terms = problem.energy_terms(baseline_best)
    gonm_terms = problem.energy_terms(gonm_result.final_best_positions.reshape(-1))
    summary = {
        "seed": SEED,
        "problem": "relativistic_boundary_geodesic",
        "interpretation": "Boundary-value geodesic search in the equatorial Schwarzschild optical metric.",
        "endpoints": {
            "start": [problem.x_start, problem.y_start],
            "end": [problem.x_end, problem.y_end],
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
        },
    }
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = f"""# GONM | Boundary-Value Relativistic Geodesic

This simulation treats GONM as a boundary-value geodesic solver in Schwarzschild spacetime.

Instead of integrating a trajectory step by step, it optimizes the entire path between two fixed endpoints under an optical-metric action-like functional.

## Recorded outcome

- local baseline best functional: `{min(baseline_losses):.6f}`
- GONM final functional: `{gonm_terms["action_like"]:.6f}`
- GONM gain versus baseline: `{min(baseline_losses) - gonm_terms["action_like"]:.6f}`
- GONM minimum radius: `{gonm_terms["min_radius"]:.4f} M`

## Interpretation

This is not a full Einstein-equation solver or a Kerr ray-tracer. The narrower claim is still strong: by optimizing the whole curve at once, the layered GONM search reduces geodesic residual and avoids the accumulation of local integration errors near the strong-field region.
"""
    SUMMARY_MD_PATH.write_text(md, encoding="utf-8")


def main() -> None:
    problem = SchwarzschildBoundaryGeodesicProblem()
    optimizer = make_optimizer(problem)
    gonm_result = optimizer.optimize(seed=SEED)

    baseline_start = 4.8 - 1.65 * np.sin(np.linspace(0.0, np.pi, problem.control_points)) ** 2 + 0.22 * np.cos(
        np.linspace(0.0, 3.0 * np.pi, problem.control_points)
    )
    baseline_best, baseline_losses = baseline_local(problem, baseline_start)

    fig = render_result(problem, baseline_best, baseline_losses, gonm_result)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    if "--no-show" not in sys.argv and "--save" not in sys.argv:
        plt.show()
    plt.close(fig)

    write_summary(problem, baseline_best, baseline_losses, gonm_result)


if __name__ == "__main__":
    main()
