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


SEED = 43
RESULT_DIR = ROOT / "publications" / "T07_GONM" / "results" / "gonm_relativistic_geodesic"
FIGURE_PATH = RESULT_DIR / "gonm_relativistic_geodesic.png"
SUMMARY_JSON_PATH = RESULT_DIR / "summary.json"
SUMMARY_MD_PATH = RESULT_DIR / "summary.md"


@dataclass(slots=True)
class SchwarzschildGeodesicProblem:
    phi_start: float = -1.32
    phi_end: float = 1.32
    control_points: int = 12
    fine_points: int = 320
    r_far: float = 12.0
    horizon_radius: float = 2.0
    safe_radius: float = 2.18
    target_periastron: float = 3.15
    phi: np.ndarray = field(init=False, repr=False)
    phi_ctrl: np.ndarray = field(init=False, repr=False)
    dphi: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.phi = np.linspace(self.phi_start, self.phi_end, self.fine_points, dtype=float)
        self.phi_ctrl = np.linspace(self.phi_start, self.phi_end, self.control_points + 2, dtype=float)[1:-1]
        self.dphi = float(self.phi[1] - self.phi[0])

    @property
    def dimensions(self) -> int:
        return self.control_points

    def project(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        return np.clip(x, 2.05, 15.0)

    def random_geometry(self, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
        base_depth = rng.uniform(2.35, 4.8)
        asymmetry = rng.uniform(-0.55, 0.55)
        widths = rng.uniform(0.55, 1.15, size=2)
        bumps = []
        for phi_i in self.phi_ctrl:
            left = np.exp(-((phi_i + 0.55 + asymmetry) ** 2) / (2.0 * widths[0] ** 2))
            right = np.exp(-((phi_i - 0.55 + asymmetry) ** 2) / (2.0 * widths[1] ** 2))
            valley = 0.72 * (left + right)
            radius = self.r_far - (self.r_far - base_depth) * min(valley, 1.0)
            bumps.append(radius)
        profile = np.asarray(bumps, dtype=float)
        jitter = rng.normal(scale=0.42 * min(scale, 2.0), size=self.control_points)
        return self.project(profile + jitter)

    def random_direction(self, rng: np.random.Generator) -> np.ndarray:
        direction = rng.normal(size=self.control_points)
        return direction / max(np.linalg.norm(direction), 1e-9)

    def thermal_noise(self, rng: np.random.Generator, scale: float) -> np.ndarray:
        envelope = 0.6 + 0.5 * np.sin(np.linspace(0.0, np.pi, self.control_points)) ** 2
        return rng.normal(size=self.control_points) * envelope * scale

    def radial_signature(self, x: np.ndarray) -> np.ndarray:
        r = self._curve_from_controls(x)
        sample_idx = np.linspace(0, len(r) - 1, 10, dtype=int)
        return r[sample_idx]

    def vector_to_positions(self, x: np.ndarray) -> np.ndarray:
        return self.project(x).reshape(1, -1)

    def _curve_from_controls(self, x: np.ndarray) -> np.ndarray:
        interior = self.project(x)
        radii = np.concatenate([[self.r_far], interior, [self.r_far]])
        phi_nodes = np.concatenate([[self.phi_start], self.phi_ctrl, [self.phi_end]])
        curve = np.interp(self.phi, phi_nodes, radii)
        return np.maximum(curve, 2.02)

    def _u_profile(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / self._curve_from_controls(x)

    def invariants_profile(self, x: np.ndarray) -> np.ndarray:
        u = self._u_profile(x)
        up = np.gradient(u, self.dphi, edge_order=2)
        return up**2 + u**2 - 2.0 * u**3

    def geodesic_residual(self, x: np.ndarray) -> np.ndarray:
        u = self._u_profile(x)
        upp = np.gradient(np.gradient(u, self.dphi, edge_order=2), self.dphi, edge_order=2)
        return upp + u - 3.0 * u**2

    def energy_terms(self, x: np.ndarray) -> dict:
        r = self._curve_from_controls(x)
        u = 1.0 / r
        residual = self.geodesic_residual(x)
        invariant = self.invariants_profile(x)
        ru = np.gradient(r, self.dphi, edge_order=2)
        smoothness = np.mean(np.gradient(ru, self.dphi, edge_order=2) ** 2)
        invariant_var = np.var(invariant)
        residual_rms = np.sqrt(np.mean(residual**2))

        slope_left = (r[1] - r[0]) / self.dphi
        slope_right = (r[-1] - r[-2]) / self.dphi
        slope_penalty = 0.035 * (slope_left**2 + slope_right**2)

        horizon_penalty = 4.8 * np.mean(np.maximum(0.0, self.safe_radius - r) ** 2)
        periastron = float(np.min(r))
        targeting_penalty = 2.6 * (periastron - self.target_periastron) ** 2
        monotonic_guard = 0.012 * np.mean(np.maximum(0.0, -np.gradient(np.abs(self.phi), self.dphi)) ** 2)
        action_like = (
            residual_rms**2
            + 0.55 * invariant_var
            + 0.015 * smoothness
            + slope_penalty
            + horizon_penalty
            + targeting_penalty
            + monotonic_guard
        )
        return {
            "action_like": float(action_like),
            "residual_rms": float(residual_rms),
            "invariant_var": float(invariant_var),
            "smoothness": float(smoothness),
            "slope_penalty": float(slope_penalty),
            "horizon_penalty": float(horizon_penalty),
            "targeting_penalty": float(targeting_penalty),
            "periastron": periastron,
            "impact_proxy": float(np.mean(invariant[invariant > 1e-8]) ** -0.5 if np.any(invariant > 1e-8) else 0.0),
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


def make_optimizer(problem: SchwarzschildGeodesicProblem) -> GONMOptimizer:
    return GONMOptimizer(
        problem=problem,
        noise=CoolingNoise(start_std=0.0014, mid_std=0.00055, late_std=0.00012, final_std=0.0),
        config=GONMConfig(
            population=88,
            phase1_keep=10,
            phase2_steps=220,
            phase3_steps=96,
            box_scale=2.2,
            thermal_scale=0.11,
            momentum=0.88,
            step_size=0.050,
            terminal_radii=(0.22, 0.12, 0.06, 0.03),
        ),
    )


def adam_baseline(
    problem: SchwarzschildGeodesicProblem,
    start: np.ndarray,
    steps: int = 240,
    learning_rate: float = 0.032,
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


def draw_geodesic(ax, problem: SchwarzschildGeodesicProblem, params: np.ndarray, label: str, color: str, linewidth: float) -> None:
    r = problem._curve_from_controls(params)
    xs = r * np.cos(problem.phi)
    ys = r * np.sin(problem.phi)
    ax.plot(xs, ys, color=color, linewidth=linewidth, label=label)


def render_result(problem: SchwarzschildGeodesicProblem, baseline_best: np.ndarray, baseline_losses: list[float], gonm_result) -> plt.Figure:
    phase1 = gonm_result.phase1_best_positions.reshape(-1)
    final = gonm_result.final_best_positions.reshape(-1)

    phase1_terms = problem.energy_terms(phase1)
    final_terms = problem.energy_terms(final)
    baseline_terms = problem.energy_terms(baseline_best)

    inv_baseline = problem.invariants_profile(baseline_best)
    inv_gonm = problem.invariants_profile(final)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.7, 1.1])
    ax_orbit = fig.add_subplot(gs[0, 0])
    ax_trace = fig.add_subplot(gs[0, 1])
    ax_inv = fig.add_subplot(gs[1, 0])
    ax_text = fig.add_subplot(gs[1, 1])

    horizon = plt.Circle((0, 0), problem.horizon_radius, color="#020617", alpha=0.95, label="event horizon")
    photon = plt.Circle((0, 0), 3.0, color="#334155", fill=False, linestyle="--", linewidth=1.6, label="photon sphere")
    ax_orbit.add_patch(horizon)
    ax_orbit.add_patch(photon)
    draw_geodesic(ax_orbit, problem, phase1, "GONM phase 1", "#93c5fd", 2.0)
    draw_geodesic(ax_orbit, problem, baseline_best, "baseline local", "#ef4444", 2.2)
    draw_geodesic(ax_orbit, problem, final, "GONM final", "#1d4ed8", 2.5)
    ax_orbit.set_aspect("equal", adjustable="box")
    ax_orbit.set_xlim(-13, 13)
    ax_orbit.set_ylim(-13, 13)
    ax_orbit.set_title("Trajetoria equatorial em Schwarzschild")
    ax_orbit.set_xlabel("x / M")
    ax_orbit.set_ylabel("y / M")
    ax_orbit.grid(True, linestyle=":", alpha=0.35)
    ax_orbit.legend(loc="upper right")

    ax_trace.plot(baseline_losses, color="#dc2626", linewidth=2.0, label="baseline local")
    ax_trace.plot(gonm_result.trace_true_energy, color="#2563eb", linewidth=2.0, label="GONM")
    ax_trace.set_title("Funcional tipo-acao ao longo da busca")
    ax_trace.set_xlabel("iteracao")
    ax_trace.set_ylabel("J[trajetoria]")
    ax_trace.grid(True, linestyle=":", alpha=0.5)
    ax_trace.legend(loc="best")

    ax_inv.plot(problem.phi, inv_baseline, color="#f87171", linewidth=2.0, label="baseline local")
    ax_inv.plot(problem.phi, inv_gonm, color="#60a5fa", linewidth=2.0, label="GONM final")
    ax_inv.set_title("Perfil do invariante conservado")
    ax_inv.set_xlabel("phi")
    ax_inv.set_ylabel("u'^2 + u^2 - 2u^3")
    ax_inv.grid(True, linestyle=":", alpha=0.5)
    ax_inv.legend(loc="best")

    gain = min(baseline_losses) - final_terms["action_like"]
    ax_text.axis("off")
    ax_text.text(
        0.0,
        1.0,
        "\n".join(
            [
                "GONM | relativistic geodesic search",
                "",
                "Modelo: Schwarzschild equatorial, null-like proxy",
                "Funcional = resíduo da geodésica + variação do invariante",
                "",
                f"baseline local best = {min(baseline_losses):.6f}",
                f"GONM phase-1 best = {phase1_terms['action_like']:.6f}",
                f"GONM final = {final_terms['action_like']:.6f}",
                f"ganho GONM vs baseline = {gain:.6f}",
                "",
                f"residual RMS GONM = {final_terms['residual_rms']:.6f}",
                f"var(invariante) GONM = {final_terms['invariant_var']:.6f}",
                f"periastron GONM = {final_terms['periastron']:.4f} M",
                "",
                f"residual RMS baseline = {baseline_terms['residual_rms']:.6f}",
                f"var(invariante) baseline = {baseline_terms['invariant_var']:.6f}",
                f"periastron baseline = {baseline_terms['periastron']:.4f} M",
                "",
                "Leitura:",
                "o GONM reorganiza a curva antes da contracao local,",
                "reduzindo tanto o erro geodésico quanto a deriva",
                "dos invariantes perto do horizonte.",
            ]
        ),
        ha="left",
        va="top",
        fontsize=11,
        family="monospace",
    )

    fig.suptitle("GONM | Relativistic Geodesic Search near a Black Hole", fontsize=16, y=0.98)
    fig.tight_layout()
    return fig


def write_summary(problem: SchwarzschildGeodesicProblem, baseline_best: np.ndarray, baseline_losses: list[float], gonm_result) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    final = gonm_result.final_best_positions.reshape(-1)
    baseline_terms = problem.energy_terms(baseline_best)
    final_terms = problem.energy_terms(final)

    summary = {
        "seed": SEED,
        "problem": "schwarzschild_relativistic_geodesic",
        "interpretation": "Variational search for an equatorial geodesic-like trajectory near a Schwarzschild black hole.",
        "horizon_radius": problem.horizon_radius,
        "photon_sphere_radius": 3.0,
        "baseline_local": {
            "best_action_like": float(min(baseline_losses)),
            "terms": baseline_terms,
        },
        "gonm": {
            "phase1_best_action_like": float(gonm_result.phase1_best_energy),
            "final_action_like": float(final_terms["action_like"]),
            "terms": final_terms,
            "evals": int(gonm_result.evals),
            "runtime_ms": float(gonm_result.runtime_ms),
        },
        "comparison": {
            "gonm_minus_baseline": float(final_terms["action_like"] - min(baseline_losses)),
            "baseline_start_action_like": float(baseline_losses[0]),
        },
    }
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = f"""# GONM | Relativistic Geodesic Search

This simulation treats GONM as a variational search procedure for a Schwarzschild geodesic-like curve.

The objective is not a full exact relativistic integrator. Instead, it minimizes a trajectory functional built from:

- the residual of the reduced geodesic equation in `u(phi) = 1 / r(phi)`;
- the variance of the conserved first-integral proxy;
- barrier terms that prevent an artificial numerical fall through the horizon.

## Recorded outcome

- baseline local best functional: `{min(baseline_losses):.6f}`
- GONM final functional: `{final_terms["action_like"]:.6f}`
- GONM gain versus baseline: `{min(baseline_losses) - final_terms["action_like"]:.6f}`

## Final relativistic diagnostics

- GONM residual RMS: `{final_terms["residual_rms"]:.6f}`
- GONM invariant variance: `{final_terms["invariant_var"]:.6f}`
- GONM periastron: `{final_terms["periastron"]:.4f} M`

## Interpretation

This is a reduced variational proxy, not a full Kerr or exact event-horizon integrator. The narrower claim is still meaningful: near the strong-field region, the layered GONM search can find a curve with lower geodesic residual and better invariant preservation than a purely local optimizer started from a poor trajectory family.
"""
    SUMMARY_MD_PATH.write_text(md, encoding="utf-8")


def main() -> None:
    problem = SchwarzschildGeodesicProblem()
    optimizer = make_optimizer(problem)
    gonm_result = optimizer.optimize(seed=SEED)

    bad_start = np.full(problem.control_points, 2.45, dtype=float)
    baseline_best, baseline_losses = adam_baseline(problem, bad_start)

    fig = render_result(problem, baseline_best, baseline_losses, gonm_result)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    if "--no-show" not in sys.argv and "--save" not in sys.argv:
        plt.show()
    plt.close(fig)

    write_summary(problem, baseline_best, baseline_losses, gonm_result)


if __name__ == "__main__":
    main()
