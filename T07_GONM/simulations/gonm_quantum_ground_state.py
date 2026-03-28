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


SEED = 37
RESULT_DIR = ROOT / "publications" / "T07_GONM" / "results" / "gonm_quantum_ground_state"
FIGURE_PATH = RESULT_DIR / "gonm_quantum_ground_state.png"
SUMMARY_JSON_PATH = RESULT_DIR / "summary.json"
SUMMARY_MD_PATH = RESULT_DIR / "summary.md"


@dataclass(slots=True)
class VariationalQuantumProblem:
    grid_min: float = -4.8
    grid_max: float = 4.8
    grid_size: int = 1200
    x: np.ndarray = field(init=False, repr=False)
    dx: float = field(init=False, repr=False)
    potential: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.x = np.linspace(self.grid_min, self.grid_max, self.grid_size, dtype=float)
        self.dx = float(self.x[1] - self.x[0])
        # Double well plus anharmonic/titled correction.
        self.potential = 0.18 * self.x**6 - 1.55 * self.x**4 + 2.7 * self.x**2 + 0.10 * self.x

    @property
    def dimensions(self) -> int:
        return 5

    def project(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float).reshape(-1)
        out = theta.copy()
        out[0] = np.clip(out[0], -1.2, 1.25)   # log sigma left
        out[1] = np.clip(out[1], -1.2, 1.25)   # log sigma right
        out[2] = np.clip(out[2], 0.1, 2.4)     # half-distance between lobes
        out[3] = np.clip(out[3], -2.8, 2.8)    # amplitude log-ratio
        out[4] = np.clip(out[4], -0.9, 0.9)    # global center shift
        return out

    def random_geometry(self, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
        base = np.array(
            [
                rng.uniform(-0.9, 0.8),
                rng.uniform(-0.9, 0.8),
                rng.uniform(0.2, 2.2),
                rng.uniform(-2.0, 2.0),
                rng.uniform(-0.7, 0.7),
            ],
            dtype=float,
        )
        jitter = rng.normal(scale=0.18 * min(scale, 2.0), size=5)
        return self.project(base + jitter)

    def random_direction(self, rng: np.random.Generator) -> np.ndarray:
        direction = rng.normal(size=5)
        return direction / max(np.linalg.norm(direction), 1e-9)

    def thermal_noise(self, rng: np.random.Generator, scale: float) -> np.ndarray:
        weights = np.array([0.30, 0.30, 0.28, 0.40, 0.18], dtype=float)
        return rng.normal(size=5) * weights * scale

    def radial_signature(self, theta: np.ndarray) -> np.ndarray:
        return self.project(theta)

    def vector_to_positions(self, theta: np.ndarray) -> np.ndarray:
        return self.project(theta).reshape(1, -1)

    def decode(self, theta: np.ndarray) -> dict:
        theta = self.project(theta)
        sigma_left = float(np.exp(theta[0]))
        sigma_right = float(np.exp(theta[1]))
        distance = float(theta[2])
        amp_ratio = float(np.exp(theta[3]))
        center_shift = float(theta[4])
        return {
            "sigma_left": sigma_left,
            "sigma_right": sigma_right,
            "half_distance": distance,
            "amplitude_ratio": amp_ratio,
            "center_shift": center_shift,
        }

    def psi(self, theta: np.ndarray) -> np.ndarray:
        theta = self.project(theta)
        sigma_left = np.exp(theta[0])
        sigma_right = np.exp(theta[1])
        distance = theta[2]
        amp = np.exp(theta[3])
        shift = theta[4]

        left = np.exp(-0.5 * ((self.x - (shift - distance)) / sigma_left) ** 2)
        right = amp * np.exp(-0.5 * ((self.x - (shift + distance)) / sigma_right) ** 2)
        psi = left + right
        norm = np.sqrt(np.trapezoid(psi**2, self.x))
        return psi / max(norm, 1e-12)

    def energy_components(self, theta: np.ndarray) -> tuple[float, float, float]:
        psi = self.psi(theta)
        grad = np.gradient(psi, self.dx, edge_order=2)
        kinetic = 0.5 * float(np.trapezoid(grad**2, self.x))
        potential = float(np.trapezoid(self.potential * psi**2, self.x))
        total = kinetic + potential
        return total, kinetic, potential

    def energy(self, theta: np.ndarray) -> float:
        total, _, _ = self.energy_components(theta)
        return total

    def gradient(self, theta: np.ndarray) -> np.ndarray:
        theta = self.project(theta)
        eps = np.array([2e-4, 2e-4, 3e-4, 2e-4, 2e-4], dtype=float)
        grad = np.zeros(5, dtype=float)
        for idx in range(5):
            delta = np.zeros(5, dtype=float)
            delta[idx] = eps[idx]
            grad[idx] = (self.energy(self.project(theta + delta)) - self.energy(self.project(theta - delta))) / (2.0 * eps[idx])
        return grad


def make_optimizer(problem: VariationalQuantumProblem) -> GONMOptimizer:
    return GONMOptimizer(
        problem=problem,
        noise=CoolingNoise(start_std=0.0010, mid_std=0.00045, late_std=0.00010, final_std=0.0),
        config=GONMConfig(
            population=84,
            phase1_keep=10,
            phase2_steps=210,
            phase3_steps=96,
            box_scale=1.8,
            thermal_scale=0.08,
            momentum=0.88,
            step_size=0.055,
            terminal_radii=(0.16, 0.09, 0.045, 0.020),
        ),
    )


def adam_baseline(
    problem: VariationalQuantumProblem,
    start: np.ndarray,
    steps: int = 240,
    learning_rate: float = 0.040,
) -> tuple[np.ndarray, list[np.ndarray], list[float]]:
    x = problem.project(start)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    beta1 = 0.9
    beta2 = 0.995
    eps = 1e-8
    trajectory = [x.copy()]
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
        trajectory.append(x.copy())
        losses.append(loss)
        if loss < best_loss:
            best_loss = loss
            best_x = x.copy()
    return best_x, trajectory, losses


def render_result(
    problem: VariationalQuantumProblem,
    adam_best: np.ndarray,
    adam_losses: list[float],
    gonm_result,
) -> plt.Figure:
    theta_phase1 = gonm_result.phase1_best_positions.reshape(-1)
    theta_final = gonm_result.final_best_positions.reshape(-1)
    psi_adam = problem.psi(adam_best)
    psi_gonm = problem.psi(theta_final)
    psi_phase1 = problem.psi(theta_phase1)

    _, adam_kin, adam_pot = problem.energy_components(adam_best)
    gonm_total, gonm_kin, gonm_pot = problem.energy_components(theta_final)
    phase1_total, _, _ = problem.energy_components(theta_phase1)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.7, 1.1])
    ax_wave = fig.add_subplot(gs[0, 0])
    ax_trace = fig.add_subplot(gs[0, 1])
    ax_potential = fig.add_subplot(gs[1, 0])
    ax_text = fig.add_subplot(gs[1, 1])

    ax_wave.plot(problem.x, psi_phase1, color="#93c5fd", linewidth=2.0, label="GONM phase 1")
    ax_wave.plot(problem.x, psi_adam, color="#ef4444", linewidth=2.2, label="baseline local")
    ax_wave.plot(problem.x, psi_gonm, color="#1d4ed8", linewidth=2.4, label="GONM final")
    ax_wave.set_title("Funcao de onda variacional")
    ax_wave.set_xlabel("x")
    ax_wave.set_ylabel("psi(x)")
    ax_wave.grid(True, linestyle=":", alpha=0.4)
    ax_wave.legend(loc="best")

    ax_trace.plot(adam_losses, color="#dc2626", linewidth=2.0, label="local Adam-style")
    ax_trace.plot(gonm_result.trace_true_energy, color="#2563eb", linewidth=2.0, label="GONM")
    ax_trace.set_title("Valor esperado da energia ao longo da busca")
    ax_trace.set_xlabel("iteracao")
    ax_trace.set_ylabel("<H>")
    ax_trace.grid(True, linestyle=":", alpha=0.5)
    ax_trace.legend(loc="best")

    scaled_potential = problem.potential / np.max(problem.potential) * np.max(np.abs(psi_gonm))
    ax_potential.plot(problem.x, scaled_potential, color="#475569", linewidth=2.0, label="potencial escalado")
    ax_potential.fill_between(problem.x, 0.0, psi_gonm**2, color="#60a5fa", alpha=0.35, label="|psi_GONM|^2")
    ax_potential.fill_between(problem.x, 0.0, psi_adam**2, color="#fca5a5", alpha=0.25, label="|psi_baseline|^2")
    ax_potential.set_title("Potencial duplo/anarmonico e densidade")
    ax_potential.set_xlabel("x")
    ax_potential.set_ylabel("escala comparativa")
    ax_potential.grid(True, linestyle=":", alpha=0.4)
    ax_potential.legend(loc="best")

    gain_vs_adam = min(adam_losses) - gonm_total
    ax_text.axis("off")
    ax_text.text(
        0.0,
        1.0,
        "\n".join(
            [
                "GONM | variational quantum ground-state search",
                "",
                "Hamiltoniano 1D em unidades naturais:",
                "H = -1/2 d^2/dx^2 + V(x)",
                "",
                f"baseline local best = {min(adam_losses):.6f}",
                f"GONM phase-1 best = {phase1_total:.6f}",
                f"GONM final = {gonm_total:.6f}",
                f"ganho GONM vs baseline = {gain_vs_adam:.6f}",
                "",
                f"cinetica GONM = {gonm_kin:.6f}",
                f"potencial GONM = {gonm_pot:.6f}",
                f"cinetica baseline = {adam_kin:.6f}",
                f"potencial baseline = {adam_pot:.6f}",
                "",
                "Leitura:",
                "o GONM reorganiza a familia variacional antes",
                "da contracao local, estabilizando a descida",
                "para uma funcao de onda de menor energia.",
            ]
        ),
        ha="left",
        va="top",
        fontsize=11,
        family="monospace",
    )

    fig.suptitle("GONM | Variational Search for a Quantum Ground State", fontsize=16, y=0.98)
    fig.tight_layout()
    return fig


def write_summary(problem: VariationalQuantumProblem, adam_best: np.ndarray, adam_losses: list[float], gonm_result) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    theta_final = gonm_result.final_best_positions.reshape(-1)
    final_total, final_kin, final_pot = problem.energy_components(theta_final)
    adam_total, adam_kin, adam_pot = problem.energy_components(adam_best)

    summary = {
        "seed": SEED,
        "problem": "quantum_ground_state_variational",
        "potential": "0.18 x^6 - 1.55 x^4 + 2.7 x^2 + 0.10 x",
        "interpretation": "Variational search for a 1D ground state in a double-well anharmonic potential.",
        "baseline_local": {
            "best_energy": float(min(adam_losses)),
            "components": {
                "kinetic": float(adam_kin),
                "potential": float(adam_pot),
            },
            "parameters": problem.decode(adam_best),
        },
        "gonm": {
            "phase1_best_energy": float(gonm_result.phase1_best_energy),
            "final_energy": float(final_total),
            "components": {
                "kinetic": float(final_kin),
                "potential": float(final_pot),
            },
            "parameters": problem.decode(theta_final),
            "evals": int(gonm_result.evals),
            "runtime_ms": float(gonm_result.runtime_ms),
        },
        "comparison": {
            "gonm_minus_baseline": float(final_total - min(adam_losses)),
            "adam_start_energy": float(adam_losses[0]),
        },
    }
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    decoded_adam = problem.decode(adam_best)
    decoded_gonm = problem.decode(theta_final)
    md = f"""# GONM | Quantum Ground-State Search

This simulation treats GONM as a variational solver for a one-dimensional Schrodinger problem.

The Hamiltonian is

`H = -1/2 d^2/dx^2 + V(x)`

with

`V(x) = 0.18 x^6 - 1.55 x^4 + 2.7 x^2 + 0.10 x`.

## Recorded outcome

- local baseline best energy: `{min(adam_losses):.6f}`
- GONM final energy: `{final_total:.6f}`
- GONM gain versus baseline: `{min(adam_losses) - final_total:.6f}`

## Variational parameters

### Baseline local

- `sigma_left = {decoded_adam["sigma_left"]:.4f}`
- `sigma_right = {decoded_adam["sigma_right"]:.4f}`
- `half_distance = {decoded_adam["half_distance"]:.4f}`
- `amplitude_ratio = {decoded_adam["amplitude_ratio"]:.4f}`
- `center_shift = {decoded_adam["center_shift"]:.4f}`

### GONM

- `sigma_left = {decoded_gonm["sigma_left"]:.4f}`
- `sigma_right = {decoded_gonm["sigma_right"]:.4f}`
- `half_distance = {decoded_gonm["half_distance"]:.4f}`
- `amplitude_ratio = {decoded_gonm["amplitude_ratio"]:.4f}`
- `center_shift = {decoded_gonm["center_shift"]:.4f}`

## Interpretation

This is a variational demonstration, not a proof of exact convergence for the full time-dependent Schrodinger equation. The point is narrower and more honest: with a physically meaningful Hamiltonian and an explicit wavefunction family, the layered GONM search finds a lower-energy state than a purely local optimizer started in a poor region.
"""
    SUMMARY_MD_PATH.write_text(md, encoding="utf-8")


def main() -> None:
    problem = VariationalQuantumProblem()
    optimizer = make_optimizer(problem)
    gonm_result = optimizer.optimize(seed=SEED)

    bad_local_start = np.array([0.95, -0.75, 0.18, -2.0, 0.55], dtype=float)
    adam_best, _adam_traj, adam_losses = adam_baseline(problem, start=bad_local_start)

    fig = render_result(problem, adam_best, adam_losses, gonm_result)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    if "--no-show" not in sys.argv and "--save" not in sys.argv:
        plt.show()
    plt.close(fig)

    write_summary(problem, adam_best, adam_losses, gonm_result)


if __name__ == "__main__":
    main()
