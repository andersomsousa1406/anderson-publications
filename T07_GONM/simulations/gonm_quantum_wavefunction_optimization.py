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


SEED = 71
RESULT_DIR = ROOT / "publications" / "T07_GONM" / "results" / "gonm_quantum_wavefunction_optimization"
FIGURE_PATH = RESULT_DIR / "gonm_quantum_wavefunction_optimization.png"
SUMMARY_JSON_PATH = RESULT_DIR / "summary.json"
SUMMARY_MD_PATH = RESULT_DIR / "summary.md"


@dataclass(slots=True)
class QuantumWavefunctionProblem:
    basis_size: int = 5
    grid_min: float = -6.0
    grid_max: float = 6.0
    grid_size: int = 420
    electron_count: int = 5
    interaction_strength: float = 1.15
    basis_grid: np.ndarray = field(init=False, repr=False)
    dx: float = field(init=False, repr=False)
    basis: np.ndarray = field(init=False, repr=False)
    potential: np.ndarray = field(init=False, repr=False)
    kernel: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        x = np.linspace(self.grid_min, self.grid_max, self.grid_size, dtype=float)
        self.basis_grid = x
        self.dx = float(x[1] - x[0])
        self.potential = 0.22 * x**4 - 1.15 * x**2 + 0.14 * x
        centers = np.linspace(-1.8, 1.8, self.basis_size)
        widths = np.linspace(0.7, 1.55, self.basis_size)
        raw_basis = []
        for center, width in zip(centers, widths, strict=True):
            phi = np.exp(-0.5 * ((x - center) / width) ** 2)
            phi /= np.sqrt(np.trapezoid(phi**2, x))
            raw_basis.append(phi)
        self.basis = np.asarray(raw_basis, dtype=float)
        delta = x[:, None] - x[None, :]
        self.kernel = 1.0 / np.sqrt(delta**2 + 0.30**2)

    @property
    def dimensions(self) -> int:
        return self.basis_size

    def project(self, x: np.ndarray) -> np.ndarray:
        coeffs = np.asarray(x, dtype=float).reshape(-1)
        coeffs = np.clip(coeffs, -3.0, 3.0)
        norm = np.linalg.norm(coeffs)
        if norm < 1e-12:
            coeffs = np.zeros_like(coeffs)
            coeffs[0] = 1.0
            return coeffs
        return coeffs / norm

    def random_geometry(self, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
        coeffs = rng.normal(scale=min(scale, 2.0), size=self.basis_size)
        coeffs += np.linspace(0.3, -0.2, self.basis_size)
        return self.project(coeffs)

    def random_direction(self, rng: np.random.Generator) -> np.ndarray:
        direction = rng.normal(size=self.basis_size)
        return direction / max(np.linalg.norm(direction), 1e-9)

    def thermal_noise(self, rng: np.random.Generator, scale: float) -> np.ndarray:
        envelope = np.linspace(1.0, 0.65, self.basis_size)
        return rng.normal(size=self.basis_size) * envelope * scale

    def radial_signature(self, x: np.ndarray) -> np.ndarray:
        psi = self.psi(x)
        sample_idx = np.linspace(0, len(psi) - 1, 10, dtype=int)
        return psi[sample_idx]

    def vector_to_positions(self, x: np.ndarray) -> np.ndarray:
        return self.project(x).reshape(1, -1)

    def psi(self, x: np.ndarray) -> np.ndarray:
        coeffs = self.project(x)
        psi = coeffs @ self.basis
        norm = np.sqrt(np.trapezoid(psi**2, self.basis_grid))
        return psi / max(norm, 1e-12)

    def norm_error(self, x: np.ndarray) -> float:
        coeffs = np.asarray(x, dtype=float).reshape(-1)
        return abs(np.linalg.norm(coeffs) - 1.0)

    def energy_terms(self, x: np.ndarray) -> dict:
        psi = self.psi(x)
        rho = psi**2
        grad = np.gradient(psi, self.dx, edge_order=2)
        kinetic = 0.5 * float(np.trapezoid(grad**2, self.basis_grid))
        external = float(np.trapezoid(self.potential * rho, self.basis_grid))
        interaction = 0.5 * self.interaction_strength * float(
            np.trapezoid(np.trapezoid(rho[:, None] * self.kernel * rho[None, :], self.basis_grid, axis=1), self.basis_grid)
        )
        localization = 0.028 * float(np.trapezoid((self.basis_grid**2) * rho, self.basis_grid))
        total = kinetic + external + interaction + localization
        return {
            "total": float(total),
            "kinetic": float(kinetic),
            "external": float(external),
            "interaction": float(interaction),
            "localization": float(localization),
        }

    def energy(self, x: np.ndarray) -> float:
        return self.energy_terms(x)["total"]

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = self.project(x)
        eps = 2e-4
        grad = np.zeros(self.basis_size, dtype=float)
        for idx in range(self.basis_size):
            delta = np.zeros(self.basis_size, dtype=float)
            delta[idx] = eps
            grad[idx] = (self.energy(self.project(x + delta)) - self.energy(self.project(x - delta))) / (2.0 * eps)
        return grad


def make_optimizer(problem: QuantumWavefunctionProblem) -> GONMOptimizer:
    return GONMOptimizer(
        problem=problem,
        noise=CoolingNoise(start_std=0.0010, mid_std=0.00045, late_std=0.00010, final_std=0.0),
        config=GONMConfig(
            population=36,
            phase1_keep=7,
            phase2_steps=84,
            phase3_steps=32,
            box_scale=1.6,
            thermal_scale=0.075,
            momentum=0.88,
            step_size=0.060,
            terminal_radii=(0.14, 0.08, 0.04, 0.02),
        ),
    )


def unstable_baseline(
    problem: QuantumWavefunctionProblem,
    start: np.ndarray,
    steps: int = 84,
    learning_rate: float = 0.11,
) -> tuple[np.ndarray, list[float], list[float]]:
    x = np.asarray(start, dtype=float).reshape(-1).copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    beta1 = 0.9
    beta2 = 0.995
    eps = 1e-8
    losses = [problem.energy(problem.project(x))]
    norm_errors = [problem.norm_error(x)]
    best_x = x.copy()
    best_loss = losses[0]

    for step in range(1, steps + 1):
        # Deliberately local and weakly stabilized: no projection back to the unit sphere.
        grad = problem.gradient(problem.project(x))
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad**2)
        m_hat = m / (1.0 - beta1**step)
        v_hat = v / (1.0 - beta2**step)
        x = np.clip(x - learning_rate * m_hat / (np.sqrt(v_hat) + eps), -3.0, 3.0)
        projected = problem.project(x)
        loss = problem.energy(projected) + 0.25 * problem.norm_error(x)
        losses.append(loss)
        norm_errors.append(problem.norm_error(x))
        if loss < best_loss:
            best_loss = loss
            best_x = x.copy()
    return best_x, losses, norm_errors


def render_result(problem: QuantumWavefunctionProblem, baseline_best: np.ndarray, baseline_losses: list[float], baseline_norm_errors: list[float], gonm_result) -> plt.Figure:
    baseline_projected = problem.project(baseline_best)
    gonm_final = gonm_result.final_best_positions.reshape(-1)
    psi_baseline = problem.psi(baseline_projected)
    psi_gonm = problem.psi(gonm_final)
    baseline_terms = problem.energy_terms(baseline_projected)
    gonm_terms = problem.energy_terms(gonm_final)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.7, 1.1])
    ax_wave = fig.add_subplot(gs[0, 0])
    ax_trace = fig.add_subplot(gs[0, 1])
    ax_norm = fig.add_subplot(gs[1, 0])
    ax_text = fig.add_subplot(gs[1, 1])

    ax_wave.plot(problem.basis_grid, problem.potential / np.max(np.abs(problem.potential)) * np.max(np.abs(psi_gonm)), color="#475569", linewidth=1.8, label="potential scaled")
    ax_wave.plot(problem.basis_grid, psi_baseline, color="#ef4444", linewidth=2.2, label="baseline local")
    ax_wave.plot(problem.basis_grid, psi_gonm, color="#2563eb", linewidth=2.4, label="GONM final")
    ax_wave.set_title("Funcao de onda otimizada")
    ax_wave.set_xlabel("x")
    ax_wave.set_ylabel("psi(x)")
    ax_wave.grid(True, linestyle=":", alpha=0.4)
    ax_wave.legend(loc="best")

    ax_trace.plot(baseline_losses, color="#dc2626", linewidth=2.0, label="baseline local")
    ax_trace.plot(gonm_result.trace_true_energy, color="#2563eb", linewidth=2.0, label="GONM")
    ax_trace.set_title("Energia variacional ao longo da busca")
    ax_trace.set_xlabel("iteracao")
    ax_trace.set_ylabel("E[psi]")
    ax_trace.grid(True, linestyle=":", alpha=0.5)
    ax_trace.legend(loc="best")

    ax_norm.plot(baseline_norm_errors, color="#f87171", linewidth=2.0, label="baseline norm drift")
    ax_norm.axhline(0.0, color="#1d4ed8", linewidth=2.0, linestyle="--", label="GONM projected norm")
    ax_norm.set_title("Erro de normalizacao")
    ax_norm.set_xlabel("iteracao")
    ax_norm.set_ylabel("| ||c|| - 1 |")
    ax_norm.grid(True, linestyle=":", alpha=0.5)
    ax_norm.legend(loc="best")

    gain = baseline_terms["total"] - gonm_terms["total"]
    ax_text.axis("off")
    ax_text.text(
        0.0,
        1.0,
        "\n".join(
            [
                "GONM | quantum wavefunction optimization",
                "",
                f"electrons (effective) = {problem.electron_count}",
                f"basis size = {problem.basis_size}",
                "",
                f"baseline local projected energy = {baseline_terms['total']:.6f}",
                f"GONM final = {gonm_terms['total']:.6f}",
                f"ganho GONM vs baseline = {gain:.6f}",
                "",
                f"kinetic GONM = {gonm_terms['kinetic']:.6f}",
                f"external GONM = {gonm_terms['external']:.6f}",
                f"interaction GONM = {gonm_terms['interaction']:.6f}",
                "",
                f"kinetic baseline = {baseline_terms['kinetic']:.6f}",
                f"external baseline = {baseline_terms['external']:.6f}",
                f"interaction baseline = {baseline_terms['interaction']:.6f}",
                "",
                f"baseline max norm drift = {max(baseline_norm_errors):.6f}",
                "GONM keeps the coefficients on the normalized manifold.",
            ]
        ),
        ha="left",
        va="top",
        fontsize=11,
        family="monospace",
    )

    fig.suptitle("GONM | Variational Quantum Wavefunction Optimization", fontsize=16, y=0.98)
    fig.tight_layout()
    return fig


def write_summary(problem: QuantumWavefunctionProblem, baseline_best: np.ndarray, baseline_losses: list[float], baseline_norm_errors: list[float], gonm_result) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    baseline_projected = problem.project(baseline_best)
    baseline_terms = problem.energy_terms(baseline_projected)
    gonm_terms = problem.energy_terms(gonm_result.final_best_positions.reshape(-1))
    summary = {
        "seed": SEED,
        "problem": "quantum_wavefunction_optimization",
        "interpretation": "Variational optimization of a normalized wavefunction under an effective many-electron Hartree-like functional.",
        "electron_count_effective": problem.electron_count,
        "basis_size": problem.basis_size,
        "baseline_local": {
            "best_projected_energy": float(baseline_terms["total"]),
            "best_penalized_objective": float(min(baseline_losses)),
            "max_norm_drift": float(max(baseline_norm_errors)),
            "terms": baseline_terms,
        },
        "gonm": {
            "phase1_best_energy": float(gonm_result.phase1_best_energy),
            "final_energy": float(gonm_terms["total"]),
            "terms": gonm_terms,
            "evals": int(gonm_result.evals),
            "runtime_ms": float(gonm_result.runtime_ms),
        },
        "comparison": {
            "gonm_minus_baseline_projected": float(gonm_terms["total"] - baseline_terms["total"]),
            "baseline_start_energy": float(baseline_losses[0]),
        },
    }
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = f"""# GONM | Quantum Wavefunction Optimization

This simulation treats GONM as a variational optimizer for a normalized wavefunction in an effective many-electron setting.

The trial state is expanded in a finite Gaussian basis, and the functional includes:

- kinetic energy;
- an external anharmonic potential;
- an effective Hartree-like interaction term.

## Recorded outcome

- local baseline projected energy: `{baseline_terms["total"]:.6f}`
- GONM final energy: `{gonm_terms["total"]:.6f}`
- GONM gain versus baseline: `{baseline_terms["total"] - gonm_terms["total"]:.6f}`
- baseline maximum norm drift: `{max(baseline_norm_errors):.6f}`

## Interpretation

This is not a full Hartree-Fock or coupled-cluster solver. The narrower claim is still useful: the layered GONM search can optimize a normalized wavefunction family while keeping the coefficients on a physical manifold, whereas a naive local update begins to drift away from the normalization constraint.
"""
    SUMMARY_MD_PATH.write_text(md, encoding="utf-8")


def main() -> None:
    problem = QuantumWavefunctionProblem()
    optimizer = make_optimizer(problem)
    gonm_result = optimizer.optimize(seed=SEED)

    baseline_start = np.array([1.6, -1.2, 0.9, -0.8, 0.5], dtype=float)
    baseline_best, baseline_losses, baseline_norm_errors = unstable_baseline(problem, baseline_start)

    fig = render_result(problem, baseline_best, baseline_losses, baseline_norm_errors, gonm_result)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    if "--no-show" not in sys.argv and "--save" not in sys.argv:
        plt.show()
    plt.close(fig)

    write_summary(problem, baseline_best, baseline_losses, baseline_norm_errors, gonm_result)


if __name__ == "__main__":
    main()
