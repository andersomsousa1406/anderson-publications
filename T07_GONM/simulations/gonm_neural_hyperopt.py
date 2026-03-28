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


SEED = 29
RESULT_DIR = ROOT / "publications" / "T07_GONM" / "results" / "gonm_neural_hyperopt"
FIGURE_PATH = RESULT_DIR / "gonm_neural_hyperopt.png"
SUMMARY_JSON_PATH = RESULT_DIR / "summary.json"
SUMMARY_MD_PATH = RESULT_DIR / "summary.md"


@dataclass(slots=True)
class HyperparameterLandscape:
    lr_bounds: tuple[float, float] = (-4.4, -1.0)
    dropout_bounds: tuple[float, float] = (0.0, 0.55)

    @property
    def dimensions(self) -> int:
        return 2

    def project(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        return np.array(
            [
                np.clip(x[0], self.lr_bounds[0], self.lr_bounds[1]),
                np.clip(x[1], self.dropout_bounds[0], self.dropout_bounds[1]),
            ],
            dtype=float,
        )

    def random_geometry(self, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
        lr_center = 0.5 * (self.lr_bounds[0] + self.lr_bounds[1])
        dr_center = 0.5 * (self.dropout_bounds[0] + self.dropout_bounds[1])
        lr_half = 0.5 * (self.lr_bounds[1] - self.lr_bounds[0]) * min(scale / 2.0, 1.0)
        dr_half = 0.5 * (self.dropout_bounds[1] - self.dropout_bounds[0]) * min(scale / 2.0, 1.0)
        point = np.array(
            [
                rng.uniform(lr_center - lr_half, lr_center + lr_half),
                rng.uniform(dr_center - dr_half, dr_center + dr_half),
            ],
            dtype=float,
        )
        return self.project(point)

    def random_direction(self, rng: np.random.Generator) -> np.ndarray:
        direction = rng.normal(size=2)
        return direction / max(np.linalg.norm(direction), 1e-9)

    def thermal_noise(self, rng: np.random.Generator, scale: float) -> np.ndarray:
        scaled = rng.normal(size=2)
        scaled[0] *= 0.55
        scaled[1] *= 0.11
        return scaled * scale

    def radial_signature(self, x: np.ndarray) -> np.ndarray:
        return self.normalize(x)

    def vector_to_positions(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float).reshape(1, 2)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        x = self.project(x)
        u = 2.0 * (x[0] - self.lr_bounds[0]) / (self.lr_bounds[1] - self.lr_bounds[0]) - 1.0
        v = 2.0 * (x[1] - self.dropout_bounds[0]) / (self.dropout_bounds[1] - self.dropout_bounds[0]) - 1.0
        return np.array([u, v], dtype=float)

    def decode(self, x: np.ndarray) -> dict:
        x = self.project(x)
        return {
            "log10_learning_rate": float(x[0]),
            "learning_rate": float(10.0 ** x[0]),
            "dropout": float(x[1]),
        }

    def energy(self, x: np.ndarray) -> float:
        u, v = self.normalize(x)
        plateau = 0.84 + 0.06 * (1.0 - np.exp(-0.7 * ((u + 0.15) ** 2 + 0.8 * (v - 0.10) ** 2)))
        broad_local = -0.18 * np.exp(-(((u + 1.05) ** 2) / 0.24 + ((v - 0.65) ** 2) / 0.22))
        narrow_global = -0.39 * np.exp(-(((u - 0.12) ** 2) / 0.045 + ((v + 0.78) ** 2) / 0.040))
        ridge = 0.055 * np.exp(-(((u - 0.48) ** 2) / 0.16 + ((v - 0.10) ** 2) / 0.11))
        corrugation = 0.018 * np.sin(4.3 * u - 1.5 * v) ** 2 + 0.012 * np.cos(2.8 * v + 0.6) ** 2
        return float(plateau + broad_local + narrow_global + ridge + corrugation)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = self.project(x)
        eps = np.array([2e-4, 1e-4], dtype=float)
        grad = np.zeros(2, dtype=float)
        for idx in range(2):
            dx = np.zeros(2, dtype=float)
            dx[idx] = eps[idx]
            grad[idx] = (self.energy(self.project(x + dx)) - self.energy(self.project(x - dx))) / (2.0 * eps[idx])
        return grad


def make_optimizer(problem: HyperparameterLandscape) -> GONMOptimizer:
    return GONMOptimizer(
        problem=problem,
        noise=CoolingNoise(start_std=0.012, mid_std=0.005, late_std=0.0015, final_std=0.0),
        config=GONMConfig(
            population=88,
            phase1_keep=10,
            phase2_steps=210,
            phase3_steps=96,
            box_scale=2.0,
            thermal_scale=0.05,
            momentum=0.87,
            step_size=0.070,
            terminal_radii=(0.18, 0.10, 0.05, 0.025),
        ),
    )


def adam_baseline(
    problem: HyperparameterLandscape,
    start: np.ndarray,
    steps: int = 220,
    learning_rate: float = 0.045,
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


def surface_grid(problem: HyperparameterLandscape, grid_n: int = 220) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lr = np.linspace(problem.lr_bounds[0], problem.lr_bounds[1], grid_n)
    dropout = np.linspace(problem.dropout_bounds[0], problem.dropout_bounds[1], grid_n)
    xx, yy = np.meshgrid(lr, dropout)
    zz = np.zeros_like(xx)
    for i in range(grid_n):
        for j in range(grid_n):
            zz[i, j] = problem.energy(np.array([xx[i, j], yy[i, j]], dtype=float))
    return xx, yy, zz


def hyperparam_table(problem: HyperparameterLandscape, x: np.ndarray) -> str:
    decoded = problem.decode(x)
    return (
        f"log10(lr) = {decoded['log10_learning_rate']:.4f}\n"
        f"lr = {decoded['learning_rate']:.6f}\n"
        f"dropout = {decoded['dropout']:.4f}"
    )


def render_result(
    problem: HyperparameterLandscape,
    adam_best: np.ndarray,
    adam_trajectory: list[np.ndarray],
    adam_losses: list[float],
    gonm_result,
    xx: np.ndarray,
    yy: np.ndarray,
    zz: np.ndarray,
) -> plt.Figure:
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1.1])
    ax_surface = fig.add_subplot(gs[0, 0])
    ax_traces = fig.add_subplot(gs[0, 1])
    ax_profile = fig.add_subplot(gs[1, 0])
    ax_text = fig.add_subplot(gs[1, 1])

    contour = ax_surface.contourf(xx, yy, zz, levels=28, cmap="viridis")
    fig.colorbar(contour, ax=ax_surface, shrink=0.86, label="validation loss proxy")
    adam_path = np.asarray(adam_trajectory)
    ax_surface.plot(adam_path[:, 0], adam_path[:, 1], color="#ef4444", linewidth=2.0, label="Adam trajectory")
    ax_surface.scatter(adam_path[0, 0], adam_path[0, 1], color="#f59e0b", s=95, label="Adam start")
    ax_surface.scatter(adam_best[0], adam_best[1], color="#dc2626", s=100, label="Adam best")

    phase1 = gonm_result.phase1_best_positions.reshape(-1)
    final = gonm_result.final_best_positions.reshape(-1)
    ax_surface.scatter(phase1[0], phase1[1], color="#60a5fa", s=90, label="GONM structural best")
    ax_surface.scatter(final[0], final[1], color="#1d4ed8", s=120, marker="*", label="GONM final best")
    ax_surface.set_title("Paisagem proxy de validacao para hiperparametros")
    ax_surface.set_xlabel("log10(learning rate)")
    ax_surface.set_ylabel("dropout")
    ax_surface.legend(loc="lower left")

    ax_traces.plot(adam_losses, color="#dc2626", linewidth=2.0, label="Adam")
    ax_traces.plot(gonm_result.trace_true_energy, color="#2563eb", linewidth=2.0, label="GONM")
    ax_traces.set_title("Perda ao longo da busca")
    ax_traces.set_xlabel("iteracao")
    ax_traces.set_ylabel("validation loss proxy")
    ax_traces.grid(True, linestyle=":", alpha=0.5)
    ax_traces.legend(loc="best")

    adam_decoded = problem.decode(adam_best)
    gonm_decoded = problem.decode(final)
    labels = ["learning rate", "dropout"]
    adam_vals = [adam_decoded["learning_rate"], adam_decoded["dropout"]]
    gonm_vals = [gonm_decoded["learning_rate"], gonm_decoded["dropout"]]
    x = np.arange(len(labels))
    width = 0.35
    ax_profile.bar(x - width / 2, adam_vals, width=width, color="#f87171", label="Adam")
    ax_profile.bar(x + width / 2, gonm_vals, width=width, color="#60a5fa", label="GONM")
    ax_profile.set_yscale("log")
    ax_profile.set_xticks(x)
    ax_profile.set_xticklabels(labels)
    ax_profile.set_title("Hiperparametros selecionados")
    ax_profile.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax_profile.legend(loc="best")

    improvement = adam_losses[-1] - gonm_result.final_true_energy
    ax_text.axis("off")
    ax_text.text(
        0.0,
        1.0,
        "\n".join(
            [
                "GONM | neural hyperparameter optimization",
                "",
                "Paisagem sintética com planalto, bacia local e vale estreito global.",
                "",
                f"Adam final loss = {adam_losses[-1]:.6f}",
                f"Adam best loss = {min(adam_losses):.6f}",
                f"GONM final loss = {gonm_result.final_true_energy:.6f}",
                f"ganho GONM vs Adam final = {improvement:.6f}",
                "",
                "Adam best:",
                hyperparam_table(problem, adam_best),
                "",
                "GONM best:",
                hyperparam_table(problem, final),
                "",
                f"GONM evaluations = {gonm_result.evals}",
                f"GONM runtime = {gonm_result.runtime_ms:.1f} ms",
                "",
                "Leitura:",
                "o GONM usa exploracao estrutural mais contracao local",
                "para escapar do planalto e entrar numa bacia mais estreita.",
            ]
        ),
        ha="left",
        va="top",
        fontsize=11,
        family="monospace",
    )

    fig.suptitle("GONM as a super-optimizer for plateau-heavy hyperparameter search", fontsize=16, y=0.98)
    fig.tight_layout()
    return fig


def write_summary(problem: HyperparameterLandscape, adam_best: np.ndarray, adam_losses: list[float], gonm_result) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    final = gonm_result.final_best_positions.reshape(-1)
    summary = {
        "seed": SEED,
        "problem": "neural_hyperparameter_proxy",
        "interpretation": "Synthetic validation-loss landscape with plateaus and local minima for neural hyperparameter search.",
        "adam": {
            "final_loss": float(adam_losses[-1]),
            "best_loss": float(min(adam_losses)),
            "best_hyperparameters": problem.decode(adam_best),
        },
        "gonm": {
            "final_loss": float(gonm_result.final_true_energy),
            "phase1_best_loss": float(gonm_result.phase1_best_energy),
            "energy_gain_vs_phase1": float(gonm_result.energy_gain_vs_phase1),
            "best_hyperparameters": problem.decode(final),
            "evals": int(gonm_result.evals),
            "runtime_ms": float(gonm_result.runtime_ms),
        },
        "comparison": {
            "gonm_minus_adam_final": float(gonm_result.final_true_energy - adam_losses[-1]),
            "gonm_minus_adam_best": float(gonm_result.final_true_energy - min(adam_losses)),
            "adam_start_loss": float(adam_losses[0]),
        },
    }
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = f"""# GONM Neural Hyperparameter Optimization

This simulation models a noisy hyperparameter search landscape for neural-network training.

The proxy variables are:

- `log10(learning rate)`
- `dropout`

The landscape intentionally contains:

- a broad plateau with weak gradients;
- a locally attractive but suboptimal basin;
- a narrower global basin that rewards structured exploration.

## Recorded outcome

- Adam final loss: `{adam_losses[-1]:.6f}`
- Adam best loss: `{min(adam_losses):.6f}`
- GONM final loss: `{gonm_result.final_true_energy:.6f}`
- GONM gain versus Adam final: `{adam_losses[-1] - gonm_result.final_true_energy:.6f}`

## Best hyperparameters

### Adam

- `log10(lr) = {problem.decode(adam_best)["log10_learning_rate"]:.4f}`
- `lr = {problem.decode(adam_best)["learning_rate"]:.6f}`
- `dropout = {problem.decode(adam_best)["dropout"]:.4f}`

### GONM

- `log10(lr) = {problem.decode(final)["log10_learning_rate"]:.4f}`
- `lr = {problem.decode(final)["learning_rate"]:.6f}`
- `dropout = {problem.decode(final)["dropout"]:.4f}`

## Interpretation

This is not a proof that GONM replaces Adam or SGD in real large-scale training. The demonstration is narrower and more honest: in plateau-heavy hyperparameter landscapes, the layered GONM search can enter better basins than a local gradient-only baseline started from a poor region.
"""
    SUMMARY_MD_PATH.write_text(md, encoding="utf-8")


def main() -> None:
    problem = HyperparameterLandscape()
    optimizer = make_optimizer(problem)
    gonm_result = optimizer.optimize(seed=SEED)

    adam_start = np.array([-4.0, 0.44], dtype=float)
    adam_best, adam_trajectory, adam_losses = adam_baseline(problem, start=adam_start)

    xx, yy, zz = surface_grid(problem)
    fig = render_result(problem, adam_best, adam_trajectory, adam_losses, gonm_result, xx, yy, zz)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    if "--no-show" not in sys.argv and "--save" not in sys.argv:
        plt.show()
    plt.close(fig)

    write_summary(problem, adam_best, adam_losses, gonm_result)


if __name__ == "__main__":
    main()
