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


SEED = 97
RESULT_DIR = ROOT / "publications" / "T07_GONM" / "results" / "gonm_control_inverted_pendulum"
FIGURE_PATH = RESULT_DIR / "gonm_control_inverted_pendulum.png"
SUMMARY_JSON_PATH = RESULT_DIR / "summary.json"
SUMMARY_MD_PATH = RESULT_DIR / "summary.md"


@dataclass(slots=True)
class InvertedPendulumGainTuningProblem:
    horizon_steps: int = 170
    dt: float = 0.03
    gravity: float = 9.81
    length: float = 0.75
    damping: float = 0.16
    control_limit: float = 14.0
    gust_scale: float = 3.6
    initial_theta: float = 0.36
    initial_omega: float = 0.0
    gust_profile: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        t = np.linspace(0.0, 1.0, self.horizon_steps, dtype=float)
        gust = 0.85 * np.sin(2.4 * np.pi * t + 0.55)
        gust += 0.70 * np.exp(-((t - 0.24) ** 2) / 0.004)
        gust -= 0.92 * np.exp(-((t - 0.61) ** 2) / 0.006)
        gust += 0.28 * np.sin(10.0 * np.pi * t)
        self.gust_profile = self.gust_scale * gust

    @property
    def dimensions(self) -> int:
        return 4

    def project(self, x: np.ndarray) -> np.ndarray:
        gains = np.asarray(x, dtype=float).reshape(-1)
        return np.array(
            [
                np.clip(gains[0], 0.0, 34.0),   # kp
                np.clip(gains[1], 0.0, 12.0),   # kd
                np.clip(gains[2], 0.0, 8.0),    # ki
                np.clip(gains[3], 0.0, 4.0),    # anti-windup leak
            ],
            dtype=float,
        )

    def random_geometry(self, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
        trial = np.array(
            [
                rng.uniform(6.0, 26.0),
                rng.uniform(1.2, 9.0),
                rng.uniform(0.0, 4.0),
                rng.uniform(0.0, 2.6),
            ],
            dtype=float,
        )
        trial += rng.normal(scale=0.8 * min(scale, 2.0), size=4)
        return self.project(trial)

    def random_direction(self, rng: np.random.Generator) -> np.ndarray:
        direction = rng.normal(size=4)
        return direction / max(np.linalg.norm(direction), 1e-9)

    def thermal_noise(self, rng: np.random.Generator, scale: float) -> np.ndarray:
        weights = np.array([1.0, 0.45, 0.30, 0.20], dtype=float)
        return rng.normal(size=4) * weights * scale

    def radial_signature(self, x: np.ndarray) -> np.ndarray:
        return self.project(x)

    def vector_to_positions(self, x: np.ndarray) -> np.ndarray:
        return self.project(x).reshape(1, -1)

    def rollout(self, x: np.ndarray) -> dict:
        kp, kd, ki, leak = self.project(x)
        theta = float(self.initial_theta)
        omega = float(self.initial_omega)
        integ = 0.0
        thetas = [theta]
        omegas = [omega]
        torques = []
        integrals = [integ]
        for k in range(self.horizon_steps):
            integ = (1.0 - leak * self.dt) * integ + self.dt * theta
            u = float(np.clip(-(kp * theta + kd * omega + ki * integ), -self.control_limit, self.control_limit))
            wind = float(self.gust_profile[k])
            theta_ddot = (self.gravity / self.length) * np.sin(theta) - self.damping * omega + u + wind
            omega += self.dt * theta_ddot
            theta += self.dt * omega
            theta = float(np.arctan2(np.sin(theta), np.cos(theta)))
            thetas.append(theta)
            omegas.append(omega)
            torques.append(u)
            integrals.append(integ)
        return {
            "theta": np.asarray(thetas, dtype=float),
            "omega": np.asarray(omegas, dtype=float),
            "torque": np.asarray(torques, dtype=float),
            "integral": np.asarray(integrals, dtype=float),
        }

    def energy_terms(self, x: np.ndarray) -> dict:
        rollout = self.rollout(x)
        theta = rollout["theta"][:-1]
        omega = rollout["omega"][:-1]
        torque = rollout["torque"]
        integ = rollout["integral"][:-1]

        tracking = float(np.mean(theta**2))
        angular = 0.18 * float(np.mean(omega**2))
        control = 0.010 * float(np.mean(torque**2))
        integral_cost = 0.020 * float(np.mean(integ**2))
        terminal = 2.8 * float(theta[-1] ** 2) + 0.45 * float(omega[-1] ** 2)
        overshoot = 0.45 * float(np.mean(np.maximum(0.0, np.abs(theta) - 0.42) ** 2))
        flip_penalty = 5.0 * float(np.mean(np.maximum(0.0, np.abs(theta) - 0.95) ** 2))
        action_like = tracking + angular + control + integral_cost + terminal + overshoot + flip_penalty
        return {
            "action_like": float(action_like),
            "tracking": tracking,
            "angular": float(angular),
            "control": float(control),
            "integral_cost": float(integral_cost),
            "terminal": float(terminal),
            "overshoot": float(overshoot),
            "flip_penalty": float(flip_penalty),
            "final_theta": float(theta[-1]),
            "final_omega": float(omega[-1]),
            "max_abs_theta": float(np.max(np.abs(theta))),
            "rms_theta": float(np.sqrt(np.mean(theta**2))),
        }

    def energy(self, x: np.ndarray) -> float:
        return self.energy_terms(x)["action_like"]

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = self.project(x)
        eps = np.array([2e-4, 2e-4, 2e-4, 2e-4], dtype=float)
        grad = np.zeros(4, dtype=float)
        for idx in range(4):
            delta = np.zeros(4, dtype=float)
            delta[idx] = eps[idx]
            grad[idx] = (self.energy(self.project(x + delta)) - self.energy(self.project(x - delta))) / (2.0 * eps[idx])
        return grad


def make_optimizer(problem: InvertedPendulumGainTuningProblem) -> GONMOptimizer:
    return GONMOptimizer(
        problem=problem,
        noise=CoolingNoise(start_std=0.0018, mid_std=0.0007, late_std=0.00015, final_std=0.0),
        config=GONMConfig(
            population=56,
            phase1_keep=8,
            phase2_steps=120,
            phase3_steps=48,
            box_scale=1.5,
            thermal_scale=0.22,
            momentum=0.90,
            step_size=0.060,
            terminal_radii=(1.2, 0.6, 0.3, 0.12),
        ),
    )


def baseline_pd(problem: InvertedPendulumGainTuningProblem) -> tuple[np.ndarray, dict]:
    gains = np.array([16.0, 5.2, 0.0, 0.0], dtype=float)
    return gains, problem.rollout(gains)


def render_result(problem: InvertedPendulumGainTuningProblem, baseline_gains: np.ndarray, baseline_rollout: dict, gonm_result) -> plt.Figure:
    baseline_terms = problem.energy_terms(baseline_gains)
    gonm_gains = gonm_result.final_best_positions.reshape(-1)
    gonm_rollout = problem.rollout(gonm_gains)
    gonm_terms = problem.energy_terms(gonm_gains)

    time_state = np.linspace(0.0, problem.horizon_steps * problem.dt, problem.horizon_steps + 1, dtype=float)
    time_ctrl = np.linspace(0.0, (problem.horizon_steps - 1) * problem.dt, problem.horizon_steps, dtype=float)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1.1])
    ax_angle = fig.add_subplot(gs[0, 0])
    ax_control = fig.add_subplot(gs[0, 1])
    ax_trace = fig.add_subplot(gs[1, 0])
    ax_text = fig.add_subplot(gs[1, 1])

    ax_angle.plot(time_state, baseline_rollout["theta"], color="#ef4444", linewidth=2.1, label="PD baseline")
    ax_angle.plot(time_state, gonm_rollout["theta"], color="#2563eb", linewidth=2.3, label="GONM tuned PID")
    ax_angle.plot(time_ctrl, problem.gust_profile / 20.0, color="#64748b", linewidth=1.6, linestyle="--", label="wind / 20")
    ax_angle.axhline(0.0, color="#111827", linewidth=1.1)
    ax_angle.set_title("Estabilizacao angular sob rajadas")
    ax_angle.set_xlabel("tempo")
    ax_angle.set_ylabel("angulo [rad]")
    ax_angle.grid(True, linestyle=":", alpha=0.45)
    ax_angle.legend(loc="best")

    ax_control.plot(time_ctrl, baseline_rollout["torque"], color="#f87171", linewidth=2.0, label="PD torque")
    ax_control.plot(time_ctrl, gonm_rollout["torque"], color="#60a5fa", linewidth=2.0, label="GONM torque")
    ax_control.set_title("Sinal de controle")
    ax_control.set_xlabel("tempo")
    ax_control.set_ylabel("torque")
    ax_control.grid(True, linestyle=":", alpha=0.45)
    ax_control.legend(loc="best")

    ax_trace.plot(gonm_result.trace_true_energy, color="#2563eb", linewidth=2.0, label="GONM objective")
    ax_trace.axhline(baseline_terms["action_like"], color="#dc2626", linewidth=1.8, linestyle="--", label="PD objective")
    ax_trace.set_title("Funcional de controle")
    ax_trace.set_xlabel("iteracao")
    ax_trace.set_ylabel("J[K]")
    ax_trace.grid(True, linestyle=":", alpha=0.45)
    ax_trace.legend(loc="best")

    gain = baseline_terms["action_like"] - gonm_terms["action_like"]
    ax_text.axis("off")
    ax_text.text(
        0.0,
        1.0,
        "\n".join(
            [
                "GONM | inverted pendulum feedback tuning",
                "",
                f"PD baseline objective = {baseline_terms['action_like']:.6f}",
                f"GONM tuned objective = {gonm_terms['action_like']:.6f}",
                f"ganho GONM vs PD = {gain:.6f}",
                "",
                f"baseline rms(theta) = {baseline_terms['rms_theta']:.6f}",
                f"baseline max |theta| = {baseline_terms['max_abs_theta']:.6f}",
                f"baseline final theta = {baseline_terms['final_theta']:.6f}",
                "",
                f"GONM rms(theta) = {gonm_terms['rms_theta']:.6f}",
                f"GONM max |theta| = {gonm_terms['max_abs_theta']:.6f}",
                f"GONM final theta = {gonm_terms['final_theta']:.6f}",
                "",
                "GONM gains:",
                f"kp = {gonm_gains[0]:.3f}",
                f"kd = {gonm_gains[1]:.3f}",
                f"ki = {gonm_gains[2]:.3f}",
                f"leak = {gonm_gains[3]:.3f}",
            ]
        ),
        ha="left",
        va="top",
        fontsize=11,
        family="monospace",
    )

    fig.suptitle("GONM | Robust Feedback Tuning for an Inverted Pendulum", fontsize=16, y=0.98)
    fig.tight_layout()
    return fig


def write_summary(problem: InvertedPendulumGainTuningProblem, baseline_gains: np.ndarray, baseline_rollout: dict, gonm_result) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    baseline_terms = problem.energy_terms(baseline_gains)
    gonm_gains = gonm_result.final_best_positions.reshape(-1)
    gonm_terms = problem.energy_terms(gonm_gains)
    summary = {
        "seed": SEED,
        "problem": "control_inverted_pendulum_feedback_tuning",
        "interpretation": "Robust feedback-gain tuning for an inverted pendulum under strong wind gusts.",
        "baseline_pd": {
            "gains": baseline_gains.tolist(),
            "objective": float(baseline_terms["action_like"]),
            "terms": baseline_terms,
        },
        "gonm": {
            "phase1_best_energy": float(gonm_result.phase1_best_energy),
            "final_gains": gonm_gains.tolist(),
            "final_energy": float(gonm_terms["action_like"]),
            "terms": gonm_terms,
            "evals": int(gonm_result.evals),
            "runtime_ms": float(gonm_result.runtime_ms),
        },
        "comparison": {
            "gonm_minus_pd": float(gonm_terms["action_like"] - baseline_terms["action_like"]),
        },
    }
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = f"""# GONM | Inverted Pendulum Feedback Tuning

This simulation treats GONM as a robust feedback-gain tuner for an inverted pendulum under strong wind disturbances.

## Recorded outcome

- PD baseline objective: `{baseline_terms["action_like"]:.6f}`
- GONM tuned objective: `{gonm_terms["action_like"]:.6f}`
- GONM gain versus PD: `{baseline_terms["action_like"] - gonm_terms["action_like"]:.6f}`
- baseline RMS angle: `{baseline_terms["rms_theta"]:.6f}`
- GONM RMS angle: `{gonm_terms["rms_theta"]:.6f}`

## Tuned gains

- `kp = {gonm_gains[0]:.4f}`
- `kd = {gonm_gains[1]:.4f}`
- `ki = {gonm_gains[2]:.4f}`
- `leak = {gonm_gains[3]:.4f}`

## Interpretation

This is not a full industrial MPC stack. The narrower claim is still useful: the layered GONM search can tune a feedback law that contracts the disturbed pendulum back toward equilibrium better than a simple hand-tuned PD controller.
"""
    SUMMARY_MD_PATH.write_text(md, encoding="utf-8")


def main() -> None:
    problem = InvertedPendulumGainTuningProblem()
    baseline_gains, baseline_rollout = baseline_pd(problem)
    optimizer = make_optimizer(problem)
    gonm_result = optimizer.optimize(seed=SEED)

    fig = render_result(problem, baseline_gains, baseline_rollout, gonm_result)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    if "--no-show" not in sys.argv and "--save" not in sys.argv:
        plt.show()
    plt.close(fig)

    write_summary(problem, baseline_gains, baseline_rollout, gonm_result)


if __name__ == "__main__":
    main()
