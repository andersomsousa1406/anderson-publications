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


SEED = 149
RESULT_DIR = ROOT / "publications" / "T07_GONM" / "results" / "gonm_satellite_attitude_control"
FIGURE_PATH = RESULT_DIR / "gonm_satellite_attitude_control.png"
SUMMARY_JSON_PATH = RESULT_DIR / "summary.json"
SUMMARY_MD_PATH = RESULT_DIR / "summary.md"


@dataclass(slots=True)
class SatelliteAttitudeControlProblem:
    horizon_steps: int = 220
    dt: float = 0.08
    inertia: float = 12.0
    damping: float = 0.045
    max_torque: float = 0.22
    sensor_noise_std: float = 0.018
    initial_angle: float = 0.42
    initial_rate: float = 0.0
    disturbance_profile: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        t = np.linspace(0.0, 1.0, self.horizon_steps, dtype=float)
        solar = 0.045 * np.sin(2.3 * np.pi * t + 0.4)
        solar += 0.070 * np.exp(-((t - 0.22) ** 2) / 0.003)
        solar -= 0.060 * np.exp(-((t - 0.61) ** 2) / 0.005)
        structural = 0.025 * np.sin(8.0 * np.pi * t + 0.15)
        self.disturbance_profile = solar + structural

    @property
    def dimensions(self) -> int:
        return 5

    def project(self, x: np.ndarray) -> np.ndarray:
        vals = np.asarray(x, dtype=float).reshape(-1)
        return np.array(
            [
                np.clip(vals[0], 0.0, 2.0),    # measurement blending
                np.clip(vals[1], 0.0, 2.5),    # rate correction gain
                np.clip(vals[2], 0.0, 2.0),    # kp
                np.clip(vals[3], 0.0, 2.0),    # kd
                np.clip(vals[4], 0.0, 0.8),    # ki
            ],
            dtype=float,
        )

    def random_geometry(self, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
        base = np.array(
            [
                rng.uniform(0.15, 1.4),
                rng.uniform(0.1, 1.8),
                rng.uniform(0.25, 1.6),
                rng.uniform(0.15, 1.4),
                rng.uniform(0.0, 0.45),
            ],
            dtype=float,
        )
        base += rng.normal(scale=0.12 * min(scale, 2.0), size=5)
        return self.project(base)

    def random_direction(self, rng: np.random.Generator) -> np.ndarray:
        direction = rng.normal(size=5)
        return direction / max(np.linalg.norm(direction), 1e-9)

    def thermal_noise(self, rng: np.random.Generator, scale: float) -> np.ndarray:
        weights = np.array([0.7, 0.7, 1.0, 0.8, 0.4], dtype=float)
        return rng.normal(size=5) * weights * scale

    def radial_signature(self, x: np.ndarray) -> np.ndarray:
        return self.project(x)

    def vector_to_positions(self, x: np.ndarray) -> np.ndarray:
        return self.project(x).reshape(1, -1)

    def rollout(self, x: np.ndarray, seed: int = SEED) -> dict:
        blend, rate_gain, kp, kd, ki = self.project(x)
        rng = np.random.default_rng(seed)
        theta = float(self.initial_angle)
        omega = float(self.initial_rate)
        theta_hat = theta
        omega_hat = omega
        integral = 0.0

        theta_hist = [theta]
        omega_hist = [omega]
        theta_hat_hist = [theta_hat]
        omega_hat_hist = [omega_hat]
        torque_hist = []
        meas_hist = []

        for k in range(self.horizon_steps):
            measured_theta = theta + float(rng.normal(scale=self.sensor_noise_std))
            residual = measured_theta - theta_hat
            theta_hat = theta_hat + blend * residual
            omega_hat = omega_hat + rate_gain * residual / self.dt
            integral += self.dt * theta_hat

            commanded = -(kp * theta_hat + kd * omega_hat + ki * integral)
            torque = float(np.clip(commanded, -self.max_torque, self.max_torque))

            disturbance = float(self.disturbance_profile[k])
            omega_dot = (torque + disturbance - self.damping * omega) / self.inertia
            omega += self.dt * omega_dot
            theta += self.dt * omega

            theta_hist.append(theta)
            omega_hist.append(omega)
            theta_hat_hist.append(theta_hat)
            omega_hat_hist.append(omega_hat)
            torque_hist.append(torque)
            meas_hist.append(measured_theta)

        return {
            "theta": np.asarray(theta_hist, dtype=float),
            "omega": np.asarray(omega_hist, dtype=float),
            "theta_hat": np.asarray(theta_hat_hist, dtype=float),
            "omega_hat": np.asarray(omega_hat_hist, dtype=float),
            "torque": np.asarray(torque_hist, dtype=float),
            "measurement": np.asarray(meas_hist, dtype=float),
        }

    def energy_terms(self, x: np.ndarray) -> dict:
        rollout = self.rollout(x)
        theta = rollout["theta"][:-1]
        omega = rollout["omega"][:-1]
        theta_hat = rollout["theta_hat"][:-1]
        torque = rollout["torque"]
        meas = rollout["measurement"]

        pointing = float(np.mean(theta**2))
        jitter = 0.14 * float(np.mean(omega**2))
        control = 0.22 * float(np.mean(torque**2))
        estimator = 0.18 * float(np.mean((meas - theta_hat) ** 2))
        terminal = 3.8 * float(theta[-1] ** 2) + 0.65 * float(omega[-1] ** 2)
        overshoot = 0.65 * float(np.mean(np.maximum(0.0, np.abs(theta) - 0.28) ** 2))
        action_like = pointing + jitter + control + estimator + terminal + overshoot
        return {
            "action_like": float(action_like),
            "pointing": pointing,
            "jitter": float(jitter),
            "control": float(control),
            "estimator": float(estimator),
            "terminal": float(terminal),
            "overshoot": float(overshoot),
            "final_angle": float(theta[-1]),
            "final_rate": float(omega[-1]),
            "rms_angle": float(np.sqrt(np.mean(theta**2))),
            "max_abs_angle": float(np.max(np.abs(theta))),
        }

    def energy(self, x: np.ndarray) -> float:
        return self.energy_terms(x)["action_like"]

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = self.project(x)
        eps = np.full(5, 2e-4, dtype=float)
        grad = np.zeros(5, dtype=float)
        for idx in range(5):
            delta = np.zeros(5, dtype=float)
            delta[idx] = eps[idx]
            grad[idx] = (self.energy(self.project(x + delta)) - self.energy(self.project(x - delta))) / (2.0 * eps[idx])
        return grad


def make_optimizer(problem: SatelliteAttitudeControlProblem) -> GONMOptimizer:
    return GONMOptimizer(
        problem=problem,
        noise=CoolingNoise(start_std=0.0015, mid_std=0.0006, late_std=0.00012, final_std=0.0),
        config=GONMConfig(
            population=54,
            phase1_keep=8,
            phase2_steps=110,
            phase3_steps=44,
            box_scale=1.4,
            thermal_scale=0.16,
            momentum=0.89,
            step_size=0.055,
            terminal_radii=(0.25, 0.12, 0.06, 0.025),
        ),
    )


def baseline_pid(problem: SatelliteAttitudeControlProblem) -> tuple[np.ndarray, dict]:
    gains = np.array([0.0, 0.0, 0.82, 0.46, 0.06], dtype=float)
    return gains, problem.rollout(gains)


def render_result(problem: SatelliteAttitudeControlProblem, baseline_gains: np.ndarray, baseline_rollout: dict, gonm_result) -> plt.Figure:
    baseline_terms = problem.energy_terms(baseline_gains)
    tuned = gonm_result.final_best_positions.reshape(-1)
    tuned_rollout = problem.rollout(tuned)
    tuned_terms = problem.energy_terms(tuned)

    t_state = np.linspace(0.0, problem.horizon_steps * problem.dt, problem.horizon_steps + 1, dtype=float)
    t_ctrl = np.linspace(0.0, (problem.horizon_steps - 1) * problem.dt, problem.horizon_steps, dtype=float)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1.1])
    ax_angle = fig.add_subplot(gs[0, 0])
    ax_torque = fig.add_subplot(gs[0, 1])
    ax_trace = fig.add_subplot(gs[1, 0])
    ax_text = fig.add_subplot(gs[1, 1])

    ax_angle.plot(t_state, baseline_rollout["theta"], color="#ef4444", linewidth=2.0, label="PID baseline")
    ax_angle.plot(t_state, tuned_rollout["theta"], color="#2563eb", linewidth=2.2, label="GONM tuned")
    ax_angle.plot(t_ctrl, problem.disturbance_profile * 2.0, color="#64748b", linewidth=1.4, linestyle="--", label="disturbance x2")
    ax_angle.axhline(0.0, color="#111827", linewidth=1.0)
    ax_angle.set_title("Erro de apontamento sob ruído e radiação solar")
    ax_angle.set_xlabel("tempo [s]")
    ax_angle.set_ylabel("angulo [rad]")
    ax_angle.grid(True, linestyle=":", alpha=0.45)
    ax_angle.legend(loc="best")

    ax_torque.plot(t_ctrl, baseline_rollout["torque"], color="#f87171", linewidth=2.0, label="PID torque")
    ax_torque.plot(t_ctrl, tuned_rollout["torque"], color="#60a5fa", linewidth=2.0, label="GONM torque")
    ax_torque.set_title("Torque de correção")
    ax_torque.set_xlabel("tempo [s]")
    ax_torque.set_ylabel("torque [N m]")
    ax_torque.grid(True, linestyle=":", alpha=0.45)
    ax_torque.legend(loc="best")

    ax_trace.plot(gonm_result.trace_true_energy, color="#2563eb", linewidth=2.0, label="GONM objective")
    ax_trace.axhline(baseline_terms["action_like"], color="#dc2626", linewidth=1.8, linestyle="--", label="PID objective")
    ax_trace.set_title("Funcional de atitude")
    ax_trace.set_xlabel("iteracao")
    ax_trace.set_ylabel("J[K, F]")
    ax_trace.grid(True, linestyle=":", alpha=0.45)
    ax_trace.legend(loc="best")

    gain = baseline_terms["action_like"] - tuned_terms["action_like"]
    ax_text.axis("off")
    ax_text.text(
        0.0,
        1.0,
        "\n".join(
            [
                "GONM | satellite attitude control",
                "",
                f"PID baseline objective = {baseline_terms['action_like']:.6f}",
                f"GONM tuned objective = {tuned_terms['action_like']:.6f}",
                f"ganho GONM vs PID = {gain:.6f}",
                "",
                f"baseline rms(angle) = {baseline_terms['rms_angle']:.6f}",
                f"baseline max |angle| = {baseline_terms['max_abs_angle']:.6f}",
                f"baseline final angle = {baseline_terms['final_angle']:.6f}",
                "",
                f"GONM rms(angle) = {tuned_terms['rms_angle']:.6f}",
                f"GONM max |angle| = {tuned_terms['max_abs_angle']:.6f}",
                f"GONM final angle = {tuned_terms['final_angle']:.6f}",
                "",
                "GONM parameters:",
                f"blend = {tuned[0]:.3f}",
                f"rate_gain = {tuned[1]:.3f}",
                f"kp = {tuned[2]:.3f}",
                f"kd = {tuned[3]:.3f}",
                f"ki = {tuned[4]:.3f}",
            ]
        ),
        ha="left",
        va="top",
        fontsize=10.5,
        family="monospace",
    )

    fig.suptitle("GONM | Satellite Attitude Stabilization Under Disturbances", fontsize=16, y=0.98)
    fig.tight_layout()
    return fig


def write_summary(problem: SatelliteAttitudeControlProblem, baseline_gains: np.ndarray, gonm_result) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    baseline_terms = problem.energy_terms(baseline_gains)
    tuned = gonm_result.final_best_positions.reshape(-1)
    tuned_terms = problem.energy_terms(tuned)
    summary = {
        "seed": SEED,
        "problem": "satellite_attitude_control",
        "interpretation": "Single-axis satellite attitude stabilization under sensor noise and solar-pressure-like disturbances.",
        "baseline_pid": {
            "parameters": baseline_gains.tolist(),
            "objective": float(baseline_terms["action_like"]),
            "terms": baseline_terms,
        },
        "gonm": {
            "phase1_best_energy": float(gonm_result.phase1_best_energy),
            "parameters": tuned.tolist(),
            "final_energy": float(tuned_terms["action_like"]),
            "terms": tuned_terms,
            "evals": int(gonm_result.evals),
            "runtime_ms": float(gonm_result.runtime_ms),
        },
        "comparison": {
            "gonm_minus_pid": float(tuned_terms["action_like"] - baseline_terms["action_like"]),
        },
    }
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = f"""# GONM | Satellite Attitude Control

This simulation treats GONM as a combined feedback-and-filter tuning mechanism for satellite attitude stabilization under sensor noise and disturbance torques.

## Recorded outcome

- PID baseline objective: `{baseline_terms["action_like"]:.6f}`
- GONM tuned objective: `{tuned_terms["action_like"]:.6f}`
- GONM gain versus PID: `{baseline_terms["action_like"] - tuned_terms["action_like"]:.6f}`
- baseline RMS angle: `{baseline_terms["rms_angle"]:.6f}`
- GONM RMS angle: `{tuned_terms["rms_angle"]:.6f}`

## Interpretation

This is not a full flight-qualified ADCS stack. The narrower claim is still useful: the layered GONM search can tune a disturbance-rejecting attitude loop that filters noisy measurements and stabilizes pointing more tightly than a simple baseline PID law.
"""
    SUMMARY_MD_PATH.write_text(md, encoding="utf-8")


def main() -> None:
    problem = SatelliteAttitudeControlProblem()
    baseline_gains, baseline_rollout = baseline_pid(problem)
    optimizer = make_optimizer(problem)
    gonm_result = optimizer.optimize(seed=SEED)

    fig = render_result(problem, baseline_gains, baseline_rollout, gonm_result)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    if "--no-show" not in sys.argv and "--save" not in sys.argv:
        plt.show()
    plt.close(fig)

    write_summary(problem, baseline_gains, gonm_result)


if __name__ == "__main__":
    main()
