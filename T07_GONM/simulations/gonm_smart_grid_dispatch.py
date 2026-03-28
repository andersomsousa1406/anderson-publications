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


SEED = 173
RESULT_DIR = ROOT / "publications" / "T07_GONM" / "results" / "gonm_smart_grid_dispatch"
FIGURE_PATH = RESULT_DIR / "gonm_smart_grid_dispatch.png"
SUMMARY_JSON_PATH = RESULT_DIR / "summary.json"
SUMMARY_MD_PATH = RESULT_DIR / "summary.md"


@dataclass(slots=True)
class SmartGridDispatchProblem:
    demand: np.ndarray = field(default_factory=lambda: np.array([1.10, 1.35, 0.90, 1.25, 0.85, 1.05], dtype=float))
    solar: np.ndarray = field(default_factory=lambda: np.array([0.15, 0.55, 0.45, 0.10, 0.30, 0.20], dtype=float))
    battery_max: np.ndarray = field(default_factory=lambda: np.array([0.25, 0.00, 0.20, 0.00, 0.30, 0.10], dtype=float))
    thermal_max: np.ndarray = field(default_factory=lambda: np.array([1.80, 1.50, 1.20, 1.60, 1.00, 1.10], dtype=float))
    thermal_min: np.ndarray = field(default_factory=lambda: np.array([0.10, 0.10, 0.05, 0.10, 0.00, 0.00], dtype=float))
    thermal_cost: np.ndarray = field(default_factory=lambda: np.array([24.0, 28.0, 33.0, 26.0, 42.0, 46.0], dtype=float))
    batt_cost: np.ndarray = field(default_factory=lambda: np.array([7.0, 0.0, 6.0, 0.0, 8.0, 5.5], dtype=float))
    capacities: np.ndarray = field(default_factory=lambda: np.array([2.2, 1.7, 1.5, 1.8, 1.6, 1.9, 1.4], dtype=float))
    edges: np.ndarray = field(default_factory=lambda: np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [0, 5],
            [1, 4],
        ],
        dtype=int,
    ))
    outage_edge: int = 2
    target_reserve: float = 0.35

    @property
    def bus_count(self) -> int:
        return len(self.demand)

    @property
    def dimensions(self) -> int:
        return self.bus_count * 2

    def project(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(self.bus_count, 2)
        thermal = np.clip(x[:, 0], self.thermal_min, self.thermal_max)
        batt = np.clip(x[:, 1], -self.battery_max, self.battery_max)
        return np.column_stack([thermal, batt]).reshape(-1)

    def random_geometry(self, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
        thermal = rng.uniform(self.thermal_min, self.thermal_max)
        batt = rng.uniform(-self.battery_max, self.battery_max)
        thermal += rng.normal(scale=0.15 * min(scale, 2.0), size=self.bus_count)
        batt += rng.normal(scale=0.05 * min(scale, 2.0), size=self.bus_count)
        return self.project(np.column_stack([thermal, batt]).reshape(-1))

    def random_direction(self, rng: np.random.Generator) -> np.ndarray:
        direction = rng.normal(size=self.dimensions)
        return direction / max(np.linalg.norm(direction), 1e-9)

    def thermal_noise(self, rng: np.random.Generator, scale: float) -> np.ndarray:
        weights = np.tile(np.array([0.18, 0.08], dtype=float), self.bus_count)
        return rng.normal(size=self.dimensions) * weights * scale

    def radial_signature(self, x: np.ndarray) -> np.ndarray:
        dispatch = self.project(x).reshape(self.bus_count, 2)
        return dispatch[:, 0] + dispatch[:, 1]

    def vector_to_positions(self, x: np.ndarray) -> np.ndarray:
        return self.project(x).reshape(self.bus_count, 2)

    def injections(self, x: np.ndarray) -> np.ndarray:
        dispatch = self.project(x).reshape(self.bus_count, 2)
        thermal = dispatch[:, 0]
        batt = dispatch[:, 1]
        return thermal + batt + self.solar - self.demand

    def _laplacian(self, with_outage: bool = True) -> np.ndarray:
        lap = np.zeros((self.bus_count, self.bus_count), dtype=float)
        for idx, (i, j) in enumerate(self.edges):
            if with_outage and idx == self.outage_edge:
                continue
            lap[i, i] += 1.0
            lap[j, j] += 1.0
            lap[i, j] -= 1.0
            lap[j, i] -= 1.0
        return lap

    def line_flows(self, x: np.ndarray, with_outage: bool = True) -> np.ndarray:
        inj = self.injections(x)
        lap = self._laplacian(with_outage=with_outage)
        reduced = lap[:-1, :-1]
        rhs = inj[:-1]
        theta = np.zeros(self.bus_count, dtype=float)
        theta[:-1] = np.linalg.solve(reduced + 1e-6 * np.eye(self.bus_count - 1), rhs)
        flows = []
        for idx, (i, j) in enumerate(self.edges):
            if with_outage and idx == self.outage_edge:
                flows.append(np.nan)
            else:
                flows.append(theta[i] - theta[j])
        return np.asarray(flows, dtype=float)

    def reserve_margin(self, x: np.ndarray) -> float:
        dispatch = self.project(x).reshape(self.bus_count, 2)
        thermal = dispatch[:, 0]
        available = np.sum(self.thermal_max - thermal + np.maximum(0.0, self.battery_max - np.abs(dispatch[:, 1])))
        return float(available)

    def energy_terms(self, x: np.ndarray) -> dict:
        dispatch = self.project(x).reshape(self.bus_count, 2)
        thermal = dispatch[:, 0]
        batt = dispatch[:, 1]
        inj = self.injections(x)
        balance_penalty = 180.0 * float(np.sum(inj) ** 2)
        line_flows = self.line_flows(x, with_outage=True)
        overload = np.nan_to_num(np.maximum(0.0, np.abs(line_flows) - self.capacities) ** 2)
        overload_penalty = 110.0 * float(np.nansum(overload))
        reserve = self.reserve_margin(x)
        reserve_penalty = 80.0 * float(max(0.0, self.target_reserve - reserve) ** 2)
        curtailment_penalty = 18.0 * float(np.sum(np.maximum(0.0, -batt)))
        thermal_cost = float(np.sum(self.thermal_cost * thermal))
        batt_cost = float(np.sum(self.batt_cost * np.abs(batt)))
        action_like = thermal_cost + batt_cost + balance_penalty + overload_penalty + reserve_penalty + curtailment_penalty
        return {
            "action_like": float(action_like),
            "thermal_cost": thermal_cost,
            "battery_cost": batt_cost,
            "balance_penalty": balance_penalty,
            "overload_penalty": overload_penalty,
            "reserve_penalty": reserve_penalty,
            "curtailment_penalty": curtailment_penalty,
            "power_balance": float(np.sum(inj)),
            "max_flow_utilization": float(np.nanmax(np.abs(line_flows) / self.capacities)),
            "reserve_margin": reserve,
            "line_flows": np.nan_to_num(line_flows, nan=0.0).tolist(),
        }

    def energy(self, x: np.ndarray) -> float:
        return self.energy_terms(x)["action_like"]

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = self.project(x)
        eps = np.full(self.dimensions, 2e-4, dtype=float)
        grad = np.zeros(self.dimensions, dtype=float)
        for idx in range(self.dimensions):
            delta = np.zeros(self.dimensions, dtype=float)
            delta[idx] = eps[idx]
            grad[idx] = (self.energy(self.project(x + delta)) - self.energy(self.project(x - delta))) / (2.0 * eps[idx])
        return grad


def make_optimizer(problem: SmartGridDispatchProblem) -> GONMOptimizer:
    return GONMOptimizer(
        problem=problem,
        noise=CoolingNoise(start_std=0.45, mid_std=0.18, late_std=0.05, final_std=0.0),
        config=GONMConfig(
            population=52,
            phase1_keep=8,
            phase2_steps=110,
            phase3_steps=44,
            box_scale=1.3,
            thermal_scale=0.12,
            momentum=0.89,
            step_size=0.050,
            terminal_radii=(0.22, 0.12, 0.06, 0.03),
        ),
    )


def baseline_merit_redispatch(problem: SmartGridDispatchProblem) -> np.ndarray:
    thermal = problem.thermal_min.copy()
    batt = np.zeros(problem.bus_count, dtype=float)
    need = float(np.sum(problem.demand - problem.solar - batt - thermal))
    merit = np.argsort(problem.thermal_cost)
    for idx in merit:
        room = problem.thermal_max[idx] - thermal[idx]
        add = min(room, max(0.0, need))
        thermal[idx] += add
        need -= add
    if need > 1e-9:
        batt = np.clip(problem.battery_max * 0.55, -problem.battery_max, problem.battery_max)
    return np.column_stack([thermal, batt]).reshape(-1)


def render_result(problem: SmartGridDispatchProblem, baseline_x: np.ndarray, gonm_result) -> plt.Figure:
    baseline_terms = problem.energy_terms(baseline_x)
    tuned_x = gonm_result.final_best_positions.reshape(-1)
    tuned_terms = problem.energy_terms(tuned_x)
    baseline_dispatch = problem.project(baseline_x).reshape(problem.bus_count, 2)
    tuned_dispatch = problem.project(tuned_x).reshape(problem.bus_count, 2)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1.1])
    ax_dispatch = fig.add_subplot(gs[0, 0])
    ax_flows = fig.add_subplot(gs[0, 1])
    ax_trace = fig.add_subplot(gs[1, 0])
    ax_text = fig.add_subplot(gs[1, 1])

    buses = np.arange(problem.bus_count)
    width = 0.36
    ax_dispatch.bar(buses - width / 2, baseline_dispatch[:, 0] + baseline_dispatch[:, 1], width=width, color="#f87171", label="baseline")
    ax_dispatch.bar(buses + width / 2, tuned_dispatch[:, 0] + tuned_dispatch[:, 1], width=width, color="#60a5fa", label="GONM")
    ax_dispatch.plot(buses, problem.demand - problem.solar, color="#111827", marker="o", linewidth=1.6, label="net demand")
    ax_dispatch.set_title("Despacho por nó após contingência")
    ax_dispatch.set_xlabel("nó")
    ax_dispatch.set_ylabel("potência")
    ax_dispatch.grid(True, axis="y", linestyle=":", alpha=0.45)
    ax_dispatch.legend(loc="best")

    edge_idx = np.arange(len(problem.edges))
    baseline_flows = np.asarray(baseline_terms["line_flows"], dtype=float)
    tuned_flows = np.asarray(tuned_terms["line_flows"], dtype=float)
    ax_flows.bar(edge_idx - width / 2, np.abs(baseline_flows), width=width, color="#fca5a5", label="baseline")
    ax_flows.bar(edge_idx + width / 2, np.abs(tuned_flows), width=width, color="#93c5fd", label="GONM")
    ax_flows.plot(edge_idx, problem.capacities, color="#111827", linewidth=1.5, marker="o", label="capacity")
    ax_flows.set_title("Utilização de linhas remanescentes")
    ax_flows.set_xlabel("linha")
    ax_flows.set_ylabel("|fluxo|")
    ax_flows.grid(True, axis="y", linestyle=":", alpha=0.45)
    ax_flows.legend(loc="best")

    ax_trace.plot(gonm_result.trace_true_energy, color="#2563eb", linewidth=2.0, label="GONM objective")
    ax_trace.axhline(baseline_terms["action_like"], color="#dc2626", linewidth=1.8, linestyle="--", label="baseline objective")
    ax_trace.set_title("Despacho econômico sob contingência")
    ax_trace.set_xlabel("iteracao")
    ax_trace.set_ylabel("J[dispatch]")
    ax_trace.grid(True, linestyle=":", alpha=0.45)
    ax_trace.legend(loc="best")

    gain = baseline_terms["action_like"] - tuned_terms["action_like"]
    ax_text.axis("off")
    ax_text.text(
        0.0,
        1.0,
        "\n".join(
            [
                "GONM | smart-grid economic dispatch",
                "",
                f"baseline objective = {baseline_terms['action_like']:.6f}",
                f"GONM objective = {tuned_terms['action_like']:.6f}",
                f"ganho GONM vs baseline = {gain:.6f}",
                "",
                f"baseline max flow utilization = {baseline_terms['max_flow_utilization']:.4f}",
                f"GONM max flow utilization = {tuned_terms['max_flow_utilization']:.4f}",
                "",
                f"baseline reserve margin = {baseline_terms['reserve_margin']:.4f}",
                f"GONM reserve margin = {tuned_terms['reserve_margin']:.4f}",
                "",
                f"baseline power balance = {baseline_terms['power_balance']:.4e}",
                f"GONM power balance = {tuned_terms['power_balance']:.4e}",
                "",
                "Observacao:",
                "a linha em falha foi removida e o GONM faz o",
                "redispatch para um novo equilíbrio seguro.",
            ]
        ),
        ha="left",
        va="top",
        fontsize=10.5,
        family="monospace",
    )

    fig.suptitle("GONM | Smart Grid Redispatch Under Line Outage", fontsize=16, y=0.98)
    fig.tight_layout()
    return fig


def write_summary(problem: SmartGridDispatchProblem, baseline_x: np.ndarray, gonm_result) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    baseline_terms = problem.energy_terms(baseline_x)
    tuned_x = gonm_result.final_best_positions.reshape(-1)
    tuned_terms = problem.energy_terms(tuned_x)
    summary = {
        "seed": SEED,
        "problem": "smart_grid_dispatch",
        "interpretation": "Economic redispatch under a transmission-line outage in a reduced smart-grid network.",
        "outage_edge_index": int(problem.outage_edge),
        "baseline": {
            "dispatch": problem.project(baseline_x).reshape(problem.bus_count, 2).tolist(),
            "objective": float(baseline_terms["action_like"]),
            "terms": baseline_terms,
        },
        "gonm": {
            "phase1_best_energy": float(gonm_result.phase1_best_energy),
            "dispatch": problem.project(tuned_x).reshape(problem.bus_count, 2).tolist(),
            "final_energy": float(tuned_terms["action_like"]),
            "terms": tuned_terms,
            "evals": int(gonm_result.evals),
            "runtime_ms": float(gonm_result.runtime_ms),
        },
        "comparison": {
            "gonm_minus_baseline": float(tuned_terms["action_like"] - baseline_terms["action_like"]),
        },
    }
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = f"""# GONM | Smart Grid Economic Dispatch

This simulation treats GONM as a redispatch optimizer after a transmission-line outage in a reduced smart-grid network.

## Recorded outcome

- baseline objective: `{baseline_terms["action_like"]:.6f}`
- GONM objective: `{tuned_terms["action_like"]:.6f}`
- GONM gain versus baseline: `{baseline_terms["action_like"] - tuned_terms["action_like"]:.6f}`
- baseline max flow utilization: `{baseline_terms["max_flow_utilization"]:.6f}`
- GONM max flow utilization: `{tuned_terms["max_flow_utilization"]:.6f}`

## Interpretation

This is not a production SCADA or full AC optimal-power-flow solver. The narrower claim is still useful: once a line contingency is imposed, the layered GONM search can rebalance generation and storage toward a lower-cost, lower-stress operating point than a simple merit-order redispatch.
"""
    SUMMARY_MD_PATH.write_text(md, encoding="utf-8")


def main() -> None:
    problem = SmartGridDispatchProblem()
    baseline_x = baseline_merit_redispatch(problem)
    optimizer = make_optimizer(problem)
    gonm_result = optimizer.optimize(seed=SEED)

    fig = render_result(problem, baseline_x, gonm_result)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    if "--no-show" not in sys.argv and "--save" not in sys.argv:
        plt.show()
    plt.close(fig)

    write_summary(problem, baseline_x, gonm_result)


if __name__ == "__main__":
    main()
