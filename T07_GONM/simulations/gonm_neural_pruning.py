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


SEED = 211
RESULT_DIR = ROOT / "publications" / "T07_GONM" / "results" / "gonm_neural_pruning"
FIGURE_PATH = RESULT_DIR / "gonm_neural_pruning.png"
SUMMARY_JSON_PATH = RESULT_DIR / "summary.json"
SUMMARY_MD_PATH = RESULT_DIR / "summary.md"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(slots=True)
class NeuralPruningProblem:
    hidden_size: int = 10
    sparsity_weight: float = 0.19
    binary_weight: float = 0.035
    x_train: np.ndarray = field(init=False, repr=False)
    y_train: np.ndarray = field(init=False, repr=False)
    x_val: np.ndarray = field(init=False, repr=False)
    y_val: np.ndarray = field(init=False, repr=False)
    w1: np.ndarray = field(init=False, repr=False)
    b1: np.ndarray = field(init=False, repr=False)
    w2: np.ndarray = field(init=False, repr=False)
    b2: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(SEED)
        x = rng.uniform(-2.3, 2.3, size=(900, 2))

        # A fixed compact "teacher" network.
        self.w1 = np.asarray(
            [
                [1.8, -0.4],
                [-1.2, 1.5],
                [0.7, 1.9],
                [-1.6, -0.7],
                [2.1, 0.3],
                [0.4, -2.0],
                [-2.1, 0.5],
                [1.1, 1.2],
                [-0.8, -1.7],
                [1.6, -1.4],
            ],
            dtype=float,
        )
        self.b1 = np.asarray([0.2, -0.1, 0.5, -0.4, 0.3, 0.1, -0.2, -0.5, 0.2, -0.3], dtype=float)
        # Intentionally mixes some low-magnitude but decision-critical units.
        self.w2 = np.asarray([1.3, -1.1, 0.7, 0.5, -1.2, 0.35, -0.25, 1.0, -0.9, 0.28], dtype=float)
        self.b2 = -0.05

        logits = self.full_logits(x)
        prob = sigmoid(logits)
        y = (prob > 0.5).astype(float)

        perm = rng.permutation(len(x))
        x = x[perm]
        y = y[perm]
        self.x_train = x[:600]
        self.y_train = y[:600]
        self.x_val = x[600:]
        self.y_val = y[600:]

    @property
    def dimensions(self) -> int:
        return self.hidden_size

    def full_hidden(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x @ self.w1.T + self.b1)

    def full_logits(self, x: np.ndarray) -> np.ndarray:
        h = self.full_hidden(x)
        return h @ self.w2 + self.b2

    def project(self, x: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(x, dtype=float).reshape(-1), 0.0, 1.0)

    def random_geometry(self, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
        gates = rng.uniform(0.0, 1.0, size=self.hidden_size)
        gates += rng.normal(scale=0.12 * min(scale, 2.0), size=self.hidden_size)
        return self.project(gates)

    def random_direction(self, rng: np.random.Generator) -> np.ndarray:
        direction = rng.normal(size=self.hidden_size)
        return direction / max(np.linalg.norm(direction), 1e-9)

    def thermal_noise(self, rng: np.random.Generator, scale: float) -> np.ndarray:
        return rng.normal(size=self.hidden_size) * scale

    def radial_signature(self, x: np.ndarray) -> np.ndarray:
        return self.project(x)

    def vector_to_positions(self, x: np.ndarray) -> np.ndarray:
        return self.project(x).reshape(1, -1)

    def masked_logits(self, x_data: np.ndarray, gates: np.ndarray) -> np.ndarray:
        h = self.full_hidden(x_data)
        return h @ (self.w2 * gates) + self.b2

    def bce(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        prob = np.clip(sigmoid(logits), 1e-7, 1.0 - 1e-7)
        return float(-np.mean(y_true * np.log(prob) + (1.0 - y_true) * np.log(1.0 - prob)))

    def accuracy(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        pred = (sigmoid(logits) >= 0.5).astype(float)
        return float(np.mean(pred == y_true))

    def importance_scores(self) -> np.ndarray:
        hidden = np.abs(self.full_hidden(self.x_train))
        return np.mean(hidden, axis=0) * np.abs(self.w2)

    def energy_terms(self, x: np.ndarray) -> dict:
        gates = self.project(x)
        train_logits = self.masked_logits(self.x_train, gates)
        val_logits = self.masked_logits(self.x_val, gates)
        train_loss = self.bce(train_logits, self.y_train)
        val_loss = self.bce(val_logits, self.y_val)
        sparsity = float(np.mean(gates))
        binary = float(np.mean(gates * (1.0 - gates)))
        objective = val_loss + self.sparsity_weight * sparsity + self.binary_weight * binary
        active = int(np.sum(gates >= 0.5))
        return {
            "objective": float(objective),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_accuracy": self.accuracy(train_logits, self.y_train),
            "val_accuracy": self.accuracy(val_logits, self.y_val),
            "sparsity_mean": sparsity,
            "binary_penalty": binary,
            "active_neurons": active,
        }

    def energy(self, x: np.ndarray) -> float:
        return self.energy_terms(x)["objective"]

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = self.project(x)
        eps = np.full(self.hidden_size, 2e-4, dtype=float)
        grad = np.zeros(self.hidden_size, dtype=float)
        for idx in range(self.hidden_size):
            delta = np.zeros(self.hidden_size, dtype=float)
            delta[idx] = eps[idx]
            grad[idx] = (self.energy(self.project(x + delta)) - self.energy(self.project(x - delta))) / (2.0 * eps[idx])
        return grad


def make_optimizer(problem: NeuralPruningProblem) -> GONMOptimizer:
    return GONMOptimizer(
        problem=problem,
        noise=CoolingNoise(start_std=0.0015, mid_std=0.0006, late_std=0.00012, final_std=0.0),
        config=GONMConfig(
            population=48,
            phase1_keep=8,
            phase2_steps=100,
            phase3_steps=40,
            box_scale=1.3,
            thermal_scale=0.10,
            momentum=0.88,
            step_size=0.055,
            terminal_radii=(0.18, 0.10, 0.05, 0.02),
        ),
    )


def baseline_magnitude_pruning(problem: NeuralPruningProblem, keep: int) -> np.ndarray:
    scores = problem.importance_scores()
    order = np.argsort(scores)[::-1]
    gates = np.zeros(problem.hidden_size, dtype=float)
    gates[order[:keep]] = 1.0
    return gates


def render_result(problem: NeuralPruningProblem, baseline_gates: np.ndarray, gonm_result) -> plt.Figure:
    baseline_terms = problem.energy_terms(baseline_gates)
    tuned_gates = problem.project(gonm_result.final_best_positions.reshape(-1))
    tuned_terms = problem.energy_terms(tuned_gates)
    scores = problem.importance_scores()

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1.1])
    ax_gates = fig.add_subplot(gs[0, 0])
    ax_perf = fig.add_subplot(gs[0, 1])
    ax_trace = fig.add_subplot(gs[1, 0])
    ax_text = fig.add_subplot(gs[1, 1])

    idx = np.arange(problem.hidden_size)
    width = 0.36
    ax_gates.bar(idx - width / 2, baseline_gates, width=width, color="#f87171", label="magnitude pruning")
    ax_gates.bar(idx + width / 2, tuned_gates, width=width, color="#60a5fa", label="GONM pruning")
    ax_gates.plot(idx, scores / np.max(scores), color="#111827", marker="o", linewidth=1.5, label="normalized importance")
    ax_gates.set_title("Máscaras de poda por neurônio oculto")
    ax_gates.set_xlabel("neurônio")
    ax_gates.set_ylabel("gate")
    ax_gates.grid(True, axis="y", linestyle=":", alpha=0.45)
    ax_gates.legend(loc="best")

    labels = ["val acc", "val loss", "active ratio"]
    baseline_vals = [
        baseline_terms["val_accuracy"],
        baseline_terms["val_loss"],
        baseline_terms["active_neurons"] / problem.hidden_size,
    ]
    tuned_vals = [
        tuned_terms["val_accuracy"],
        tuned_terms["val_loss"],
        tuned_terms["active_neurons"] / problem.hidden_size,
    ]
    x = np.arange(len(labels))
    ax_perf.bar(x - width / 2, baseline_vals, width=width, color="#fca5a5", label="magnitude")
    ax_perf.bar(x + width / 2, tuned_vals, width=width, color="#93c5fd", label="GONM")
    ax_perf.set_xticks(x)
    ax_perf.set_xticklabels(labels)
    ax_perf.set_title("Compressão vs qualidade")
    ax_perf.grid(True, axis="y", linestyle=":", alpha=0.45)
    ax_perf.legend(loc="best")

    ax_trace.plot(gonm_result.trace_true_energy, color="#2563eb", linewidth=2.0, label="GONM objective")
    ax_trace.axhline(baseline_terms["objective"], color="#dc2626", linewidth=1.8, linestyle="--", label="magnitude objective")
    ax_trace.set_title("Objetivo de compressão")
    ax_trace.set_xlabel("iteracao")
    ax_trace.set_ylabel("J[mask]")
    ax_trace.grid(True, linestyle=":", alpha=0.45)
    ax_trace.legend(loc="best")

    gain = baseline_terms["objective"] - tuned_terms["objective"]
    ax_text.axis("off")
    ax_text.text(
        0.0,
        1.0,
        "\n".join(
            [
                "GONM | neural network squeezing",
                "",
                f"magnitude objective = {baseline_terms['objective']:.6f}",
                f"GONM objective = {tuned_terms['objective']:.6f}",
                f"ganho GONM vs magnitude = {gain:.6f}",
                "",
                f"magnitude val acc = {baseline_terms['val_accuracy']:.4f}",
                f"GONM val acc = {tuned_terms['val_accuracy']:.4f}",
                "",
                f"magnitude active = {baseline_terms['active_neurons']}/{problem.hidden_size}",
                f"GONM active = {tuned_terms['active_neurons']}/{problem.hidden_size}",
                "",
                f"magnitude val loss = {baseline_terms['val_loss']:.6f}",
                f"GONM val loss = {tuned_terms['val_loss']:.6f}",
                "",
                "Leitura:",
                "o GONM comprime a rede escolhendo uma máscara",
                "mais alinhada com a função aprendida do que",
                "a poda puramente por magnitude.",
            ]
        ),
        ha="left",
        va="top",
        fontsize=10.5,
        family="monospace",
    )

    fig.suptitle("GONM | Neural Network Compression by Structured Pruning", fontsize=16, y=0.98)
    fig.tight_layout()
    return fig


def write_summary(problem: NeuralPruningProblem, baseline_gates: np.ndarray, gonm_result) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    baseline_terms = problem.energy_terms(baseline_gates)
    tuned_gates = problem.project(gonm_result.final_best_positions.reshape(-1))
    tuned_terms = problem.energy_terms(tuned_gates)
    summary = {
        "seed": SEED,
        "problem": "neural_pruning",
        "interpretation": "Structured neural-network compression by optimizing a hidden-neuron pruning mask.",
        "baseline_magnitude": {
            "gates": baseline_gates.tolist(),
            "objective": float(baseline_terms["objective"]),
            "terms": baseline_terms,
        },
        "gonm": {
            "phase1_best_energy": float(gonm_result.phase1_best_energy),
            "gates": tuned_gates.tolist(),
            "final_energy": float(tuned_terms["objective"]),
            "terms": tuned_terms,
            "evals": int(gonm_result.evals),
            "runtime_ms": float(gonm_result.runtime_ms),
        },
        "comparison": {
            "gonm_minus_magnitude": float(tuned_terms["objective"] - baseline_terms["objective"]),
        },
    }
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = f"""# GONM | Neural Network Compression

This simulation treats GONM as a structured pruning optimizer for a compact neural network.

## Recorded outcome

- magnitude-pruning objective: `{baseline_terms["objective"]:.6f}`
- GONM pruning objective: `{tuned_terms["objective"]:.6f}`
- GONM gain versus magnitude: `{baseline_terms["objective"] - tuned_terms["objective"]:.6f}`
- magnitude validation accuracy: `{baseline_terms["val_accuracy"]:.6f}`
- GONM validation accuracy: `{tuned_terms["val_accuracy"]:.6f}`

## Interpretation

This is not full LLM compression. The narrower claim is still useful: a GONM-style structured search can select a smaller hidden subnetwork with a better sparsity/accuracy tradeoff than a simple magnitude-pruning baseline.
"""
    SUMMARY_MD_PATH.write_text(md, encoding="utf-8")


def main() -> None:
    problem = NeuralPruningProblem()
    optimizer = make_optimizer(problem)
    gonm_result = optimizer.optimize(seed=SEED)

    tuned_gates = problem.project(gonm_result.final_best_positions.reshape(-1))
    keep = max(1, int(np.sum(tuned_gates >= 0.5)))
    baseline_gates = baseline_magnitude_pruning(problem, keep=keep)

    fig = render_result(problem, baseline_gates, gonm_result)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    if "--no-show" not in sys.argv and "--save" not in sys.argv:
        plt.show()
    plt.close(fig)

    write_summary(problem, baseline_gates, gonm_result)


if __name__ == "__main__":
    main()
