from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from anderson.noise import CoolingNoise, FixedNoise
from anderson.problems.molecular import MolecularCluster


@dataclass(slots=True)
class GONMConfig:
    population: int = 64
    phase1_keep: int = 8
    phase2_steps: int = 140
    phase3_steps: int = 60
    box_scale: float = 2.8
    thermal_scale: float = 0.012
    momentum: float = 0.84
    step_size: float = 0.026
    terminal_radii: tuple[float, ...] = (0.18, 0.11, 0.07, 0.04)


@dataclass(slots=True)
class GONMResult:
    seed: int
    evals: int
    runtime_ms: float
    initial_best_energy: float
    phase1_best_energy: float
    final_true_energy: float
    energy_gain_vs_phase1: float
    phase1_best_positions: np.ndarray
    final_best_positions: np.ndarray
    trace_true_energy: list[float]


class _NoisyOracle:
    def __init__(self, problem: MolecularCluster, noise, rng: np.random.Generator):
        self.problem = problem
        self.noise = noise
        self.rng = rng
        self.evals = 0

    def true(self, x: np.ndarray) -> float:
        return self.problem.energy(x)

    def averaged(self, x: np.ndarray, n_samples: int, stage: str, step: int, total_steps: int) -> float:
        vals = []
        scale = self.noise.scale(stage, step, total_steps)
        for _ in range(n_samples):
            self.evals += 1
            vals.append(self.true(x) + float(self.rng.normal(scale=scale)))
        return float(np.mean(vals))


class GONMOptimizer:
    def __init__(self, problem: MolecularCluster, noise: FixedNoise | CoolingNoise, config: GONMConfig | None = None):
        self.problem = problem
        self.noise = noise
        self.config = config or GONMConfig()

    def _sample_thermal_noise(self, rng: np.random.Generator, scale: float) -> np.ndarray:
        if hasattr(self.problem, "thermal_noise"):
            return np.asarray(self.problem.thermal_noise(rng, scale), dtype=float).reshape(-1)
        thermal = rng.normal(size=(self.problem.atom_count, 3))
        thermal -= np.mean(thermal, axis=0, keepdims=True)
        thermal = thermal.reshape(-1)
        return thermal * scale

    def _sample_direction(self, rng: np.random.Generator) -> np.ndarray:
        if hasattr(self.problem, "random_direction"):
            direction = np.asarray(self.problem.random_direction(rng), dtype=float).reshape(-1)
            return direction / max(np.linalg.norm(direction), 1e-9)
        direction = rng.normal(size=(self.problem.atom_count, 3))
        direction -= np.mean(direction, axis=0, keepdims=True)
        direction = direction.reshape(-1)
        direction /= max(np.linalg.norm(direction), 1e-9)
        return direction

    def _cluster_candidates(self, candidates: list[np.ndarray], tol: float = 0.55) -> list[np.ndarray]:
        clusters: list[np.ndarray] = []
        signatures: list[np.ndarray] = []
        for x in candidates:
            sig = self.problem.radial_signature(x)
            if any(np.linalg.norm(sig - other) <= tol for other in signatures):
                continue
            signatures.append(sig)
            clusters.append(x)
        return clusters

    def _local_refine(self, oracle: _NoisyOracle, x0: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, list[float]]:
        cfg = self.config
        x = x0.copy()
        velocity = np.zeros_like(x)
        best_x = x.copy()
        best_score = oracle.averaged(x, 5, "phase2", 0, cfg.phase2_steps)
        trace = [oracle.true(x)]

        for t in range(cfg.phase2_steps):
            grad = self.problem.gradient(x)
            grad_norm = np.linalg.norm(grad) + 1e-9
            thermal = self._sample_thermal_noise(rng, cfg.thermal_scale * np.exp(-t / 45.0))

            velocity = cfg.momentum * velocity - cfg.step_size * grad / grad_norm + thermal
            proposal = self.problem.project(x + velocity)
            old_score = oracle.averaged(x, 3, "phase2", t, cfg.phase2_steps)
            new_score = oracle.averaged(proposal, 3, "phase2", t, cfg.phase2_steps)
            temp = 0.12 + 0.02 * np.exp(-t / 35.0)
            if new_score < old_score or rng.random() < np.exp(-(new_score - old_score) / temp):
                x = proposal
                if new_score < best_score:
                    best_score = new_score
                    best_x = x.copy()
            else:
                velocity *= 0.45
            trace.append(oracle.true(best_x))
        return best_x, trace

    def _terminal_search(self, oracle: _NoisyOracle, x0: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, list[float]]:
        cfg = self.config
        x = x0.copy()
        best_x = x.copy()
        best_score = oracle.averaged(x, 7, "phase3", 0, cfg.phase3_steps)
        trace = [oracle.true(x)]
        total_step = 0

        for radius in cfg.terminal_radii:
            for _ in range(max(1, cfg.phase3_steps // len(cfg.terminal_radii))):
                local_best = x
                local_score = best_score
            for _ in range(9):
                direction = self._sample_direction(rng)
                proposal = self.problem.project(x + radius * direction)
                score = oracle.averaged(proposal, 5, "phase3", total_step, cfg.phase3_steps)
                if score < local_score:
                        local_score = score
                        local_best = proposal
                if local_score < best_score:
                    x = local_best
                    best_x = local_best.copy()
                    best_score = local_score
                trace.append(oracle.true(best_x))
                total_step += 1

        return best_x, trace

    def optimize(self, seed: int = 17) -> GONMResult:
        cfg = self.config
        rng = np.random.default_rng(seed)
        oracle = _NoisyOracle(self.problem, self.noise, rng)
        start = perf_counter()

        initial_population = [
            self.problem.random_geometry(rng, cfg.box_scale * rng.uniform(0.75, 1.25))
            for _ in range(cfg.population)
        ]
        scored = [(oracle.averaged(x, 3, "phase1", 0, 1), x) for x in initial_population]
        scored.sort(key=lambda item: item[0])

        elites = [x for _, x in scored[: cfg.phase1_keep * 2]]
        clustered = self._cluster_candidates(elites)
        if len(clustered) < cfg.phase1_keep:
            clustered = elites[: cfg.phase1_keep]
        else:
            clustered = clustered[: cfg.phase1_keep]

        phase1_best = min(clustered, key=oracle.true)
        phase1_best_energy = oracle.true(phase1_best)
        initial_best = min(initial_population, key=oracle.true)
        initial_best_energy = oracle.true(initial_best)

        refined_runs = []
        for x in clustered:
            refined_x, phase2_trace = self._local_refine(oracle, x, rng)
            terminal_x, phase3_trace = self._terminal_search(oracle, refined_x, rng)
            refined_runs.append((oracle.true(terminal_x), terminal_x, phase2_trace + phase3_trace))

        refined_runs.sort(key=lambda item: item[0])
        best_energy, best_final, best_trace = refined_runs[0]
        runtime_ms = (perf_counter() - start) * 1000.0

        return GONMResult(
            seed=seed,
            evals=oracle.evals,
            runtime_ms=runtime_ms,
            initial_best_energy=initial_best_energy,
            phase1_best_energy=phase1_best_energy,
            final_true_energy=best_energy,
            energy_gain_vs_phase1=phase1_best_energy - best_energy,
            phase1_best_positions=self.problem.vector_to_positions(phase1_best),
            final_best_positions=self.problem.vector_to_positions(best_final),
            trace_true_energy=best_trace,
        )
