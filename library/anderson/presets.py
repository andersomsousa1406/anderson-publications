from __future__ import annotations

from dataclasses import dataclass

from anderson.noise import CoolingNoise, FixedNoise
from anderson.optimizers.gonm import GONMConfig, GONMOptimizer, GONMResult
from anderson.problems.molecular import MolecularCluster


@dataclass(slots=True)
class AndersonSimulation:
    name: str
    problem: MolecularCluster
    optimizer: GONMOptimizer
    default_seed: int

    def run(self, seed: int | None = None) -> GONMResult:
        return self.optimizer.optimize(seed=self.default_seed if seed is None else seed)


def lj12_fixed(seed: int = 17) -> AndersonSimulation:
    problem = MolecularCluster(atom_count=12)
    optimizer = GONMOptimizer(
        problem=problem,
        noise=FixedNoise(0.08),
        config=GONMConfig(),
    )
    return AndersonSimulation("lj12_fixed", problem, optimizer, seed)


def lj12_cooling(seed: int = 17) -> AndersonSimulation:
    problem = MolecularCluster(atom_count=12)
    optimizer = GONMOptimizer(
        problem=problem,
        noise=CoolingNoise(0.08),
        config=GONMConfig(),
    )
    return AndersonSimulation("lj12_cooling", problem, optimizer, seed)


def lj38_default(seed: int = 38) -> AndersonSimulation:
    problem = MolecularCluster(atom_count=38, bounds=(-2.5, 2.5))
    optimizer = GONMOptimizer(
        problem=problem,
        noise=FixedNoise(0.05),
        config=GONMConfig(
            population=72,
            phase1_keep=10,
            phase2_steps=120,
            phase3_steps=48,
            box_scale=4.6,
            thermal_scale=0.010,
            momentum=0.86,
            step_size=0.018,
            terminal_radii=(0.12, 0.08, 0.05, 0.03),
        ),
    )
    return AndersonSimulation("lj38_default", problem, optimizer, seed)


def lj38_budget200k(seed: int = 38) -> AndersonSimulation:
    problem = MolecularCluster(atom_count=38, bounds=(-2.5, 2.5))
    optimizer = GONMOptimizer(
        problem=problem,
        noise=FixedNoise(0.05),
        config=GONMConfig(
            population=120,
            phase1_keep=14,
            phase2_steps=220,
            phase3_steps=220,
            box_scale=4.6,
            thermal_scale=0.010,
            momentum=0.86,
            step_size=0.018,
            terminal_radii=(0.12, 0.08, 0.05, 0.03),
        ),
    )
    return AndersonSimulation("lj38_budget200k", problem, optimizer, seed)
