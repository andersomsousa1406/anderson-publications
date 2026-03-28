"""Diagnostic helpers for structural walls, budgets, and noisy resolution."""

from __future__ import annotations

import math


def structural_wall_coefficient(best_value: float, dimension: int) -> float:
    """Return the normalized wall coefficient used in the GONM manuscript."""
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    return float(best_value) / math.sqrt(float(dimension))


def budget_density(total_budget: int, dimension: int) -> float:
    """Return the effective budget per coordinate."""
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    return float(total_budget) / float(dimension)


def terminal_noise_scale(noise_std: float, replications: int) -> float:
    """Return the repeated-average noise scale sigma/sqrt(m)."""
    if replications <= 0:
        raise ValueError("replications must be positive")
    return float(noise_std) / math.sqrt(float(replications))
