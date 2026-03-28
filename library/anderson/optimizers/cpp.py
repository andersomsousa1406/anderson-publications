"""Contractive local refinement operators derived from the CPP line."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CPPConfig:
    """Configuration for a simple contractive local update."""

    step_size: float = 0.2
    contraction: float = 0.8


def contractive_update(x: np.ndarray, direction: np.ndarray, config: CPPConfig | None = None) -> np.ndarray:
    """Return a damped update step inspired by CPP."""
    cfg = config or CPPConfig()
    x_arr = np.asarray(x, dtype=float)
    d_arr = np.asarray(direction, dtype=float)
    if x_arr.shape != d_arr.shape:
        raise ValueError("x and direction must have the same shape")
    return x_arr + cfg.step_size * cfg.contraction * d_arr
