"""Trajectory-level filters inspired by the MQLM line."""

from __future__ import annotations

import math
from typing import Iterable


def multiplicative_path_score(values: Iterable[float], eps: float = 1.0e-12) -> float:
    """Return a multiplicative score for a positive trajectory."""
    total = 0.0
    count = 0
    for value in values:
        total += math.log(max(float(value), eps))
        count += 1
    if count == 0:
        raise ValueError("values must be non-empty")
    return math.exp(total / count)


def log_mean_score(values: Iterable[float], eps: float = 1.0e-12) -> float:
    """Return the arithmetic mean in log space."""
    total = 0.0
    count = 0
    for value in values:
        total += math.log(max(float(value), eps))
        count += 1
    if count == 0:
        raise ValueError("values must be non-empty")
    return total / count
