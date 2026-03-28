"""Analytic benchmark functions used throughout the archive."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class BenchmarkFunction:
    """Simple callable wrapper for archive benchmark functions."""

    name: str
    dimension: int
    fn: Callable[[np.ndarray], float]

    def __call__(self, x: np.ndarray) -> float:
        arr = np.asarray(x, dtype=float)
        if arr.shape != (self.dimension,):
            raise ValueError(f"{self.name} expects shape {(self.dimension,)}, got {arr.shape}")
        return float(self.fn(arr))


def ackley(dimension: int, a: float = 20.0, b: float = 0.2, c: float = 2.0 * np.pi) -> BenchmarkFunction:
    """Return the standard Ackley benchmark in the given dimension."""

    def _fn(x: np.ndarray) -> float:
        sq = np.mean(x * x)
        cs = np.mean(np.cos(c * x))
        return float(-a * np.exp(-b * np.sqrt(sq)) - np.exp(cs) + a + np.e)

    return BenchmarkFunction(name=f"ackley_{dimension}d", dimension=dimension, fn=_fn)


def rastrigin(dimension: int, amplitude: float = 10.0) -> BenchmarkFunction:
    """Return the standard Rastrigin benchmark."""

    def _fn(x: np.ndarray) -> float:
        return float(amplitude * dimension + np.sum(x * x - amplitude * np.cos(2.0 * np.pi * x)))

    return BenchmarkFunction(name=f"rastrigin_{dimension}d", dimension=dimension, fn=_fn)


def himmelblau() -> BenchmarkFunction:
    """Return Himmelblau's function in 2D."""

    def _fn(x: np.ndarray) -> float:
        x0, x1 = x
        return float((x0 * x0 + x1 - 11.0) ** 2 + (x0 + x1 * x1 - 7.0) ** 2)

    return BenchmarkFunction(name="himmelblau_2d", dimension=2, fn=_fn)


def gaussian_mixture_2d() -> BenchmarkFunction:
    """Return a smooth two-dimensional multimodal Gaussian-mixture landscape."""

    centers = np.array([[-2.5, -2.0], [2.0, -1.5], [0.0, 2.5]], dtype=float)
    weights = np.array([1.0, 0.85, 1.15], dtype=float)
    scales = np.array([1.2, 0.9, 1.0], dtype=float)

    def _fn(x: np.ndarray) -> float:
        diff = x[None, :] - centers
        quad = np.sum(diff * diff, axis=1) / (2.0 * scales * scales)
        return float(-np.sum(weights * np.exp(-quad)))

    return BenchmarkFunction(name="gaussian_mix_2d", dimension=2, fn=_fn)
