from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _project_simplex(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float).reshape(-1)
    if len(w) == 0:
        return w
    u = np.sort(w)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, len(w) + 1)
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    projected = np.maximum(w - theta, 0.0)
    total = float(np.sum(projected))
    if total <= 1e-12:
        projected[:] = 1.0 / len(projected)
    else:
        projected /= total
    return projected


@dataclass(slots=True)
class SparsePortfolioProblem:
    expected_returns: np.ndarray
    covariance: np.ndarray
    asset_names: list[str]
    sector_names: list[str]
    sectors: np.ndarray
    risk_aversion: float = 8.0
    return_weight: float = 1.0
    sparsity_weight: float = 0.11
    sparsity_epsilon: float = 1e-4
    sector_cap: float = 0.42

    def __post_init__(self) -> None:
        self.expected_returns = np.asarray(self.expected_returns, dtype=float).reshape(-1)
        self.covariance = np.asarray(self.covariance, dtype=float)
        self.sectors = np.asarray(self.sectors, dtype=int).reshape(-1)
        if self.covariance.shape != (len(self.expected_returns), len(self.expected_returns)):
            raise ValueError("Covariance shape must match the number of assets.")
        if len(self.asset_names) != len(self.expected_returns):
            raise ValueError("asset_names length must match the number of assets.")
        if len(self.sectors) != len(self.expected_returns):
            raise ValueError("sectors length must match the number of assets.")

    @property
    def atom_count(self) -> int:
        return len(self.expected_returns)

    @property
    def dimensions(self) -> int:
        return len(self.expected_returns)

    def vector_to_positions(self, x: np.ndarray) -> np.ndarray:
        return self.project(x)

    def positions_to_vector(self, positions: np.ndarray) -> np.ndarray:
        return self.project(positions)

    def project(self, x: np.ndarray) -> np.ndarray:
        w = _project_simplex(np.asarray(x, dtype=float).reshape(-1))
        if self.sector_cap < 1.0:
            for _ in range(3):
                adjusted = False
                for sector in np.unique(self.sectors):
                    mask = self.sectors == sector
                    sector_weight = float(np.sum(w[mask]))
                    if sector_weight > self.sector_cap:
                        w[mask] *= self.sector_cap / sector_weight
                        adjusted = True
                if not adjusted:
                    break
                w = _project_simplex(w)
        return w

    def random_geometry(self, rng: np.random.Generator, scale: float) -> np.ndarray:
        concentration = max(0.12, min(0.9, 0.45 / max(scale, 1e-9)))
        sample = rng.dirichlet(np.full(self.dimensions, concentration))
        return self.project(sample)

    def random_direction(self, rng: np.random.Generator) -> np.ndarray:
        direction = rng.normal(size=self.dimensions)
        direction -= float(np.mean(direction))
        return direction / max(float(np.linalg.norm(direction)), 1e-9)

    def thermal_noise(self, rng: np.random.Generator, scale: float) -> np.ndarray:
        noise = rng.normal(size=self.dimensions)
        noise -= float(np.mean(noise))
        return scale * noise

    def expected_portfolio_return(self, x: np.ndarray) -> float:
        w = self.project(x)
        return float(w @ self.expected_returns)

    def portfolio_variance(self, x: np.ndarray) -> float:
        w = self.project(x)
        return float(w @ self.covariance @ w)

    def sparsity_penalty(self, x: np.ndarray) -> float:
        w = self.project(x)
        return float(np.sum(np.sqrt(w + self.sparsity_epsilon)))

    def energy(self, x: np.ndarray) -> float:
        w = self.project(x)
        variance = float(w @ self.covariance @ w)
        expected = float(w @ self.expected_returns)
        sparse = float(np.sum(np.sqrt(w + self.sparsity_epsilon)))
        return self.risk_aversion * variance - self.return_weight * expected + self.sparsity_weight * sparse

    def gradient(self, x: np.ndarray) -> np.ndarray:
        w = self.project(x)
        grad = (
            2.0 * self.risk_aversion * (self.covariance @ w)
            - self.return_weight * self.expected_returns
            + self.sparsity_weight * 0.5 / np.sqrt(w + self.sparsity_epsilon)
        )
        grad -= float(np.mean(grad))
        return grad

    def radial_signature(self, x: np.ndarray) -> np.ndarray:
        w = self.project(x)
        return np.sort(w)[::-1]

    def effective_assets(self, x: np.ndarray, threshold: float = 0.02) -> int:
        w = self.project(x)
        return int(np.sum(w >= threshold))

    def sector_allocations(self, x: np.ndarray) -> np.ndarray:
        w = self.project(x)
        alloc = np.zeros(len(self.sector_names), dtype=float)
        for sector_idx in range(len(self.sector_names)):
            alloc[sector_idx] = float(np.sum(w[self.sectors == sector_idx]))
        return alloc
