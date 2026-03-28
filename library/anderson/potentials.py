from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class LennardJonesPotential:
    sigma: float = 1.0
    epsilon: float = 1.0
    soften: float = 0.04

    def energy(self, positions: np.ndarray) -> float:
        pos = np.asarray(positions, dtype=float)
        energy = 0.0
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                diff = pos[i] - pos[j]
                r = max(float(np.linalg.norm(diff)), self.soften)
                sr6 = (self.sigma / r) ** 6
                energy += 4.0 * self.epsilon * (sr6 * sr6 - sr6)
        return float(energy)

    def gradient(self, positions: np.ndarray) -> np.ndarray:
        pos = np.asarray(positions, dtype=float)
        grad = np.zeros_like(pos, dtype=float)
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                diff = pos[i] - pos[j]
                r = max(float(np.linalg.norm(diff)), self.soften)
                inv_r = 1.0 / r
                sr6 = (self.sigma * inv_r) ** 6
                coeff = 24.0 * self.epsilon * (2.0 * sr6 * sr6 - sr6) * inv_r * inv_r
                force = coeff * diff
                grad[i] += force
                grad[j] -= force
        grad -= np.mean(grad, axis=0, keepdims=True)
        return grad
