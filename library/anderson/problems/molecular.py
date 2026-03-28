from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from anderson.potentials import LennardJonesPotential


@dataclass(slots=True)
class MolecularCluster:
    atom_count: int
    potential: LennardJonesPotential = field(default_factory=LennardJonesPotential)
    bounds: tuple[float, float] | None = None

    @property
    def dimensions(self) -> int:
        return self.atom_count * 3

    def _apply_boundary(self, pos: np.ndarray) -> np.ndarray:
        pos = np.asarray(pos, dtype=float)
        pos = pos - np.mean(pos, axis=0, keepdims=True)
        if self.bounds is not None:
            lo, hi = self.bounds
            pos = np.clip(pos, lo, hi)
        return pos

    def vector_to_positions(self, x: np.ndarray) -> np.ndarray:
        pos = np.asarray(x, dtype=float).reshape(self.atom_count, 3)
        return self._apply_boundary(pos)

    def positions_to_vector(self, positions: np.ndarray) -> np.ndarray:
        return self._apply_boundary(np.asarray(positions, dtype=float)).reshape(-1)

    def project(self, x: np.ndarray) -> np.ndarray:
        return self.positions_to_vector(self.vector_to_positions(x))

    def random_geometry(self, rng: np.random.Generator, scale: float) -> np.ndarray:
        pos = rng.normal(size=(self.atom_count, 3))
        pos -= np.mean(pos, axis=0, keepdims=True)
        rms = np.sqrt(np.mean(np.sum(pos * pos, axis=1)))
        pos *= scale / max(float(rms), 1e-9)
        return self.positions_to_vector(pos)

    def energy(self, x: np.ndarray) -> float:
        return self.potential.energy(self.vector_to_positions(x))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        grad = self.potential.gradient(self.vector_to_positions(x))
        return self.positions_to_vector(grad)

    def radial_signature(self, x: np.ndarray) -> np.ndarray:
        pos = self.vector_to_positions(x)
        return np.sort(np.linalg.norm(pos, axis=1))

    def pairwise_distances(self, x: np.ndarray) -> np.ndarray:
        pos = self.vector_to_positions(x)
        dists = []
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                dists.append(np.linalg.norm(pos[i] - pos[j]))
        return np.asarray(dists, dtype=float)
