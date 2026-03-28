from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ProteinFoldingChain:
    sequence: str
    bond_length: float = 1.0
    bond_weight: float = 28.0
    bend_weight: float = 0.9
    contact_sigma: float = 1.15
    hh_epsilon: float = 1.2
    repulsion_epsilon: float = 0.08
    compactness_weight: float = 0.18
    softening: float = 0.35
    bounds: tuple[float, float] | None = (-4.5, 4.5)

    def __post_init__(self) -> None:
        if len(self.sequence) < 4:
            raise ValueError("ProteinFoldingChain requires at least four residues.")
        if any(residue not in {"H", "P"} for residue in self.sequence):
            raise ValueError("ProteinFoldingChain only supports H/P sequences.")

    @property
    def atom_count(self) -> int:
        return len(self.sequence)

    @property
    def dimensions(self) -> int:
        return self.atom_count * 3

    def _apply_bounds(self, pos: np.ndarray) -> np.ndarray:
        pos = np.asarray(pos, dtype=float)
        pos = pos - np.mean(pos, axis=0, keepdims=True)
        if self.bounds is not None:
            lo, hi = self.bounds
            pos = np.clip(pos, lo, hi)
        return pos

    def _repair_chain(self, pos: np.ndarray) -> np.ndarray:
        pos = np.asarray(pos, dtype=float).reshape(self.atom_count, 3).copy()
        fallback_axes = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        for i in range(1, self.atom_count):
            diff = pos[i] - pos[i - 1]
            norm = float(np.linalg.norm(diff))
            if norm < 1e-9:
                diff = fallback_axes[(i - 1) % len(fallback_axes)]
                norm = 1.0
            pos[i] = pos[i - 1] + self.bond_length * diff / norm
        return self._apply_bounds(pos)

    def vector_to_positions(self, x: np.ndarray) -> np.ndarray:
        pos = np.asarray(x, dtype=float).reshape(self.atom_count, 3)
        return self._repair_chain(pos)

    def positions_to_vector(self, positions: np.ndarray) -> np.ndarray:
        return self._repair_chain(np.asarray(positions, dtype=float)).reshape(-1)

    def project(self, x: np.ndarray) -> np.ndarray:
        return self.positions_to_vector(self.vector_to_positions(x))

    def random_geometry(self, rng: np.random.Generator, scale: float) -> np.ndarray:
        pos = np.zeros((self.atom_count, 3), dtype=float)
        direction = rng.normal(size=3)
        direction /= max(float(np.linalg.norm(direction)), 1e-9)
        persistence = float(np.clip(0.78 - 0.08 * (scale - 2.8), 0.45, 0.88))
        for i in range(1, self.atom_count):
            direction = persistence * direction + (1.0 - persistence) * rng.normal(size=3)
            direction /= max(float(np.linalg.norm(direction)), 1e-9)
            pos[i] = pos[i - 1] + self.bond_length * direction
        pos += rng.normal(scale=0.12, size=pos.shape)
        return self.positions_to_vector(pos)

    def _contact_energy_and_gradient(self, pos: np.ndarray) -> tuple[float, np.ndarray]:
        energy = 0.0
        grad = np.zeros_like(pos, dtype=float)
        for i in range(self.atom_count):
            for j in range(i + 3, self.atom_count):
                diff = pos[i] - pos[j]
                r = max(float(np.linalg.norm(diff)), self.softening)
                inv_r = 1.0 / r
                sr6 = (self.contact_sigma * inv_r) ** 6
                sr12 = sr6 * sr6
                both_h = self.sequence[i] == "H" and self.sequence[j] == "H"
                if both_h:
                    energy += 4.0 * self.hh_epsilon * (sr12 - sr6)
                    coeff = 24.0 * self.hh_epsilon * (2.0 * sr12 - sr6) * inv_r * inv_r
                else:
                    energy += self.repulsion_epsilon * sr12
                    coeff = -12.0 * self.repulsion_epsilon * sr12 * inv_r * inv_r
                pair_grad = coeff * diff
                grad[i] += pair_grad
                grad[j] -= pair_grad
        return float(energy), grad

    def energy(self, x: np.ndarray) -> float:
        pos = self.vector_to_positions(x)

        bond_vectors = pos[1:] - pos[:-1]
        bond_lengths = np.linalg.norm(bond_vectors, axis=1)
        bond_residual = bond_lengths - self.bond_length
        bond_energy = 0.5 * self.bond_weight * float(np.sum(bond_residual * bond_residual))

        second_diff = pos[2:] - 2.0 * pos[1:-1] + pos[:-2]
        bend_energy = 0.5 * self.bend_weight * float(np.sum(second_diff * second_diff))

        h_mask = np.asarray([residue == "H" for residue in self.sequence], dtype=bool)
        h_pos = pos[h_mask]
        h_center = np.mean(h_pos, axis=0, keepdims=True)
        compactness_energy = 0.5 * self.compactness_weight * float(np.sum((h_pos - h_center) ** 2))

        contact_energy, _ = self._contact_energy_and_gradient(pos)
        return bond_energy + bend_energy + compactness_energy + contact_energy

    def gradient(self, x: np.ndarray) -> np.ndarray:
        pos = self.vector_to_positions(x)
        grad = np.zeros_like(pos, dtype=float)

        for i in range(self.atom_count - 1):
            diff = pos[i + 1] - pos[i]
            norm = max(float(np.linalg.norm(diff)), 1e-9)
            coeff = self.bond_weight * (norm - self.bond_length) / norm
            pair_grad = coeff * diff
            grad[i] -= pair_grad
            grad[i + 1] += pair_grad

        for i in range(1, self.atom_count - 1):
            curvature = pos[i + 1] - 2.0 * pos[i] + pos[i - 1]
            grad[i - 1] += self.bend_weight * curvature
            grad[i] -= 2.0 * self.bend_weight * curvature
            grad[i + 1] += self.bend_weight * curvature

        h_indices = [i for i, residue in enumerate(self.sequence) if residue == "H"]
        h_pos = pos[h_indices]
        h_center = np.mean(h_pos, axis=0)
        for idx in h_indices:
            grad[idx] += self.compactness_weight * (pos[idx] - h_center)

        _, contact_grad = self._contact_energy_and_gradient(pos)
        grad += contact_grad
        grad -= np.mean(grad, axis=0, keepdims=True)
        return grad.reshape(-1)

    def radial_signature(self, x: np.ndarray) -> np.ndarray:
        pos = self.vector_to_positions(x)
        radii = np.sort(np.linalg.norm(pos, axis=1))
        local_span = np.linalg.norm(pos[2:] - pos[:-2], axis=1)
        return np.concatenate([radii, np.sort(local_span)])

    def radius_of_gyration(self, x: np.ndarray) -> float:
        pos = self.vector_to_positions(x)
        return float(np.sqrt(np.mean(np.sum(pos * pos, axis=1))))

    def hydrophobic_contacts(self, x: np.ndarray, threshold: float = 1.65) -> int:
        pos = self.vector_to_positions(x)
        contacts = 0
        for i in range(self.atom_count):
            if self.sequence[i] != "H":
                continue
            for j in range(i + 3, self.atom_count):
                if self.sequence[j] != "H":
                    continue
                if np.linalg.norm(pos[i] - pos[j]) <= threshold:
                    contacts += 1
        return contacts
