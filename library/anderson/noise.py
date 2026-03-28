from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class FixedNoise:
    std: float

    def scale(self, stage: str, step: int, total_steps: int) -> float:
        _ = stage, step, total_steps
        return self.std


@dataclass(slots=True)
class CoolingNoise:
    start_std: float
    mid_std: float = 0.04
    late_std: float = 0.01
    final_std: float = 0.0

    def scale(self, stage: str, step: int, total_steps: int) -> float:
        if stage == "phase1":
            return self.start_std
        if total_steps <= 1:
            return self.final_std
        frac = min(max(step / (total_steps - 1), 0.0), 1.0)
        if frac < 0.35:
            return self.start_std
        if frac < 0.65:
            return self.mid_std
        if frac < 0.88:
            return self.late_std
        return self.final_std
