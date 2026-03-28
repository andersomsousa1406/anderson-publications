"""Composable layered operators inspired by the GONM manuscript."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class Stage(Protocol):
    """Protocol for a composable stage in a layered optimization pipeline."""

    def __call__(self, state: Any) -> Any:
        ...


@dataclass
class LayeredOperator:
    """Minimal executable composition of structural, local, and terminal stages."""

    structural_stage: Stage
    regime_stage: Stage
    local_stage: Stage
    filter_stage: Stage
    terminal_stage: Stage
    comparison_stage: Stage

    def __call__(self, state: Any) -> Any:
        state = self.structural_stage(state)
        state = self.regime_stage(state)
        state = self.local_stage(state)
        state = self.filter_stage(state)
        state = self.terminal_stage(state)
        return self.comparison_stage(state)
