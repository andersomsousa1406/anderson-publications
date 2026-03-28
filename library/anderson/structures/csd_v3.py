"""Single-chart nonlinear structural descriptors associated with T05_CSD_V3."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CSDV3Descriptor:
    """Placeholder descriptor for the nonlinear single-chart regime."""

    chart_penalty: float = 1.0
    residual_weight: float = 1.0
