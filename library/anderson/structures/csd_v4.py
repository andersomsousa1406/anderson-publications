"""Multichart structural descriptors associated with T06_CSD_V4."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CSDV4Descriptor:
    """Placeholder descriptor for multichart structural organization."""

    chart_count: int = 1
    fusion_weight: float = 1.0
    gauge_weight: float = 1.0
