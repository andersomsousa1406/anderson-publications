"""Affine structural descriptors associated with T04_CSD."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CSDV2Descriptor:
    """Affine structural descriptor placeholder for the V2 regime."""

    translation_weight: float = 1.0
    deformation_weight: float = 1.0
    residual_weight: float = 1.0
