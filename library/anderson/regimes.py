"""Regime descriptors for structural and noisy optimization behavior."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegimeAssessment:
    """Qualitative description of an optimization regime."""

    label: str
    structural: bool
    local: bool
    thin_funnel: bool
    noisy: bool
    notes: str = ""


STRUCTURAL_REGIME = RegimeAssessment(
    label="structural",
    structural=True,
    local=False,
    thin_funnel=False,
    noisy=True,
    notes="Basin organization dominates the search budget.",
)

MIXED_REGIME = RegimeAssessment(
    label="mixed",
    structural=True,
    local=True,
    thin_funnel=False,
    noisy=True,
    notes="Structural delivery and local contraction are both relevant.",
)

LOCAL_REGIME = RegimeAssessment(
    label="local",
    structural=False,
    local=True,
    thin_funnel=False,
    noisy=False,
    notes="Clean local refinement dominates.",
)

THIN_FUNNEL_REGIME = RegimeAssessment(
    label="thin_funnel",
    structural=False,
    local=True,
    thin_funnel=True,
    noisy=True,
    notes="Local differential information becomes resolution-limited.",
)


__all__ = [
    "LOCAL_REGIME",
    "MIXED_REGIME",
    "RegimeAssessment",
    "STRUCTURAL_REGIME",
    "THIN_FUNNEL_REGIME",
]
