# gonm_relativistic_boundary_geodesic

This folder stores the exported result bundle for one `T07_GONM` experiment or simulation.

## Source

- Script or reference entry: `simulations/gonm_relativistic_boundary_geodesic.py`

## Image

![gonm_relativistic_boundary_geodesic](./gonm_relativistic_boundary_geodesic.png)

## Files

- `summary.md`
- `summary.json`
- `gonm_relativistic_boundary_geodesic.png`

## Result Summary

# GONM | Boundary-Value Relativistic Geodesic

This simulation treats GONM as a boundary-value geodesic solver in Schwarzschild spacetime.

Instead of integrating a trajectory step by step, it optimizes the entire path between two fixed endpoints under an optical-metric action-like functional.

## Recorded outcome

- local baseline best functional: `28.378418`
- GONM final functional: `28.179075`
- GONM gain versus baseline: `0.199343`
- GONM minimum radius: `5.8943 M`

## Interpretation

This is not a full Einstein-equation solver or a Kerr ray-tracer. The narrower claim is still strong: by optimizing the whole curve at once, the layered GONM search reduces geodesic residual and avoids the accumulation of local integration errors near the strong-field region.

