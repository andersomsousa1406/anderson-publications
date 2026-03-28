# Python Library Archive

This directory stores the reusable Python software base associated with the publication archive.

## anderson

`anderson` is my Python library for building simulations and optimization workflows based on the theories developed in this project.

It is distributed under the GNU AGPLv3 license.

It currently provides:

- structural optimization components inspired by `CSD`, `CPP`, `MQLM`, and `GONM`
- noisy and thermal perturbation models
- molecular problem definitions, including Lennard-Jones clusters
- reusable rendering utilities for geometric and energy-based visualization
- presets for `LJ-12`, `LJ-38`, and related demonstrations

Within the archive, this library is especially connected to `T07_GONM`, because it supports the CPU-only molecular demonstrations and the reusable implementation layer behind the physical simulations.

## Planned Architecture

Looking at the current articles, the next natural expansion of the library is to make the theoretical layers appear as explicit reusable modules rather than only as ideas embedded in scripts.

- `anderson.benchmarks`
  - canonical benchmark families such as `Ackley`, `Rastrigin`, `Himmelblau`, Gaussian mixtures, and noisy wrappers
- `anderson.regimes`
  - regime classifiers for structural, mixed, local, and thin-funnel behavior
- `anderson.composition`
  - layered operator objects such as structural stage, contractive stage, trajectory filter, and terminal closure
- `anderson.optimizers.cpp`
  - reusable contractive local refinement operators derived from the `T02_CPP` line
- `anderson.filters.mqlm`
  - logarithmic and multiplicative trajectory filters linked to `T03_MQLM`
- `anderson.structures.csd_v2`
  - affine structural decomposition tools linked to `T04_CSD`
- `anderson.structures.csd_v3`
  - nonlinear single-chart structural tools linked to `T05_CSD_V3`
- `anderson.structures.csd_v4`
  - multichart structural tools linked to `T06_CSD_V4`
- `anderson.diagnostics`
  - wall coefficients, resolution barriers, budget-density diagnostics, and basin-quality measures
- `anderson.reports`
  - reproducible tables, figures, summaries, and export utilities for articles

In that development plan, `T07_GONM` remains the main synthesis layer, while `T01` through `T06` provide the mathematical modules that should gradually become first-class software components.
