# anderson

`anderson` is the Python library of this project for building simulations and optimizers based on the theories developed in the archive.

The library is distributed under the GNU AGPLv3 license.

The current library brings together:

- `CSD` for structural basin organization
- `CPP` for local contraction and stability
- `MQLM` for trajectory-level filtering
- physical extensions such as thermal noise, potentials, and rendering

The purpose of the library is to make new simulations reproducible and composable without duplicating large blocks of code across standalone scripts.

## Initial modules

- `anderson.potentials`: physical potentials, including Lennard-Jones
- `anderson.noise`: noisy oracles and cooling schedules
- `anderson.problems.molecular`: atomic cluster representations
- `anderson.optimizers.gonm`: reusable implementation of the GONM line
- `anderson.presets`: ready-to-run simulations such as `lj12_fixed`, `lj12_cooling`, `lj38_default`, and `lj38_budget200k`
- `anderson.rendering`: static visualization of geometries and energy trajectories

## Quick use

```python
from anderson import lj12_cooling

sim = lj12_cooling()
result = sim.run()
print(result.final_true_energy)
```

## Current presets

- `lj12_fixed()`: LJ-12 with fixed thermal noise
- `lj12_cooling()`: LJ-12 with a cooling schedule
- `lj38_default()`: LJ-38 under the default budget
- `lj38_budget200k()`: LJ-38 under a heavier budget

## Planned architecture

The articles in the archive suggest a broader module structure than the one currently implemented. The most natural next steps are:

- `anderson.benchmarks`
  - benchmark landscapes and noisy wrappers for `Ackley`, `Rastrigin`, `Himmelblau`, and Gaussian mixtures
- `anderson.regimes`
  - regime identification for structural, mixed, local, and thin-funnel cases
- `anderson.composition`
  - explicit layered operators combining structural selection, contractive refinement, trajectory filtering, and terminal closure
- `anderson.optimizers.cpp`
  - reusable contractive refinement operators derived from the `CPP` line
- `anderson.filters.mqlm`
  - logarithmic and multiplicative filters derived from the `MQLM` line
- `anderson.structures.csd_v2`, `anderson.structures.csd_v3`, `anderson.structures.csd_v4`
  - structural modules corresponding to the `T04`, `T05`, and `T06` articles
- `anderson.diagnostics`
  - resolution-barrier analysis, wall coefficients, budget-density diagnostics, and basin-quality measurements
- `anderson.reports`
  - reproducible tables, figures, and article-oriented summaries

The current implementation already supports the physical and optimization demonstrations, but the long-term goal is to make the theoretical architecture of the archive directly programmable as a library.
