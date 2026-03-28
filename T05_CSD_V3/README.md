# T05_CSD_V3

This folder contains the fifth theory in my publication archive: `CSD_V3`, the nonlinear single-chart regime beyond the affine CSD theory.

In this work, I formulate the first regime beyond the affine structural theory. Instead of representing evolving densities only through mass, translation, affine deformation, and residual shape, I represent them through mass, a nonlinear global structural chart, and a normalized reference profile. The affine theory in `T04_CSD` remains the exact low-order core and corresponds to the regime internally referenced in the manuscripts as `V2`, while `V3` studies how far a single nonlinear chart can be pushed before a multichart ontology becomes necessary.

## Article

- Main English source: `article/Canonical Nonlinear Global-Chart Structural Decomposition in the Penalized Quadratic Regime.tex`
- Associated PDF, when available: `article/Canonical Nonlinear Global-Chart Structural Decomposition in the Penalized Quadratic Regime.pdf`
- Zenodo record: `https://zenodo.org/records/19290316`

## Structure

- `article/`: the main manuscript in English, with the canonical publication title.
- `experiments/`: scripts associated with the core numerical and observational validations of the `V3` regime.
- `simulations/`: simulation notes for this theory.
- `results/`: compact summaries of the strongest evidence reported for `V3`.

## Status

This theory is already organized as a publication unit. The English manuscript is the primary article in this folder.

## Initial validation suite

This theory now includes the first executable validation seed for the nonlinear single-chart regime:

- `experiments/t05_single_chart_consistency_test.py`
- `simulations/t05_chart_distortion_simulation.py`

These files start the broader plan of building ten tests and ten simulations for this theory.
