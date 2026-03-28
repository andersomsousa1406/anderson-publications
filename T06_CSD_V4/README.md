# T06_CSD_V4

This folder contains the sixth theory in my publication archive: `CSD_V4`, the canonical multichart extension of the structural theory in its competitive regime.

In this work, I move beyond the affine base regime in `T04_CSD` (internally referenced as `V2`) and the single-chart nonlinear ontology of `V3`, and I formulate the multichart, multicomponent regime. The central point is that the external gauge is no longer left at the level of solver heuristics: it is specified through an admissible external class, a canonical fusion operator, a reduced competitive class, a lexicographic selector, and a formal fallback rule to `V3`.

## Article

- Main English source: `article/Canonical Multichart Structural Decomposition in the Competitive Regime.tex`
- Associated PDF, when available: `article/Canonical Multichart Structural Decomposition in the Competitive Regime.pdf`
- Zenodo record: `https://zenodo.org/records/19290675`

## Structure

- `article/`: the main manuscript in English, with the canonical publication title `Canonical Multichart Structural Decomposition in the Competitive Regime`.
- `experiments/`: solver and validation scripts associated with the `V4` regime.
- `simulations/`: simulation notes for this theory.
- `results/`: compact summaries of the strongest `V4` evidence.

## Status

This theory is already organized as a publication unit. The English manuscript is the primary article in this folder.

## Initial validation suite

This theory now includes the first executable validation seed for the multichart regime:

- `experiments/t06_multichart_consistency_test.py`
- `simulations/t06_transition_simulation.py`

These files start the broader plan of building ten tests and ten simulations for this theory.
