# T04_CSD results

This folder stores the recorded numerical artifacts currently attached to `T04_CSD`.

## Source

- Experiment: `experiments/t04_affine_recovery_test.py`
- Simulation: `simulations/t04_affine_mode_simulation.py`

## Files

- `summary.md`
- `t04_affine_mode_simulation.json`
- `t04_affine_recovery_test.json`

## Result Summary

# CSD Results Summary

This summary records the compact evidence currently attached to the CSD article.

## Core reading

- The manuscript formulates the affine structural decomposition in coordinates that separate mass, translation, global linear deformation, and residual shape.
- The isotropic theory is recovered as the special case in which the affine matrix is a scalar multiple of the identity.
- The current computational evidence is explicitly presented as one-dimensional numerical verification for that special isotropic case.

## Numerical verification

- Main script: `verify_conjectures.py`
- Environment reported in the manuscript: Python 3.14.3, NumPy 2.4.3, and SciPy 1.17.1

## Interpretation

The main value of this stage is theoretical organization: it establishes the affine canonical gauge and the orthogonalized residual dynamics that later versions refine and extend.
