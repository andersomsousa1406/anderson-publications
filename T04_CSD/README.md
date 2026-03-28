# T04_CSD

This folder contains the fourth theory in my publication archive: `CSD`, short for `Canonical Structural Decomposition`. In the internal regime sequence of the structural line, this folder corresponds to the affine base regime later referenced as `V2`.

In this work, I formulate the structural decomposition of evolving densities in affine coordinates. Instead of separating only mass, translation, and isotropic scale, I separate mass, translation, a global linear deformation, and a normalized residual shape. The central goal is to place measure evolution in coordinates where affine transport and genuine internal deformation are disentangled as canonically as possible.

## Article

- Main English source: `article/Canonical Structural Decomposition and Orthogonalization of Dynamical Modes in Measure Evolution.tex`
- Associated PDF, when available: `article/Canonical Structural Decomposition and Orthogonalization of Dynamical Modes in Measure Evolution.pdf`
- Zenodo record: `https://zenodo.org/records/19290144`

## Structure

- `article/`: the main manuscript in English, with the canonical publication title.
- `experiments/`: numerical verification material associated with the article.
- `simulations/`: simulation notes for this theory.
- `results/`: compact summaries of the evidence recorded in the manuscript.

## Status

This theory is already organized as a publication unit. The English manuscript is the primary article in this folder.

## Initial validation suite

This theory now includes the first executable validation seed for the affine `CSD` regime:

- `experiments/t04_affine_recovery_test.py`
- `simulations/t04_affine_mode_simulation.py`

These files start the broader plan of building ten tests and ten simulations for this theory.
