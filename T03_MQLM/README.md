# T03_MQLM

This folder contains the third theory in my publication archive: `MQLM`, short for `Multiplicative Quadrature via Logarithmic Means`.

In this work, I develop a quadrature rule for positive integrands based on the logarithmic mean and interpret it as the exact integral of the piecewise log-linear interpolant of nodal data. The theory is not intended as a universal replacement for classical quadrature, but as a structurally adapted rule for problems in which `log f` is more regular than `f` itself.

## Article

- Main English source: `article/Multiplicative Quadrature via Logarithmic Means (MQLM) From Log-Linear Interpolation to High-Order Numerical Integration.tex`
- Associated PDF, when available: `article/Multiplicative Quadrature via Logarithmic Means (MQLM) From Log-Linear Interpolation to High-Order Numerical Integration.pdf`
- Zenodo record: `https://zenodo.org/records/19289974`

## Structure

- `article/`: the main manuscript in English, with the canonical publication title.
- `experiments/`: benchmark code and short experimental notes associated with the article.
- `simulations/`: simulation notes for this theory.
- `results/`: compact summaries and benchmark outputs used to support the article.

## Status

This theory is already organized as a publication unit. The English manuscript is the primary article in this folder.

## Initial validation suite

This theory now includes the first executable validation seed for `MQLM`:

- `experiments/t03_mqlm_log_linear_exactness.py`
- `simulations/t03_mqlm_vs_trapezoid.py`

These files start the broader plan of building ten tests and ten simulations for this theory.
