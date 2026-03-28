# T03_MQLM results

This folder stores the recorded numerical artifacts currently attached to `T03_MQLM`.

## Source

- Experiment: `experiments/t03_mqlm_log_linear_exactness.py`
- Simulation: `simulations/t03_mqlm_vs_trapezoid.py`

## Files

- `summary.md`
- `results.json`
- `t03_mqlm_log_linear_exactness.json`
- `t03_mqlm_vs_trapezoid.json`

## Result Summary

The benchmark archive in `summary.md` records the broad quadrature comparison behind the `MQLM` article.

Key reading:

- `MQLM` and `MQLM+Richardson` are strongest on positive integrands with multiplicative structure.
- On exponential and near-exponential profiles, the multiplicative family shows a structural advantage over purely additive rules.
- On sign-changing or discontinuous functions, trapezoidal and Simpson-style rules remain the fairer language.
- `weighted_blend` and `adaptive_blend` act as transition tests between arithmetic and logarithmic structure.

## Note

This theory currently has no exported figure inside `results/`. The main artifact here is the benchmark summary and the large `results.json` dataset.
