# Experiments

The MQLM study is benchmark-driven. I compared midpoint, trapezoidal, Simpson, geometric, MQLM, MQLM+Richardson, weighted blends, adaptive blends, and Gauss32 across multiple categories of integrands.

## Benchmark axes

- smooth positive functions;
- high-curvature positive functions;
- simulated real positive profiles;
- oscillatory, discontinuous, singular, and sign-changing functions.

## Main script

- `quadrature_benchmark_suite.py`

The compact interpretation of the confirmed evidence is stored in `../results/summary.md`.
