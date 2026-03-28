# MQLM Results Summary

This folder records the canonical benchmark evidence associated with the MQLM article.

## Core reading

- MQLM and MQLM+Richardson are naturally restricted to positive integrands.
- On exponential and near-exponential positive profiles, the MQLM family shows a structural advantage over purely additive rules.
- On sign-changing or discontinuous functions, multiplicative quadrature is no longer the natural language, and the fair comparison shifts back to trapezoidal and Simpson-type rules.
- Weighted and adaptive blends were included to test whether mixed arithmetic-logarithmic structure helps in transitional regimes.

## Benchmark archive

The full benchmark summary from the original study is reproduced below.

# Broad Quadrature Benchmark

Methods compared: Midpoint, Trapezoid, Simpson, Geometric, MQLM, MQLM+Richardson, Weighted Blend, Adaptive Blend e Gauss32.

## Summary by category on the medium grid (n=100)

### `discontinuous`
- `midpoint`: mean error=5.551e-17, mean time=0.009 ms, mean evaluations=100.0
- `trapezoid`: mean error=2.500e-03, mean time=0.012 ms, mean evaluations=101.0
- `simpson`: mean error=1.667e-03, mean time=0.012 ms, mean evaluations=101.0
- `geometric`: no valid cases
- `mqlm`: no valid cases
- `mqlm_rich`: no valid cases
- `weighted_blend`: no valid cases
- `adaptive_blend`: mean error=2.500e-03, mean time=0.023 ms, mean evaluations=302.0
- `gauss32`: mean error=5.701e-03, mean time=0.470 ms, mean evaluations=32.0

### `high_curvature`
- `midpoint`: mean error=4.587e-01, mean time=0.008 ms, mean evaluations=100.0
- `trapezoid`: mean error=9.176e-01, mean time=0.012 ms, mean evaluations=101.0
- `simpson`: mean error=6.111e-04, mean time=0.012 ms, mean evaluations=101.0
- `geometric`: mean error=4.591e-01, mean time=0.015 ms, mean evaluations=101.0
- `mqlm`: mean error=2.606e-04, mean time=0.030 ms, mean evaluations=101.0
- `mqlm_rich`: mean error=3.569e-07, mean time=0.059 ms, mean evaluations=302.0
- `weighted_blend`: mean error=4.589e-01, mean time=0.029 ms, mean evaluations=101.0
- `adaptive_blend`: mean error=2.137e-03, mean time=0.045 ms, mean evaluations=201.0
- `gauss32`: mean error=4.764e-04, mean time=0.483 ms, mean evaluations=32.0

### `oscillatory`
- `midpoint`: mean error=1.004e-04, mean time=0.009 ms, mean evaluations=100.0
- `trapezoid`: mean error=1.990e-04, mean time=0.012 ms, mean evaluations=101.0
- `simpson`: mean error=1.074e-05, mean time=0.013 ms, mean evaluations=101.0
- `geometric`: no valid cases
- `mqlm`: no valid cases
- `mqlm_rich`: no valid cases
- `weighted_blend`: no valid cases
- `adaptive_blend`: mean error=1.990e-04, mean time=0.022 ms, mean evaluations=302.0
- `gauss32`: mean error=3.876e-05, mean time=0.472 ms, mean evaluations=32.0

### `sign_change`
- `midpoint`: mean error=2.540e-17, mean time=0.009 ms, mean evaluations=100.0
- `trapezoid`: mean error=1.746e-17, mean time=0.014 ms, mean evaluations=101.0
- `simpson`: mean error=2.571e-18, mean time=0.012 ms, mean evaluations=101.0
- `geometric`: no valid cases
- `mqlm`: no valid cases
- `mqlm_rich`: no valid cases
- `weighted_blend`: no valid cases
- `adaptive_blend`: mean error=1.746e-17, mean time=0.026 ms, mean evaluations=302.0
- `gauss32`: mean error=1.784e-17, mean time=0.474 ms, mean evaluations=32.0

### `simulated_real`
- `midpoint`: mean error=1.096e-03, mean time=0.011 ms, mean evaluations=100.0
- `trapezoid`: mean error=2.181e-03, mean time=0.016 ms, mean evaluations=101.0
- `simpson`: mean error=5.766e-05, mean time=0.016 ms, mean evaluations=101.0
- `geometric`: mean error=2.841e-02, mean time=0.017 ms, mean evaluations=101.0
- `mqlm`: mean error=1.967e-02, mean time=0.037 ms, mean evaluations=101.0
- `mqlm_rich`: mean error=1.589e-04, mean time=0.075 ms, mean evaluations=302.0
- `weighted_blend`: mean error=1.093e-02, mean time=0.035 ms, mean evaluations=101.0
- `adaptive_blend`: mean error=9.194e-03, mean time=0.054 ms, mean evaluations=201.0
- `gauss32`: mean error=4.326e-09, mean time=0.557 ms, mean evaluations=32.0

### `singularity`
- `midpoint`: mean error=1.756e-03, mean time=0.008 ms, mean evaluations=100.0
- `trapezoid`: mean error=1.853e-02, mean time=0.011 ms, mean evaluations=101.0
- `simpson`: mean error=1.118e-02, mean time=0.011 ms, mean evaluations=101.0
- `geometric`: no valid cases
- `mqlm`: no valid cases
- `mqlm_rich`: no valid cases
- `weighted_blend`: no valid cases
- `adaptive_blend`: mean error=1.853e-02, mean time=0.020 ms, mean evaluations=302.0
- `gauss32`: mean error=2.971e-04, mean time=0.495 ms, mean evaluations=32.0

### `smooth`
- `midpoint`: mean error=2.444e-05, mean time=0.009 ms, mean evaluations=100.0
- `trapezoid`: mean error=4.887e-05, mean time=0.014 ms, mean evaluations=101.0
- `simpson`: mean error=2.730e-09, mean time=0.013 ms, mean evaluations=101.0
- `geometric`: mean error=7.159e-06, mean time=0.016 ms, mean evaluations=101.0
- `mqlm`: mean error=2.220e-16, mean time=0.039 ms, mean evaluations=101.0
- `mqlm_rich`: mean error=2.220e-16, mean time=0.101 ms, mean evaluations=302.0
- `weighted_blend`: mean error=7.159e-06, mean time=0.057 ms, mean evaluations=101.0
- `adaptive_blend`: mean error=4.529e-05, mean time=0.039 ms, mean evaluations=276.8
- `gauss32`: mean error=1.804e-16, mean time=0.648 ms, mean evaluations=32.0

## Convergence

- `sin_0_pi` / `trapezoid`: order~2.00, initial_error=1.648e-02, final_error=1.645e-06, monotone_improvement=True
- `sin_0_pi` / `simpson`: order~4.00, initial_error=1.095e-04, final_error=1.082e-12, monotone_improvement=True
- `sin_0_pi` / `adaptive_blend`: order~2.00, initial_error=1.648e-02, final_error=1.645e-06, monotone_improvement=True
- `exp_0_1` / `trapezoid`: order~2.00, initial_error=1.432e-03, final_error=1.432e-07, monotone_improvement=True
- `exp_0_1` / `simpson`: order~4.00, initial_error=9.535e-07, final_error=9.770e-15, monotone_improvement=True
- `exp_0_1` / `mqlm`: order~-2.60, initial_error=1.000e-30, final_error=1.000e-30, monotone_improvement=True
- `exp_0_1` / `mqlm_rich`: order~0.15, initial_error=6.661e-16, final_error=2.220e-16, monotone_improvement=True
- `exp_0_1` / `adaptive_blend`: order~1.99, initial_error=2.953e-07, final_error=3.252e-11, monotone_improvement=True
- `x2_0_1` / `trapezoid`: order~2.00, initial_error=1.667e-03, final_error=1.667e-07, monotone_improvement=True
- `x2_0_1` / `simpson`: order~-2.65, initial_error=1.000e-30, final_error=1.000e-30, monotone_improvement=True
- `x2_0_1` / `adaptive_blend`: order~2.00, initial_error=1.667e-03, final_error=1.667e-07, monotone_improvement=True
- `x3_m1_1` / `trapezoid`: order~1.96, initial_error=6.661e-17, final_error=2.842e-17, monotone_improvement=True
- `x3_m1_1` / `simpson`: order~2.65, initial_error=4.811e-17, final_error=1.000e-30, monotone_improvement=True
- `x3_m1_1` / `adaptive_blend`: order~1.96, initial_error=6.661e-17, final_error=2.842e-17, monotone_improvement=True
- `sqrt_0_1` / `trapezoid`: order~1.49, initial_error=6.157e-03, final_error=6.532e-06, monotone_improvement=True
- `sqrt_0_1` / `simpson`: order~1.50, initial_error=2.567e-03, final_error=2.567e-06, monotone_improvement=True
- `sqrt_0_1` / `adaptive_blend`: order~1.49, initial_error=6.157e-03, final_error=6.532e-06, monotone_improvement=True
- `log_eps_1` / `trapezoid`: order~1.14, initial_error=4.829e-01, final_error=2.542e-03, monotone_improvement=True
- `log_eps_1` / `simpson`: order~1.15, initial_error=2.994e-01, final_error=1.466e-03, monotone_improvement=True
- `log_eps_1` / `adaptive_blend`: order~1.14, initial_error=4.829e-01, final_error=2.542e-03, monotone_improvement=True
- `exp10_0_1` / `trapezoid`: order~2.00, initial_error=1.806e+02, final_error=1.835e-02, monotone_improvement=True
- `exp10_0_1` / `simpson`: order~3.98, initial_error=1.092e+01, final_error=1.224e-07, monotone_improvement=True
- `exp10_0_1` / `mqlm`: order~-6.23, initial_error=1.000e-30, final_error=4.547e-13, monotone_improvement=False
- `exp10_0_1` / `mqlm_rich`: order~-9.54, initial_error=1.000e-30, final_error=4.547e-13, monotone_improvement=False
- `exp10_0_1` / `adaptive_blend`: order~1.63, initial_error=2.069e-02, final_error=3.436e-05, monotone_improvement=False
- `rational_peak` / `trapezoid`: order~3.07, initial_error=2.825e-02, final_error=1.307e-08, monotone_improvement=True
- `rational_peak` / `simpson`: order~6.76, initial_error=7.388e-02, final_error=4.063e-14, monotone_improvement=True
- `rational_peak` / `mqlm`: order~1.83, initial_error=1.966e-02, final_error=5.240e-06, monotone_improvement=True
- `rational_peak` / `mqlm_rich`: order~4.03, initial_error=7.982e-03, final_error=7.199e-11, monotone_improvement=True
- `rational_peak` / `adaptive_blend`: order~1.90, initial_error=2.680e-02, final_error=2.887e-06, monotone_improvement=True
- `sin10_0_1` / `trapezoid`: order~2.00, initial_error=1.559e-02, final_error=1.533e-06, monotone_improvement=True
- `sin10_0_1` / `simpson`: order~4.02, initial_error=1.158e-03, final_error=1.022e-11, monotone_improvement=True
- `sin10_0_1` / `adaptive_blend`: order~2.00, initial_error=1.559e-02, final_error=1.533e-06, monotone_improvement=True
- `sin50_0_1` / `trapezoid`: order~2.13, initial_error=3.046e-03, final_error=1.460e-07, monotone_improvement=True
- `sin50_0_1` / `simpson`: order~4.13, initial_error=3.482e-03, final_error=2.434e-11, monotone_improvement=True
- `sin50_0_1` / `adaptive_blend`: order~2.13, initial_error=3.046e-03, final_error=1.460e-07, monotone_improvement=True
- `cos100_0_1` / `trapezoid`: order~1.85, initial_error=1.255e-02, final_error=4.220e-06, monotone_improvement=False
- `cos100_0_1` / `simpson`: order~3.75, initial_error=4.108e-02, final_error=2.816e-09, monotone_improvement=True
- `cos100_0_1` / `adaptive_blend`: order~1.85, initial_error=1.255e-02, final_error=4.220e-06, monotone_improvement=False
- `x3_minus_x` / `trapezoid`: order~4.13, initial_error=8.882e-17, final_error=1.000e-30, monotone_improvement=True
- `x3_minus_x` / `simpson`: order~2.24, initial_error=5.921e-17, final_error=2.842e-17, monotone_improvement=True
- `x3_minus_x` / `adaptive_blend`: order~4.13, initial_error=8.882e-17, final_error=1.000e-30, monotone_improvement=True
- `sin_cos` / `trapezoid`: order~0.06, initial_error=1.797e-18, final_error=1.924e-19, monotone_improvement=True
- `sin_cos` / `simpson`: order~0.06, initial_error=2.205e-17, final_error=5.965e-17, monotone_improvement=True
- `sin_cos` / `adaptive_blend`: order~0.06, initial_error=1.797e-18, final_error=1.924e-19, monotone_improvement=True
- `step_0_1` / `trapezoid`: order~1.00, initial_error=5.000e-02, final_error=5.000e-04, monotone_improvement=True
- `step_0_1` / `simpson`: order~1.13, initial_error=6.667e-02, final_error=3.333e-04, monotone_improvement=False
- `step_0_1` / `adaptive_blend`: order~1.00, initial_error=5.000e-02, final_error=5.000e-04, monotone_improvement=True
- `sign_m1_1` / `trapezoid`: order~-0.00, initial_error=1.000e-30, final_error=1.000e-30, monotone_improvement=True
- `sign_m1_1` / `simpson`: order~-0.00, initial_error=1.000e-30, final_error=1.000e-30, monotone_improvement=True
- `sign_m1_1` / `adaptive_blend`: order~-0.00, initial_error=1.000e-30, final_error=1.000e-30, monotone_improvement=True
- `velocity_distance_1` / `trapezoid`: order~2.03, initial_error=2.519e-01, final_error=2.114e-05, monotone_improvement=True
- `velocity_distance_1` / `simpson`: order~4.45, initial_error=1.736e+00, final_error=1.269e-09, monotone_improvement=True
- `velocity_distance_1` / `mqlm`: order~1.86, initial_error=6.994e-01, final_error=1.533e-04, monotone_improvement=True
- `velocity_distance_1` / `mqlm_rich`: order~3.96, initial_error=2.016e-01, final_error=2.475e-09, monotone_improvement=True
- `velocity_distance_1` / `adaptive_blend`: order~1.86, initial_error=4.034e-01, final_error=8.230e-05, monotone_improvement=True
- `velocity_distance_2` / `trapezoid`: order~1.90, initial_error=1.023e-01, final_error=2.221e-05, monotone_improvement=True
- `velocity_distance_2` / `simpson`: order~3.83, initial_error=2.167e-01, final_error=9.483e-09, monotone_improvement=True
- `velocity_distance_2` / `mqlm`: order~1.17, initial_error=2.568e-02, final_error=2.528e-04, monotone_improvement=False
- `velocity_distance_2` / `mqlm_rich`: order~3.55, initial_error=2.592e-01, final_error=3.017e-08, monotone_improvement=True
- `velocity_distance_2` / `adaptive_blend`: order~1.48, initial_error=6.586e-02, final_error=1.107e-04, monotone_improvement=False

## Noise (n=100, average across seeds)

- `sin_0_pi` / `trapezoid`: mean_error=5.052e-03, std=4.662e-03, valid_rate=100.00%
- `sin_0_pi` / `simpson`: mean_error=6.266e-03, std=4.606e-03, valid_rate=100.00%
- `sin_0_pi` / `mqlm`: mean_error=2.022e-03, std=1.769e-03, valid_rate=16.67%
- `sin_0_pi` / `mqlm_rich`: mean_error=3.201e-03, std=1.842e-03, valid_rate=16.67%
- `sin_0_pi` / `adaptive_blend`: mean_error=4.924e-03, std=4.719e-03, valid_rate=100.00%
- `exp_0_1` / `trapezoid`: mean_error=2.961e-03, std=2.138e-03, valid_rate=100.00%
- `exp_0_1` / `simpson`: mean_error=3.468e-03, std=2.438e-03, valid_rate=100.00%
- `exp_0_1` / `mqlm`: mean_error=2.993e-03, std=2.180e-03, valid_rate=100.00%
- `exp_0_1` / `mqlm_rich`: mean_error=2.961e-03, std=2.138e-03, valid_rate=100.00%
- `exp_0_1` / `adaptive_blend`: mean_error=2.970e-03, std=2.149e-03, valid_rate=100.00%
- `x2_0_1` / `trapezoid`: mean_error=7.188e-04, std=3.329e-04, valid_rate=100.00%
- `x2_0_1` / `simpson`: mean_error=7.489e-04, std=5.322e-04, valid_rate=100.00%
- `x2_0_1` / `mqlm`: mean_error=7.341e-04, std=3.451e-04, valid_rate=100.00%
- `x2_0_1` / `mqlm_rich`: mean_error=7.188e-04, std=3.329e-04, valid_rate=100.00%
- `x2_0_1` / `adaptive_blend`: mean_error=7.339e-04, std=3.450e-04, valid_rate=100.00%
- `x3_m1_1` / `trapezoid`: mean_error=3.199e-03, std=2.905e-03, valid_rate=100.00%
- `x3_m1_1` / `simpson`: mean_error=3.989e-03, std=2.932e-03, valid_rate=100.00%
- `x3_m1_1` / `mqlm`: no valid runs
- `x3_m1_1` / `mqlm_rich`: no valid runs
- `x3_m1_1` / `adaptive_blend`: mean_error=3.199e-03, std=2.905e-03, valid_rate=100.00%
- `sqrt_0_1` / `trapezoid`: mean_error=1.241e-03, std=8.996e-04, valid_rate=100.00%
- `sqrt_0_1` / `simpson`: mean_error=1.335e-03, std=9.778e-04, valid_rate=100.00%
- `sqrt_0_1` / `mqlm`: mean_error=1.537e-03, std=1.102e-03, valid_rate=100.00%
- `sqrt_0_1` / `mqlm_rich`: mean_error=1.247e-03, std=9.096e-04, valid_rate=100.00%
- `sqrt_0_1` / `adaptive_blend`: mean_error=1.267e-03, std=9.464e-04, valid_rate=100.00%
- `log_eps_1` / `trapezoid`: mean_error=5.243e-02, std=2.547e-02, valid_rate=100.00%
- `log_eps_1` / `simpson`: mean_error=3.852e-02, std=2.721e-02, valid_rate=100.00%
- `log_eps_1` / `mqlm`: no valid runs
- `log_eps_1` / `mqlm_rich`: no valid runs
- `log_eps_1` / `adaptive_blend`: mean_error=5.243e-02, std=2.547e-02, valid_rate=100.00%
- `exp10_0_1` / `trapezoid`: mean_error=7.539e+00, std=6.012e+00, valid_rate=100.00%
- `exp10_0_1` / `simpson`: mean_error=8.495e+00, std=7.504e+00, valid_rate=100.00%
- `exp10_0_1` / `mqlm`: mean_error=8.212e+00, std=5.839e+00, valid_rate=100.00%
- `exp10_0_1` / `mqlm_rich`: mean_error=7.539e+00, std=6.012e+00, valid_rate=100.00%
- `exp10_0_1` / `adaptive_blend`: mean_error=7.643e+00, std=5.972e+00, valid_rate=100.00%
- `rational_peak` / `trapezoid`: mean_error=7.170e-04, std=6.239e-04, valid_rate=100.00%
- `rational_peak` / `simpson`: mean_error=7.945e-04, std=6.866e-04, valid_rate=100.00%
- `rational_peak` / `mqlm`: mean_error=8.723e-04, std=6.632e-04, valid_rate=100.00%
- `rational_peak` / `mqlm_rich`: mean_error=7.170e-04, std=6.239e-04, valid_rate=100.00%
- `rational_peak` / `adaptive_blend`: mean_error=7.199e-04, std=6.302e-04, valid_rate=100.00%
- `sin10_0_1` / `trapezoid`: mean_error=1.630e-03, std=1.542e-03, valid_rate=100.00%
- `sin10_0_1` / `simpson`: mean_error=1.994e-03, std=1.466e-03, valid_rate=100.00%
- `sin10_0_1` / `mqlm`: no valid runs
- `sin10_0_1` / `mqlm_rich`: no valid runs
- `sin10_0_1` / `adaptive_blend`: mean_error=1.630e-03, std=1.542e-03, valid_rate=100.00%
- `sin50_0_1` / `trapezoid`: mean_error=1.602e-03, std=1.461e-03, valid_rate=100.00%
- `sin50_0_1` / `simpson`: mean_error=1.994e-03, std=1.466e-03, valid_rate=100.00%
- `sin50_0_1` / `mqlm`: no valid runs
- `sin50_0_1` / `mqlm_rich`: no valid runs
- `sin50_0_1` / `adaptive_blend`: mean_error=1.602e-03, std=1.461e-03, valid_rate=100.00%
- `cos100_0_1` / `trapezoid`: mean_error=1.542e-03, std=1.228e-03, valid_rate=100.00%
- `cos100_0_1` / `simpson`: mean_error=2.005e-03, std=1.472e-03, valid_rate=100.00%
- `cos100_0_1` / `mqlm`: no valid runs
- `cos100_0_1` / `mqlm_rich`: no valid runs
- `cos100_0_1` / `adaptive_blend`: mean_error=1.542e-03, std=1.228e-03, valid_rate=100.00%
- `x3_minus_x` / `trapezoid`: mean_error=3.199e-03, std=2.905e-03, valid_rate=100.00%
- `x3_minus_x` / `simpson`: mean_error=3.989e-03, std=2.932e-03, valid_rate=100.00%
- `x3_minus_x` / `mqlm`: no valid runs
- `x3_minus_x` / `mqlm_rich`: no valid runs
- `x3_minus_x` / `adaptive_blend`: mean_error=3.199e-03, std=2.905e-03, valid_rate=100.00%
- `sin_cos` / `trapezoid`: mean_error=5.025e-03, std=4.563e-03, valid_rate=100.00%
- `sin_cos` / `simpson`: mean_error=6.266e-03, std=4.606e-03, valid_rate=100.00%
- `sin_cos` / `mqlm`: no valid runs
- `sin_cos` / `mqlm_rich`: no valid runs
- `sin_cos` / `adaptive_blend`: mean_error=5.025e-03, std=4.563e-03, valid_rate=100.00%
- `step_0_1` / `trapezoid`: mean_error=3.874e-03, std=1.843e-03, valid_rate=100.00%
- `step_0_1` / `simpson`: mean_error=2.823e-03, std=1.792e-03, valid_rate=100.00%
- `step_0_1` / `mqlm`: no valid runs
- `step_0_1` / `mqlm_rich`: no valid runs
- `step_0_1` / `adaptive_blend`: mean_error=3.874e-03, std=1.843e-03, valid_rate=100.00%
- `sign_m1_1` / `trapezoid`: mean_error=3.199e-03, std=2.905e-03, valid_rate=100.00%
- `sign_m1_1` / `simpson`: mean_error=3.989e-03, std=2.932e-03, valid_rate=100.00%
- `sign_m1_1` / `mqlm`: no valid runs
- `sign_m1_1` / `mqlm_rich`: no valid runs
- `sign_m1_1` / `adaptive_blend`: mean_error=3.199e-03, std=2.905e-03, valid_rate=100.00%
- `velocity_distance_1` / `trapezoid`: mean_error=5.008e-02, std=4.180e-02, valid_rate=100.00%
- `velocity_distance_1` / `simpson`: mean_error=6.372e-02, std=3.545e-02, valid_rate=100.00%
- `velocity_distance_1` / `mqlm`: mean_error=6.099e-02, std=4.224e-02, valid_rate=100.00%
- `velocity_distance_1` / `mqlm_rich`: mean_error=5.008e-02, std=4.180e-02, valid_rate=100.00%
- `velocity_distance_1` / `adaptive_blend`: mean_error=5.198e-02, std=4.265e-02, valid_rate=100.00%
- `velocity_distance_2` / `trapezoid`: mean_error=1.852e-02, std=1.652e-02, valid_rate=100.00%
- `velocity_distance_2` / `simpson`: mean_error=2.121e-02, std=1.688e-02, valid_rate=100.00%
- `velocity_distance_2` / `mqlm`: mean_error=3.604e-02, std=1.981e-02, valid_rate=100.00%
- `velocity_distance_2` / `mqlm_rich`: mean_error=1.852e-02, std=1.652e-02, valid_rate=100.00%
- `velocity_distance_2` / `adaptive_blend`: mean_error=2.071e-02, std=1.730e-02, valid_rate=100.00%

## Consolidated reading

- `MQLM` e `MQLM+Richardson` are only cleanly applicable to positive integrands.
- on exponential and near-exponential functions, the MQLM family should have a structural advantage over purely additive rules.
- on functions with sign change or discontinuity, multiplicative rules cease to be the natural language; here the fairest comparisons are against trapezoidal and Simpson rules.
- `weighted_blend` e `adaptive_blend` test whether combinations of arithmetic averaging and logarithmic structure help in mixed regimes.
- `gauss32` enters as a classical high-accuracy fixed-cost comparator, not as a uniform-grid method.
