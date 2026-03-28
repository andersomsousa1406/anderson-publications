# dimensionality_law_1000d

This folder stores the exported result bundle for one `T07_GONM` experiment or simulation.

## Source

- Script or reference entry: `experiments/optimization_multimodal_noisy_benchmark_dimensionality_law_1000d.py`

## Files

- `summary.md`
- `results.json`

## Result Summary

# GONM Dimensionality Law 1000D

Budget fixo: `1040` | seeds: `3`

## 1000D Phase-1 Sweep

- `gonm_p80`: best_true_mean=`5.9399e+00`, dist_mean=`3.7478e+01`, residual_ratio_mean=`5.6914e-01`
- `gonm_p95`: best_true_mean=`5.9399e+00`, dist_mean=`3.7478e+01`, residual_ratio_mean=`5.6914e-01`

## Scaling Check: 240D vs 1000D

### dim=`240`
- `sim_anneal`: best_true_mean=`1.0317e+01`, residual_ratio_mean=`9.6485e-01`, wall_coeff=`6.6596e-01`, residual_wall_coeff=`6.2281e-02`
- `csd`: best_true_mean=`7.2232e+00`, residual_ratio_mean=`6.7543e-01`, wall_coeff=`4.6625e-01`, residual_wall_coeff=`4.3599e-02`
- `gonm_p80`: best_true_mean=`5.6611e+00`, residual_ratio_mean=`5.2965e-01`, wall_coeff=`3.6542e-01`, residual_wall_coeff=`3.4189e-02`
- `gonm_p95`: best_true_mean=`5.6611e+00`, residual_ratio_mean=`5.2965e-01`, wall_coeff=`3.6542e-01`, residual_wall_coeff=`3.4189e-02`

### dim=`1000`
- `sim_anneal`: best_true_mean=`1.0374e+01`, residual_ratio_mean=`9.9433e-01`, wall_coeff=`3.2805e-01`, residual_wall_coeff=`3.1443e-02`
- `csd`: best_true_mean=`7.2442e+00`, residual_ratio_mean=`6.9434e-01`, wall_coeff=`2.2908e-01`, residual_wall_coeff=`2.1957e-02`
- `gonm_p80`: best_true_mean=`5.9399e+00`, residual_ratio_mean=`5.6914e-01`, wall_coeff=`1.8784e-01`, residual_wall_coeff=`1.7998e-02`
- `gonm_p95`: best_true_mean=`5.9399e+00`, residual_ratio_mean=`5.6914e-01`, wall_coeff=`1.8784e-01`, residual_wall_coeff=`1.7998e-02`


