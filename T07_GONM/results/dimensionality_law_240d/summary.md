# GONM Dimensionality Law 240D

Budget fixo: `1040` | seeds: `4`

## 240D Phase-1 Sweep

- `gonm_p52`: best_true_mean=`5.7752e+00`, dist_mean=`1.7569e+01`, residual_ratio_mean=`5.4356e-01`
- `gonm_p80`: best_true_mean=`5.7752e+00`, dist_mean=`1.7569e+01`, residual_ratio_mean=`5.4356e-01`
- `gonm_p90`: best_true_mean=`5.7741e+00`, dist_mean=`1.7569e+01`, residual_ratio_mean=`5.4345e-01`

## Scaling Check: 24D vs 240D

### dim=`24`
- `sim_anneal`: best_true_mean=`8.7044e+00`, residual_ratio_mean=`8.4568e-01`, wall_coeff=`1.7768e+00`, residual_wall_coeff=`1.7262e-01`
- `csd`: best_true_mean=`5.1993e+00`, residual_ratio_mean=`5.0554e-01`, wall_coeff=`1.0613e+00`, residual_wall_coeff=`1.0319e-01`
- `gonm_p80`: best_true_mean=`3.9534e+00`, residual_ratio_mean=`3.8448e-01`, wall_coeff=`8.0698e-01`, residual_wall_coeff=`7.8482e-02`
- `gonm_p90`: best_true_mean=`3.9565e+00`, residual_ratio_mean=`3.8480e-01`, wall_coeff=`8.0761e-01`, residual_wall_coeff=`7.8546e-02`

### dim=`240`
- `sim_anneal`: best_true_mean=`1.0321e+01`, residual_ratio_mean=`9.7078e-01`, wall_coeff=`6.6620e-01`, residual_wall_coeff=`6.2663e-02`
- `csd`: best_true_mean=`7.1777e+00`, residual_ratio_mean=`6.7499e-01`, wall_coeff=`4.6332e-01`, residual_wall_coeff=`4.3571e-02`
- `gonm_p80`: best_true_mean=`5.7752e+00`, residual_ratio_mean=`5.4356e-01`, wall_coeff=`3.7279e-01`, residual_wall_coeff=`3.5087e-02`
- `gonm_p90`: best_true_mean=`5.7741e+00`, residual_ratio_mean=`5.4345e-01`, wall_coeff=`3.7272e-01`, residual_wall_coeff=`3.5080e-02`

