# GONM Dimensionality Law

## 16D Phase-1 Sweep

### `ackley_16d`, budget=`520`
- `gonm_p52`: best_true_mean=`4.9850e+00`, dist_mean=`3.7122e+00`, residual_ratio_mean=`4.7459e-01`
- `gonm_p68`: best_true_mean=`4.5743e+00`, dist_mean=`3.2660e+00`, residual_ratio_mean=`4.3510e-01`
- `gonm_p80`: best_true_mean=`4.4529e+00`, dist_mean=`3.1648e+00`, residual_ratio_mean=`4.2370e-01`

### `ackley_16d`, budget=`1040`
- `gonm_p52`: best_true_mean=`4.2578e+00`, dist_mean=`2.8296e+00`, residual_ratio_mean=`4.0499e-01`
- `gonm_p68`: best_true_mean=`4.0051e+00`, dist_mean=`2.6445e+00`, residual_ratio_mean=`3.7884e-01`
- `gonm_p80`: best_true_mean=`3.6657e+00`, dist_mean=`2.1751e+00`, residual_ratio_mean=`3.4731e-01`

## Ackley Scaling vs Dimension

### budget=`520`
- dim=`2`
  - `sim_anneal`: best_true_mean=`5.6741e-02`, residual_ratio_mean=`6.8120e-03`, wall_coeff=`4.0122e-02`, residual_wall_coeff=`4.8168e-03`
  - `csd`: best_true_mean=`8.4345e-01`, residual_ratio_mean=`9.6659e-02`, wall_coeff=`5.9641e-01`, residual_wall_coeff=`6.8348e-02`
  - `gonm_p80`: best_true_mean=`1.8965e+00`, residual_ratio_mean=`2.1916e-01`, wall_coeff=`1.3411e+00`, residual_wall_coeff=`1.5497e-01`
- dim=`4`
  - `sim_anneal`: best_true_mean=`3.8265e-01`, residual_ratio_mean=`3.3333e-02`, wall_coeff=`1.9133e-01`, residual_wall_coeff=`1.6666e-02`
  - `csd`: best_true_mean=`2.4216e+00`, residual_ratio_mean=`2.2014e-01`, wall_coeff=`1.2108e+00`, residual_wall_coeff=`1.1007e-01`
  - `gonm_p80`: best_true_mean=`3.6638e+00`, residual_ratio_mean=`3.3204e-01`, wall_coeff=`1.8319e+00`, residual_wall_coeff=`1.6602e-01`
- dim=`8`
  - `sim_anneal`: best_true_mean=`3.7459e+00`, residual_ratio_mean=`3.3572e-01`, wall_coeff=`1.3244e+00`, residual_wall_coeff=`1.1870e-01`
  - `csd`: best_true_mean=`3.5984e+00`, residual_ratio_mean=`3.2648e-01`, wall_coeff=`1.2722e+00`, residual_wall_coeff=`1.1543e-01`
  - `gonm_p80`: best_true_mean=`3.5623e+00`, residual_ratio_mean=`3.2460e-01`, wall_coeff=`1.2595e+00`, residual_wall_coeff=`1.1476e-01`
- dim=`10`
  - `sim_anneal`: best_true_mean=`5.3814e+00`, residual_ratio_mean=`5.2427e-01`, wall_coeff=`1.7018e+00`, residual_wall_coeff=`1.6579e-01`
  - `csd`: best_true_mean=`4.0455e+00`, residual_ratio_mean=`3.9561e-01`, wall_coeff=`1.2793e+00`, residual_wall_coeff=`1.2510e-01`
  - `gonm_p80`: best_true_mean=`3.9822e+00`, residual_ratio_mean=`3.9001e-01`, wall_coeff=`1.2593e+00`, residual_wall_coeff=`1.2333e-01`
- dim=`16`
  - `sim_anneal`: best_true_mean=`7.9412e+00`, residual_ratio_mean=`7.5716e-01`, wall_coeff=`1.9853e+00`, residual_wall_coeff=`1.8929e-01`
  - `csd`: best_true_mean=`5.1066e+00`, residual_ratio_mean=`4.8521e-01`, wall_coeff=`1.2766e+00`, residual_wall_coeff=`1.2130e-01`
  - `gonm_p80`: best_true_mean=`4.4529e+00`, residual_ratio_mean=`4.2370e-01`, wall_coeff=`1.1132e+00`, residual_wall_coeff=`1.0593e-01`

### budget=`1040`
- dim=`2`
  - `sim_anneal`: best_true_mean=`2.3082e-02`, residual_ratio_mean=`2.7528e-03`, wall_coeff=`1.6322e-02`, residual_wall_coeff=`1.9466e-03`
  - `csd`: best_true_mean=`4.4576e-01`, residual_ratio_mean=`5.7649e-02`, wall_coeff=`3.1520e-01`, residual_wall_coeff=`4.0764e-02`
  - `gonm_p80`: best_true_mean=`6.3369e-01`, residual_ratio_mean=`7.2510e-02`, wall_coeff=`4.4809e-01`, residual_wall_coeff=`5.1272e-02`
- dim=`4`
  - `sim_anneal`: best_true_mean=`9.0311e-01`, residual_ratio_mean=`7.6611e-02`, wall_coeff=`4.5156e-01`, residual_wall_coeff=`3.8306e-02`
  - `csd`: best_true_mean=`2.2358e+00`, residual_ratio_mean=`2.0200e-01`, wall_coeff=`1.1179e+00`, residual_wall_coeff=`1.0100e-01`
  - `gonm_p80`: best_true_mean=`2.6129e+00`, residual_ratio_mean=`2.4210e-01`, wall_coeff=`1.3064e+00`, residual_wall_coeff=`1.2105e-01`
- dim=`8`
  - `sim_anneal`: best_true_mean=`3.9805e+00`, residual_ratio_mean=`3.5681e-01`, wall_coeff=`1.4073e+00`, residual_wall_coeff=`1.2615e-01`
  - `csd`: best_true_mean=`3.3349e+00`, residual_ratio_mean=`3.0277e-01`, wall_coeff=`1.1791e+00`, residual_wall_coeff=`1.0704e-01`
  - `gonm_p80`: best_true_mean=`2.8940e+00`, residual_ratio_mean=`2.6257e-01`, wall_coeff=`1.0232e+00`, residual_wall_coeff=`9.2832e-02`
- dim=`10`
  - `sim_anneal`: best_true_mean=`4.5843e+00`, residual_ratio_mean=`4.5183e-01`, wall_coeff=`1.4497e+00`, residual_wall_coeff=`1.4288e-01`
  - `csd`: best_true_mean=`3.6530e+00`, residual_ratio_mean=`3.5615e-01`, wall_coeff=`1.1552e+00`, residual_wall_coeff=`1.1262e-01`
  - `gonm_p80`: best_true_mean=`3.3153e+00`, residual_ratio_mean=`3.2317e-01`, wall_coeff=`1.0484e+00`, residual_wall_coeff=`1.0220e-01`
- dim=`16`
  - `sim_anneal`: best_true_mean=`7.0385e+00`, residual_ratio_mean=`6.6358e-01`, wall_coeff=`1.7596e+00`, residual_wall_coeff=`1.6589e-01`
  - `csd`: best_true_mean=`4.5780e+00`, residual_ratio_mean=`4.3440e-01`, wall_coeff=`1.1445e+00`, residual_wall_coeff=`1.0860e-01`
  - `gonm_p80`: best_true_mean=`3.6657e+00`, residual_ratio_mean=`3.4731e-01`, wall_coeff=`9.1643e-01`, residual_wall_coeff=`8.6828e-02`

