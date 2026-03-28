# CPP Results Summary

This summary records the compact results reported in the `CPP` article. Lower `MSE` is better in tomography. Lower average iterations and higher convergence rate are better in `SCF`.

## Tomography

| Setting | Baseline | Best CPP regime | Baseline MSE | CPP MSE | Relative gain |
|---|---|---:|---:|---:|---:|
| Tomography, 48 projections | Kaczmarz | `eta = 0.22` | 0.07296 | 0.03169 | 56.6% |
| Tomography, 32 projections | Kaczmarz | `eta = 0.22` | 0.09972 | 0.05426 | 45.6% |
| Tomography, 24 projections | Kaczmarz | `eta = 0.22` | 0.10130 | 0.06175 | 39.0% |

## SCF

| Setting | Baseline | Best CPP regime | Baseline value | CPP value |
|---|---|---:|---:|---:|
| SCF toy, average iterations | Anderson | `alpha = 0.35` | 101.00 | 56.78 |
| SCF toy, convergence rate | Anderson | `alpha = 0.35` | 0.00 | 1.00 |

## Interpretation

The tomographic branch provides direct applied evidence: the improved `CPP` regime beats tuned baselines under undersampling and noise.

The `SCF` branch provides structural evidence: the improved `CPP` regime creates a robust iterative regime where the accelerated baseline tested does not converge.
