# Experiments

The experimental logic of `CPP` is comparative. I evaluated natural baselines and `CPP`-guided regimes, then deepened the favorable branches through parameter sweeps and final best-vs-best comparisons.

## Confirmed branches

- sparse and noisy tomography, measured by `MSE`;
- self-consistent `SCF` maps, measured by convergence rate and average iterations to tolerance.

## Canonical comparisons

- Tomography baseline: `Kaczmarz`
- Tomography best `CPP` regime: `eta = 0.22`
- SCF baseline: accelerated `Anderson`-type iteration
- SCF best `CPP` regime: `alpha = 0.35`

The compact numbers used in the article are summarized in `../results/summary.md`.
