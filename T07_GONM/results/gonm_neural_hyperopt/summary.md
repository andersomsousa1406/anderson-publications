# GONM Neural Hyperparameter Optimization

This simulation models a noisy hyperparameter search landscape for neural-network training.

The proxy variables are:

- `log10(learning rate)`
- `dropout`

The landscape intentionally contains:

- a broad plateau with weak gradients;
- a locally attractive but suboptimal basin;
- a narrower global basin that rewards structured exploration.

## Recorded outcome

- Adam final loss: `0.710720`
- Adam best loss: `0.710720`
- GONM final loss: `0.491841`
- GONM gain versus Adam final: `0.218879`

## Best hyperparameters

### Adam

- `log10(lr) = -4.4000`
- `lr = 0.000040`
- `dropout = 0.4489`

### GONM

- `log10(lr) = -2.5130`
- `lr = 0.003069`
- `dropout = 0.0598`

## Interpretation

This is not a proof that GONM replaces Adam or SGD in real large-scale training. The demonstration is narrower and more honest: in plateau-heavy hyperparameter landscapes, the layered GONM search can enter better basins than a local gradient-only baseline started from a poor region.
