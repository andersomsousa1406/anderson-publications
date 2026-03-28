# GONM | Quantum Ground-State Search

This simulation treats GONM as a variational solver for a one-dimensional Schrodinger problem.

The Hamiltonian is

`H = -1/2 d^2/dx^2 + V(x)`

with

`V(x) = 0.18 x^6 - 1.55 x^4 + 2.7 x^2 + 0.10 x`.

## Recorded outcome

- local baseline best energy: `0.679467`
- GONM final energy: `-0.326213`
- GONM gain versus baseline: `1.005679`

## Variational parameters

### Baseline local

- `sigma_left = 1.0975`
- `sigma_right = 0.9595`
- `half_distance = 0.7430`
- `amplitude_ratio = 0.2770`
- `center_shift = 0.1768`

### GONM

- `sigma_left = 0.4391`
- `sigma_right = 1.1752`
- `half_distance = 1.0724`
- `amplitude_ratio = 0.1030`
- `center_shift = -0.9000`

## Interpretation

This is a variational demonstration, not a proof of exact convergence for the full time-dependent Schrodinger equation. The point is narrower and more honest: with a physically meaningful Hamiltonian and an explicit wavefunction family, the layered GONM search finds a lower-energy state than a purely local optimizer started in a poor region.
