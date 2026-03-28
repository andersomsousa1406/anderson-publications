# gonm_quantum_wavefunction_optimization

This folder stores the exported result bundle for one `T07_GONM` experiment or simulation.

## Source

- Script or reference entry: `simulations/gonm_quantum_wavefunction_optimization.py`

## Image

![gonm_quantum_wavefunction_optimization](./gonm_quantum_wavefunction_optimization.png)

## Files

- `summary.md`
- `summary.json`
- `gonm_quantum_wavefunction_optimization.png`

## Result Summary

# GONM | Quantum Wavefunction Optimization

This simulation treats GONM as a variational optimizer for a normalized wavefunction in an effective many-electron setting.

The trial state is expanded in a finite Gaussian basis, and the functional includes:

- kinetic energy;
- an external anharmonic potential;
- an effective Hartree-like interaction term.

## Recorded outcome

- local baseline best energy: `0.861675`
- GONM final energy: `0.066660`
- GONM gain versus baseline: `0.795015`
- baseline maximum norm drift: `3.843801`

## Interpretation

This is not a full Hartree-Fock or coupled-cluster solver. The narrower claim is still useful: the layered GONM search can optimize a normalized wavefunction family while keeping the coefficients on a physical manifold, whereas a naive local update begins to drift away from the normalization constraint.

