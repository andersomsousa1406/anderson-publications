# Experiments

The experimental support for GONM is built around three lines:

- low-dimensional diagnostic benchmarks;
- dimensional sweeps and dimensional-law studies on `Ackley_d`;
- barrier and terminal-closure studies focused on noisy local resolution.

## Main scripts

The scripts included here cover the benchmark checkpoints that most directly support the manuscript, especially the dimensional transition argument and the analysis of the local resolution barrier.

## Visual gallery

This folder stores the benchmark and diagnostic scripts. The generated figures currently live under `../results/`. For convenience, the full visual set already associated with `T07_GONM` is shown below.

### Atomic and molecular results

| Ar13 | LJ-13 |
| --- | --- |
| ![Ar13](../results/gonm_atomic_structure_argon13/gonm_atomic_structure_argon13.png) | ![LJ-13](../results/gonm_atomic_structure_lj13/gonm_atomic_structure_lj13.png) |

| Protein folding | Portfolio |
| --- | --- |
| ![Protein folding](../results/gonm_protein_folding/gonm_protein_folding.png) | ![Portfolio](../results/gonm_quant_portfolio/gonm_quant_portfolio.png) |

### Logistics, learning, and control

| Logistics VRP | Neural hyperopt |
| --- | --- |
| ![Logistics VRP](../results/gonm_logistics_vrp/gonm_logistics_vrp.png) | ![Neural hyperopt](../results/gonm_neural_hyperopt/gonm_neural_hyperopt.png) |

| Neural pruning | Inverted pendulum |
| --- | --- |
| ![Neural pruning](../results/gonm_neural_pruning/gonm_neural_pruning.png) | ![Inverted pendulum](../results/gonm_control_inverted_pendulum/gonm_control_inverted_pendulum.png) |

| Satellite attitude control | Smart-grid dispatch |
| --- | --- |
| ![Satellite attitude control](../results/gonm_satellite_attitude_control/gonm_satellite_attitude_control.png) | ![Smart-grid dispatch](../results/gonm_smart_grid_dispatch/gonm_smart_grid_dispatch.png) |

### Physics and variational results

| Quantum ground state | Quantum wavefunction |
| --- | --- |
| ![Quantum ground state](../results/gonm_quantum_ground_state/gonm_quantum_ground_state.png) | ![Quantum wavefunction](../results/gonm_quantum_wavefunction_optimization/gonm_quantum_wavefunction_optimization.png) |

| Relativistic geodesic | Boundary-value geodesic |
| --- | --- |
| ![Relativistic geodesic](../results/gonm_relativistic_geodesic/gonm_relativistic_geodesic.png) | ![Boundary geodesic](../results/gonm_relativistic_boundary_geodesic/gonm_relativistic_boundary_geodesic.png) |

| Cosmology N-body | Chaotic synchronization |
| --- | --- |
| ![Cosmology N-body](../results/gonm_cosmology_nbody/gonm_cosmology_nbody.png) | ![Chaotic synchronization](../results/gonm_chaotic_crypto/gonm_chaotic_crypto.png) |
