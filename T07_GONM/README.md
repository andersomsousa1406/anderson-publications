# T07_GONM

This folder contains the seventh theory in my publication archive: `GONM`.

In this work, I formulate GONM as a layered mathematical and experimental framework for noisy multimodal optimization. Rather than claiming a universal optimizer, I decompose the problem into structural basin selection, stable local contraction, trajectory-level filtering, and robust terminal closure. The manuscript also records dimensional regime transitions and physical demonstrations through Lennard-Jones molecular optimization and visual coarse-grained folding.

## Article

- Main English source: `article/GONM A Layered Mathematical and Experimental Framework for Global Optimization on Noisy Multimodal Surfaces.tex`
- Associated PDF, when available: `article/GONM A Layered Mathematical and Experimental Framework for Global Optimization on Noisy Multimodal Surfaces.pdf`
- Zenodo record: `https://zenodo.org/records/19290813`
- Supplementary simulation atlas and computational results: `https://zenodo.org/records/19291319`

## Structure

- `article/`: the main manuscript in English, with the canonical publication title.
- `experiments/`: benchmark and dimensional-study scripts associated with the article.
- `simulations/`: physical and molecular simulation scripts associated with the GONM line.
- `results/`: compact benchmark, dimensional, and physical result folders supporting the manuscript.

## Status

This theory is already organized as a publication unit. The English manuscript is the primary article in this folder.

## Initial validation suite

This theory now includes the first lightweight executable validation seed for the layered GONM formulation:

- `experiments/t07_repeated_average_terminal_test.py`
- `simulations/t07_resolution_barrier_simulation.py`

These files complement the heavier benchmark and molecular scripts already stored in this folder and mark the start of the explicit ten-tests and ten-simulations program for `T07`.

## Visual demonstrations

The organized archive now also includes:

- `simulations/gonm_atomic_structure_lj13.py`
- `results/gonm_atomic_structure_lj13/`
- `simulations/gonm_protein_folding.py`
- `results/gonm_protein_folding/`

These visual examples show how the GONM architecture can be used beyond synthetic benchmarks, both in atomic-cluster calculation and in a compact three-dimensional coarse-grained folding problem.

The archive now also includes a simple real-element example:

- `simulations/gonm_atomic_structure_argon13.py`
- `results/gonm_atomic_structure_argon13/`

This `Ar13` demonstration uses argon-specific Lennard-Jones parameters, making the atomic example less abstract while remaining inexpensive to compute.

The organized simulation archive also includes:

- `simulations/gonm_quant_portfolio.py`
- `results/gonm_quant_portfolio/`

This example translates the layered GONM logic into sparse portfolio search under a high-dimensional risk-return objective with diversification caps.

The organized simulation archive also includes:

- `simulations/gonm_logistics_vrp.py`
- `results/gonm_logistics_vrp/`

This demonstration translates the layered GONM logic into geometric delivery routing, using structural sector assignment, local contraction, and route-level closure.

The organized simulation archive also includes:

- `simulations/gonm_neural_hyperopt.py`
- `results/gonm_neural_hyperopt/`

This demonstration translates the layered GONM logic into plateau-heavy neural hyperparameter search, comparing a local Adam-style trajectory against the structural plus contractive search.

The organized simulation archive also includes:

- `simulations/gonm_quantum_ground_state.py`
- `results/gonm_quantum_ground_state/`

This demonstration treats GONM as a variational solver for a one-dimensional Schr\"odinger Hamiltonian in a double-well anharmonic potential.

The organized simulation archive also includes:

- `simulations/gonm_relativistic_geodesic.py`
- `results/gonm_relativistic_geodesic/`

This demonstration treats GONM as a variational search procedure for an equatorial Schwarzschild geodesic-like trajectory near the strong-field region.

The organized simulation archive also includes:

- `simulations/gonm_cosmology_nbody.py`
- `results/gonm_cosmology_nbody/`

This demonstration compares naive near-singular N-body gravity against a layered GONM clustering search with short-range structural regularization.

The organized simulation archive also includes:

- `simulations/gonm_quantum_wavefunction_optimization.py`
- `results/gonm_quantum_wavefunction_optimization/`

This demonstration treats GONM as a normalized variational optimizer for an effective many-electron wavefunction in a finite basis.

The organized simulation archive also includes:

- `simulations/gonm_relativistic_boundary_geodesic.py`
- `results/gonm_relativistic_boundary_geodesic/`

This demonstration treats GONM as a boundary-value geodesic optimizer that adjusts the whole relativistic path between fixed endpoints in Schwarzschild spacetime.

The organized simulation archive also includes:

- `simulations/gonm_control_inverted_pendulum.py`
- `results/gonm_control_inverted_pendulum/`

This demonstration treats GONM as a robust feedback-gain tuner for inverted-pendulum stabilization under strong wind disturbances.

The organized simulation archive also includes:

- `simulations/gonm_satellite_attitude_control.py`
- `results/gonm_satellite_attitude_control/`

This demonstration treats GONM as a disturbance-rejecting attitude-control tuner for a satellite under noisy sensing and solar-pressure-like torques.

The organized simulation archive also includes:

- `simulations/gonm_smart_grid_dispatch.py`
- `results/gonm_smart_grid_dispatch/`

This demonstration treats GONM as a redispatch optimizer for a smart grid after a transmission-line outage.

The organized simulation archive also includes:

- `simulations/gonm_neural_pruning.py`
- `results/gonm_neural_pruning/`

This demonstration treats GONM as a structured neural-network compression procedure, selecting a smaller hidden subnetwork with a better sparsity-accuracy tradeoff than a simple magnitude-pruning baseline.
