# Publications Index

This index explains the sequence of theories currently organized in my `publications/` archive. The collection is not a random set of isolated manuscripts. It records a research line that begins with mathematical stability and reformulation, moves through structural decomposition, and culminates in layered optimization under noise.

## Highlight Results

The archive includes mathematical manuscripts, executable experiments, and visual demonstrations. A few of the most immediately striking results from `T07_GONM` are shown below.

| Atomic structure | Logistics routing |
| --- | --- |
| ![Ar13 atomic structure](T07_GONM/results/gonm_atomic_structure_argon13/gonm_atomic_structure_argon13.png) | ![Geometric logistics routing](T07_GONM/results/gonm_logistics_vrp/gonm_logistics_vrp.png) |
| `Ar13` Lennard-Jones structure with near-reference final energy. | Vehicle-routing demonstration with `25.31%` path-length reduction. |

| Satellite control | Quantum ground state |
| --- | --- |
| ![Satellite attitude control](T07_GONM/results/gonm_satellite_attitude_control/gonm_satellite_attitude_control.png) | ![Quantum ground-state optimization](T07_GONM/results/gonm_quantum_ground_state/gonm_quantum_ground_state.png) |
| Attitude-control tuning under disturbances with a large gain over PID baseline. | Variational search reaching a substantially lower energy than the local baseline. |

| Protein folding | Smart-grid dispatch |
| --- | --- |
| ![Protein folding demonstration](T07_GONM/results/gonm_protein_folding/gonm_protein_folding.png) | ![Smart-grid redispatch](T07_GONM/results/gonm_smart_grid_dispatch/gonm_smart_grid_dispatch.png) |
| Coarse-grained folding demonstration with compact final configuration. | Reduced redispatch example with improved objective and lower line stress. |

These examples are meant to make the archive visually legible at a glance. The broader context, caveats, and supporting scripts remain organized inside `T07_GONM/results/`, `T07_GONM/simulations/`, and `T07_GONM/experiments/`.

## Zenodo Records

- Main `T07_GONM` article record: https://zenodo.org/records/19290813
- Supplementary simulation atlas and computational results: https://zenodo.org/records/19291319

## Sequence Overview

### T01_CDRCGM
`Contractive Dynamical Reformulation of a Classical Geometric Method for the Approximation of pi`

This is the opening mathematical layer of the archive. In it, I show how a classical geometric construction can be reformulated as a contractive dynamical system. The main themes are contraction, stability, perturbation damping, and structural reformulation by change of variables. This article is foundational because it establishes the style of reasoning that later appears again in iterative stability, structural decomposition, and optimization.

### T02_CPP
`Contractive Propagation of Perturbations as a Principle of Stability in Iterative Reconstruction and Self-Consistent Maps`

This is the first direct expansion of the stability perspective into contemporary numerical problems. Here I treat contractive propagation of perturbations (`CPP`) as an operational stability principle rather than as an isolated recurrence property. The article tests that principle in sparse tomography and self-consistent iterative maps.

### T03_MQLM
`Multiplicative Quadrature via Logarithmic Means (MQLM): From Log-Linear Interpolation to High-Order Numerical Integration`

This branch develops a structurally adapted quadrature rule for positive integrands. The central idea is that, for certain families of functions, the right object to interpolate is `log f` rather than `f`. This article stands somewhat parallel to the main decomposition line, but it remains consistent with the same research instinct: numerical methods should be adapted to the geometry that the problem actually presents.

### T04_CSD
`Canonical Structural Decomposition and Orthogonalization of Dynamical Modes in Measure Evolution`

This is the affine base regime of the structural theory. Internally, it is the regime later referenced as `V2`. Here I formulate the canonical structural decomposition of evolving densities in affine coordinates, separating mass, translation, global linear deformation, and residual shape. This article becomes the mathematical base from which the later nonlinear and multichart regimes are developed.

### T05_CSD_V3
`Canonical Nonlinear Global-Chart Structural Decomposition in the Penalized Quadratic Regime`

This is the first regime beyond the affine theory. Here I move from affine structural coordinates to a nonlinear single-chart ontology. The main question is how far one can push a single global chart before that language becomes structurally insufficient. In that sense, `V3` is the nonlinear continuation of `T04_CSD`.

### T06_CSD_V4
`Canonical Multichart Structural Decomposition in the Competitive Regime`

This is the multichart extension of the structural theory. Once the single-chart ontology of `V3` ceases to be adequate, `V4` introduces a multicomponent language together with an external gauge, fusion rules, a competitive reduced class, and a transition criterion from `V3` to `V4`. This article closes the next structural layer of the decomposition line.

### T07_GONM
`GONM: A Layered Mathematical and Experimental Framework for Global Optimization on Noisy Multimodal Surfaces`

This is the optimization synthesis of the archive. In it, I combine ideas that were developed separately in earlier works: structural selection, contractive refinement, robust trajectory filtering, and terminal statistical closure. The manuscript also includes dimensional studies and a CPU-only physical demonstration through Lennard-Jones molecular optimization. Rather than proposing a universal optimizer, the article presents a layered theory of difficulty decomposition for noisy multimodal optimization.

## Python Library

### anderson
`anderson`

Alongside the manuscripts, I maintain the Python library `anderson` as the executable side of the project. It provides reusable components for noisy physical and optimization simulations, including molecular problems, Lennard-Jones potentials, rendering utilities, and a practical GONM-oriented optimization layer. In the current archive, this library is stored under `publications/library/anderson`, and it is especially connected to `T07_GONM`, because it supports the CPU-only LJ-12 and LJ-38 demonstrations and serves as the main reusable software base for continuing that line. The library is distributed under the GNU AGPLv3 license.

## How to Read the Collection

A good reading order is the archive order itself:

1. `T01_CDRCGM`
2. `T02_CPP`
3. `T03_MQLM`
4. `T04_CSD`
5. `T05_CSD_V3`
6. `T06_CSD_V4`
7. `T07_GONM`

If the main interest is the structural line, the shortest path is:

1. `T04_CSD`
2. `T05_CSD_V3`
3. `T06_CSD_V4`
4. `T07_GONM`

If the main interest is stability and numerical mechanism, a useful path is:

1. `T01_CDRCGM`
2. `T02_CPP`
3. `T07_GONM`

## General Interpretation

Taken together, these articles document a single research program. The guiding idea is that difficult numerical problems become more intelligible when one identifies the correct structural coordinates, separates stable and unstable modes, and respects the geometry of the problem instead of forcing a uniform computational language onto every regime.
