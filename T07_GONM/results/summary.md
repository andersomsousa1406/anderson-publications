# GONM Results Summary

This summary records the strongest compact evidence attached to the GONM manuscript.

For the full visual catalogue with one section per simulation, image previews, and the main quantitative outcomes, see:

- `results/SIMULATION_ATLAS.md`
- Supplementary Zenodo record: `https://zenodo.org/records/19291319`

## Core findings

- The low-dimensional benchmark is heterogeneous by regime; no single local mechanism dominates all surfaces.
- Ackley acts as the main diagnostic case for a local numerical-statistical resolution barrier under noise.
- The pure structural layer `CSD` exhibits a partial `sqrt(D)` law in the medium/high-dimensional regime.
- The full `GONM` architecture operates systematically below the structural wall of `CSD`.
- At extreme dimension, the experiment enters a budget-limited regime.

## Dimensional sweep highlights

- `1D-4D`: low-dimensional local regime.
- `5D-100D`: coherent medium/high-dimensional structural regime.
- `100D-1000D`: transition into the extreme budget-limited regime.

## Physical demonstration

The `LJ-38` branch provides a CPU-only physical demonstration of the architecture. Representative energy progression recorded in the manuscript:

- baseline LJ-38: `-15.0152`
- dense cold start + boundary: `-100.4449`
- continued refinement: `-151.1026`
- parallel escape-freeze best branch: `-151.915874`

The organized archive also includes a compact `LJ-13` atomic-structure calculation:

- final energy: recorded in `results/gonm_atomic_structure_lj13/summary.json`
- output: three-dimensional final cluster image plus energy trace
- purpose: a smaller visual entry point for the same layered physical logic

The archive also includes a simple real-element variant:

- `Ar13` argon cluster
- Lennard-Jones parameters: `sigma = 3.405 A`, `epsilon = 0.0103 eV`
- output stored in `results/gonm_atomic_structure_argon13/`

The simulation archive also includes a quantitative-finance demonstration:

- sparse portfolio search over `60` assets
- high-dimensional simplex-constrained optimization
- comparison between a projected-gradient baseline and the layered GONM search

The archive also includes a logistics demonstration:

- `120` delivery points
- `6` vehicles
- geometric baseline versus layered GONM routing
- relative route-length gain of about `25.31%` in the recorded run

The archive also includes a neural-hyperparameter demonstration:

- proxy variables: `log10(learning rate)` and `dropout`
- synthetic validation landscape with plateau, local basin, and narrow global basin
- Adam final loss: `0.710720`
- GONM final loss: `0.491841`
- gain of about `0.218879` in the recorded run

The archive also includes a quantum variational demonstration:

- one-dimensional Schr\"odinger Hamiltonian in a double-well anharmonic potential
- local variational baseline best energy: `0.679467`
- GONM final variational energy: `-0.326213`
- gain of about `1.005679` in the recorded run

The archive also includes a relativistic geodesic demonstration:

- equatorial Schwarzschild trajectory in a strong-field scattering configuration
- local baseline best functional: `255.944316`
- GONM final functional: `15.737340`
- final periastron near `4.0238 M`
- markedly lower geodesic residual and invariant drift in the recorded run

The archive also includes a reduced cosmological N-body demonstration:

- naive near-singular gravity reaches minimum pair distance `0.019775`
- instability threshold is crossed around step `13`
- GONM final minimum pair distance remains around `1.416875`
- the baseline local optimizer still minimizes the softened functional slightly more, but with less short-range safety margin than GONM

The archive also includes a quantum wavefunction optimization demonstration:

- effective many-electron variational problem in a finite Gaussian basis
- baseline projected energy: `0.226261`
- GONM final energy: `0.135305`
- energy gain of about `0.090956`
- baseline maximum normalization drift: `2.911990`

The archive also includes a relativistic boundary-value demonstration:

- geodesic-like Schwarzschild path linking two fixed endpoints
- local baseline best functional: `28.378418`
- GONM final functional: `28.179075`
- gain of about `0.199343`
- GONM optimizes the whole curve at once rather than integrating point by point

The archive also includes a control-engineering demonstration:

- inverted pendulum under strong wind gusts
- hand-tuned PD baseline objective: `1.098000`
- GONM tuned feedback objective: `0.320285`
- gain of about `0.777715`
- GONM reduces RMS angular deviation from `0.484950` to `0.154879`

The archive also includes an aerospace attitude-control demonstration:

- single-axis satellite pointing under noisy sensing and solar-pressure-like disturbances
- PID baseline objective: `18.604880`
- GONM tuned objective: `0.063012`
- gain of about `18.541869`
- RMS pointing error reduced from `0.850595` to `0.216987`

The archive also includes a smart-grid dispatch demonstration:

- reduced network with generation, solar, storage, and one transmission-line outage
- baseline redispatch objective: `122.850000`
- GONM redispatch objective: `103.861295`
- gain of about `18.988705`
- maximum line utilization reduced from `0.348684` to `0.295587`

The archive also includes a neural-compression demonstration:

- hidden-layer structured pruning with `10` candidate neurons
- magnitude-pruning objective: `0.454141`
- GONM pruning objective: `0.393141`
- gain of about `0.061000`
- validation accuracy improved from `0.823333` to `0.850000` with the same effective `4/10` active neurons

The organized archive now also includes a first visual `protein folding` demonstration:

- sequence length: `22`
- ambient dimension: `66`
- phase-1 energy: `19.0188`
- final energy: `7.7029`
- radius of gyration reduced from `3.7123` to `2.8627`

This example is intentionally coarse-grained, but it shows the same layered logic acting on a chain-constrained three-dimensional folding problem.

## Interpretation

The current evidence supports GONM not as a single universal optimizer, but as a layered theory of difficulty decomposition for noisy multimodal optimization, with both synthetic and physical support.
