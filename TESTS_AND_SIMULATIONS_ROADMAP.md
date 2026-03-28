# Tests and Simulations Roadmap

This roadmap starts the systematic validation program of the publication archive. For each theory `T01` through `T07`, I list:

- `10 tests`: targeted checks of mathematical, numerical, or structural claims
- `10 simulations`: executable demonstrations, parameter sweeps, or visual experiments

The purpose of this document is to turn the archive into a reproducible research program rather than a collection of manuscripts alone.

## T01_CDRCGM

### Tests
1. Contraction-factor verification on the invariant interval.
2. Fixed-point uniqueness check on the geometric interval.
3. Numerical confirmation of monotonicity of the transformed map.
4. Perturbation-decay rate versus the asymptotic factor `1/4`.
5. Stability under floating-point perturbations of the recurrence.
6. Agreement between geometric and transformed-variable formulations.
7. Lower/upper bound convergence for the `pi` construction.
8. Richardson extrapolation gain versus the raw recurrence.
9. Residual-series convergence check.
10. Sensitivity of convergence rate to initial sector discretization.

### Simulations
1. Orbit plot of the transformed recurrence.
2. Cobweb diagram toward the fixed point.
3. Perturbation-decay animation across iterations.
4. Comparison between exact and floating-point trajectories.
5. Convergence of lower and upper `pi` bounds.
6. Error-decay plot in logarithmic scale.
7. Richardson acceleration comparison.
8. Residual-series partial sums versus `pi`.
9. Derivative profile of the transformed map.
10. Geometric sector refinement visualization.

## T02_CPP

### Tests
1. Contractive propagation on synthetic scalar maps.
2. Contractive propagation on linear iterative systems.
3. Noise amplification versus contraction-factor comparison.
4. Sparse tomography perturbation damping.
5. Self-consistent map stabilization test.
6. Sensitivity to step-size variations.
7. Robustness under random initialization.
8. Comparison with non-contractive baselines.
9. Error recursion validation against theoretical bound.
10. Multi-seed stability summary.

### Simulations
1. Perturbation evolution on a scalar contractive map.
2. Error-field visualization for iterative reconstruction.
3. Contractive versus expansive regime comparison.
4. Sparse tomography reconstruction movie.
5. Self-consistent map convergence trajectories.
6. Stability heatmap over contraction and noise.
7. Basin of stable convergence under perturbations.
8. Local versus global damping comparison.
9. Contraction-factor sweep dashboard.
10. Aggregate stability report over seeds.

## T03_MQLM

### Tests
1. Exactness on exponential and log-linear families.
2. Positivity preservation under multiplicative interpolation.
3. Order-of-accuracy comparison versus trapezoidal rule.
4. Order-of-accuracy comparison versus Simpson rule.
5. Stability on steep positive integrands.
6. Error profile on oscillatory positive functions.
7. Sensitivity to mesh refinement.
8. Robustness under scaling of the integrand.
9. Benchmark-suite regression check.
10. Multidimensional extension sanity tests.

### Simulations
1. Log-linear interpolation visualization.
2. Error curves across mesh sizes.
3. Multiplicative versus additive quadrature comparison.
4. Positive integrand family sweep.
5. Benchmark dashboard across classical rules.
6. Stability under extreme dynamic range.
7. Local quadrature contribution heatmap.
8. Scaling behavior under multiplicative rescaling.
9. Integrand-family gallery.
10. Multidimensional prototype experiment.

## T04_CSD

### Tests
1. Affine-mode separation recovery on synthetic densities.
2. Translation invariance check.
3. Mass preservation check.
4. Linear deformation extraction accuracy.
5. Residual-shape isolation consistency.
6. Orthogonality of decomposed modes.
7. Stability under small perturbations of the measure.
8. Comparison with undecomposed representations.
9. Gauge consistency under equivalent affine descriptions.
10. Reproducibility across seeds and discretizations.

### Simulations
1. Affine deformation decomposition movie.
2. Translation/shape separation visualization.
3. Mode-energy evolution through time.
4. Residual-shape heatmaps.
5. Recovery of known affine transformations.
6. Comparative visualization of decomposed versus raw densities.
7. Synthetic measure evolution gallery.
8. Orthogonality diagnostic plots.
9. Affine-parameter sweep.
10. Aggregate structural summary dashboard.

## T05_CSD_V3

### Tests
1. Single-chart nonlinear recovery on synthetic densities.
2. Penalized quadratic functional validation.
3. Chart adequacy versus distortion test.
4. Comparison against affine `V2` baseline.
5. Residual stability under nonlinear warping.
6. Gauge consistency within the single-chart ontology.
7. Robustness across seeds and discretization levels.
8. Observational validation on benchmark families.
9. Failure-mode detection near chart breakdown.
10. Cross-validation against current `V4` frontier.

### Simulations
1. Single-chart nonlinear structural evolution movie.
2. Distortion growth across the chart domain.
3. `V2` versus `V3` qualitative comparison.
4. Penalization strength sweep.
5. Nonlinear residual-shape gallery.
6. Gauge-comparison visualization.
7. Observational-validation summary plots.
8. Robustness heatmap over seeds and parameters.
9. Transition-to-breakdown diagnostics.
10. Structural-fit dashboard.

## T06_CSD_V4

### Tests
1. Multichart recovery on composite synthetic densities.
2. Transition criterion validation from `V3` to `V4`.
3. External-gauge stability test.
4. Fusion-rule correctness under overlapping charts.
5. Competitive reduced-class validation.
6. Saddle-guided restart effectiveness.
7. Comparison with `V3` on multicomponent cases.
8. Robustness under chart-count variation.
9. Stability of calibrated `tau_k`.
10. Multi-seed reproducibility summary.

### Simulations
1. Multichart decomposition movie.
2. `V3` to `V4` transition visualization.
3. Gauge-alignment animation.
4. Fusion behavior over overlapping components.
5. Restart trajectories on saddle-dominated cases.
6. Chart-count sweep dashboard.
7. Reduced-class competition plots.
8. Calibration plots for `tau_k`.
9. Multicomponent density gallery.
10. End-to-end `V4` structural report.

## T07_GONM

### Tests
1. Structural-delivery quality under noisy multimodal benchmarks.
2. Contractive local refinement quality after basin delivery.
3. MQLM path-filtering stability under noisy trajectories.
4. Terminal repeated-average closure versus differential closure.
5. Composed-operator stability bound sanity check.
6. Ackley resolution-barrier reproduction.
7. Dimensional wall-coefficient regression.
8. Budget-density transition test.
9. LJ-12 energy-improvement reproducibility.
10. LJ-38 continuation and escape-freeze reproducibility.

### Simulations
1. 2D benchmark comparison dashboard.
2. Ackley terminal-resolution animation.
3. Dimensional sweep plots from `1D` to `1000D`.
4. Structural-wall coefficient evolution.
5. Budget-allocation sweep visualization.
6. LJ-12 relaxation movie.
7. LJ-12 cooling versus fixed-noise comparison.
8. LJ-38 continuation lineage plot.
9. LJ-38 escape-freeze branch comparison.
10. CPU-only physical demonstration dashboard.

## Initial implementation status

The archive starts this program with `T01_CDRCGM`, where the first executable experiment and the first simulation are introduced as the seed of the broader validation suite.
