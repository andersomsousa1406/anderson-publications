# T06_CSD_V4 results

This folder stores the recorded numerical artifacts currently attached to `T06_CSD_V4`.

## Source

- Experiment: `experiments/t06_multichart_consistency_test.py`
- Simulation: `simulations/t06_transition_simulation.py`

## Files

- `summary.md`
- `t06_multichart_consistency_test.json`
- `t06_transition_simulation.json`

## Result Summary

# CSD V4 Results Summary

This summary records the strongest compact evidence attached to the `V4` manuscript.

## Current V2 / V3 / V4 comparison

### clean_final_multiseed
- Cases: 30
- Mean error: `V2 = 6.429e-01`, `V3_quad = 6.097e-01`, `V3_cubic = 3.549e-01`, `V4 = 2.364e-01`
- Wins: `V3_quad > V2 = 30/30`, `V3_cubic > V2 = 30/30`, `V4 > V3_quad = 30/30`, `V4 > V3_cubic = 24/30`
- Transition: `V3_quad -> V4 = 63.33%`, `V3_cubic -> V4 = 6.67%`

### observational_degraded_final
- Cases: 20
- Mean error: `V2 = 6.636e-01`, `V3_quad = 6.329e-01`, `V3_cubic = 3.603e-01`, `V4 = 2.504e-01`
- Wins: `V4 > V3_quad = 20/20`, `V4 > V3_cubic = 19/20`
- Transition: `V3_quad -> V4 = 45.00%`, `V3_cubic -> V4 = 10.00%`

### stress_mixed_final
- Cases: 40
- Mean error: `V2 = 8.936e-01`, `V3_quad = 8.675e-01`, `V3_cubic = 5.893e-01`, `V4 = 3.587e-01`
- Wins: `V4 > V3_quad = 39/40`, `V4 > V3_cubic = 38/40`
- Transition: `V3_quad -> V4 = 57.50%`, `V3_cubic -> V4 = 17.50%`

## Ontological transition validation against V3

- Cases: 240
- Mean error: `V3 quadratic = 2.493e+00`, `V3 cubic = 1.649e+00`, `V4 = 6.031e-01`
- `V4` beats `V3 quadratic`: `240/240`
- `V4` beats `V3 cubic`: `228/240`
- Transition from `V3 quadratic`: `79.17%`
- Transition from `V3 cubic`: `46.25%`
- `V4` convergence: `24.17%`
- `V4` near-convergence: `75.83%`
- `V4` iterative stability: `0.735`
- `V4 -> V3` fallback: `0.00%`

## Interpretation

The current evidence supports `V4` as the coherent multichart regime beyond `V3`. In the present validation suite, it systematically improves over the stable quadratic `V3` layer and very frequently improves over the stronger cubic `V3` extension, while also providing an explicit ontological transition criterion rather than relying on heuristic chart multiplication.
