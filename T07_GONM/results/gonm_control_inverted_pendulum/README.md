# gonm_control_inverted_pendulum

This folder stores the exported result bundle for one `T07_GONM` experiment or simulation.

## Source

- Script or reference entry: `simulations/gonm_control_inverted_pendulum.py`

## Image

![gonm_control_inverted_pendulum](./gonm_control_inverted_pendulum.png)

## Files

- `summary.md`
- `summary.json`
- `gonm_control_inverted_pendulum.png`

## Result Summary

# GONM | Inverted Pendulum Feedback Tuning

This simulation treats GONM as a robust feedback-gain tuner for an inverted pendulum under strong wind disturbances.

## Recorded outcome

- PD baseline objective: `1.098000`
- GONM tuned objective: `0.320285`
- GONM gain versus PD: `0.777715`
- baseline RMS angle: `0.484950`
- GONM RMS angle: `0.154879`

## Tuned gains

- `kp = 34.0000`
- `kd = 12.0000`
- `ki = 8.0000`
- `leak = 3.3865`

## Interpretation

This is not a full industrial MPC stack. The narrower claim is still useful: the layered GONM search can tune a feedback law that contracts the disturbed pendulum back toward equilibrium better than a simple hand-tuned PD controller.

