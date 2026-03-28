# GONM | Satellite Attitude Control

This simulation treats GONM as a combined feedback-and-filter tuning mechanism for satellite attitude stabilization under sensor noise and disturbance torques.

## Recorded outcome

- PID baseline objective: `18.604880`
- GONM tuned objective: `0.063012`
- GONM gain versus PID: `18.541869`
- baseline RMS angle: `0.850595`
- GONM RMS angle: `0.216987`

## Interpretation

This is not a full flight-qualified ADCS stack. The narrower claim is still useful: the layered GONM search can tune a disturbance-rejecting attitude loop that filters noisy measurements and stabilizes pointing more tightly than a simple baseline PID law.
