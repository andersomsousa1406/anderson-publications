# GONM | Relativistic Geodesic Search

This simulation treats GONM as a variational search procedure for a Schwarzschild geodesic-like curve.

The objective is not a full exact relativistic integrator. Instead, it minimizes a trajectory functional built from:

- the residual of the reduced geodesic equation in `u(phi) = 1 / r(phi)`;
- the variance of the conserved first-integral proxy;
- barrier terms that prevent an artificial numerical fall through the horizon.

## Recorded outcome

- baseline local best functional: `255.944316`
- GONM final functional: `15.737340`
- GONM gain versus baseline: `240.206976`

## Final relativistic diagnostics

- GONM residual RMS: `1.128309`
- GONM invariant variance: `0.000349`
- GONM periastron: `4.0238 M`

## Interpretation

This is a reduced variational proxy, not a full Kerr or exact event-horizon integrator. The narrower claim is still meaningful: near the strong-field region, the layered GONM search can find a curve with lower geodesic residual and better invariant preservation than a purely local optimizer started from a poor trajectory family.
