# GONM | Cosmological N-Body Stability

This simulation compares a naive gravitational N-body evolution against a layered GONM clustering search.

The gravitational setting is intentionally reduced, but it preserves the main difficulty:

- long-range attraction;
- numerical near-collisions at short distance;
- the need to keep clustering compact without collapsing into an ultraviolet singularity.

## Recorded outcome

- naive minimum pair distance: `0.003121`
- local baseline best functional: `-250.531471`
- GONM final functional: `-250.295655`
- GONM gain versus baseline: `-0.235816`

## Interpretation

This is not a million-body production cosmology code. The narrower demonstration is still meaningful: the GONM-style structural search can form a compact clustered configuration while maintaining a finite minimum separation, whereas a naive near-singular evolution rapidly approaches the ultraviolet catastrophe.
