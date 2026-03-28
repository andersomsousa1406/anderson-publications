# gonm_chaotic_crypto

This folder stores the exported result bundle for one `T07_GONM` experiment or simulation.

## Source

- Script or reference entry: `simulations/gonm_chaotic_crypto.py`

## Image

![gonm_chaotic_crypto](./gonm_chaotic_crypto.png)

## Files

- `summary.md`
- `summary.json`
- `gonm_chaotic_crypto.png`

## Result Summary

# GONM | Chaotic Contractive Synchronization

This simulation is an educational demonstration of contractive synchronization over a chaotic channel.

## Recorded outcome

- receiver BER: `0.399194`
- attacker BER: `0.383065`
- receiver final synchronization error: `1.747145e-01`
- attacker final synchronization error: `1.036201e-02`

## Interpretation

This is not a production cryptographic primitive and should not be used to secure real communications. The narrower point is conceptual: a contractive update law can let an authorized receiver stabilize onto the sender's internal dynamics while an outsider remains desynchronized.

