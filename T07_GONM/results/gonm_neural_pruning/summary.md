# GONM | Neural Network Compression

This simulation treats GONM as a structured pruning optimizer for a compact neural network.

## Recorded outcome

- magnitude-pruning objective: `0.454141`
- GONM pruning objective: `0.393141`
- GONM gain versus magnitude: `0.061000`
- magnitude validation accuracy: `0.823333`
- GONM validation accuracy: `0.850000`

## Interpretation

This is not full LLM compression. The narrower claim is still useful: a GONM-style structured search can select a smaller hidden subnetwork with a better sparsity/accuracy tradeoff than a simple magnitude-pruning baseline.
