# T02_CPP

This folder contains the second theory in my publication archive: `CPP`, short for `Contractive Propagation of Perturbations`.

In this work, I treat `CPP` not as a universal algorithm, but as a stability principle for iterative problems in which local perturbations, noise, or ill-conditioning can dominate numerical behavior. The article focuses on two branches where the idea became most convincing for me:

- sparse and noisy tomographic reconstruction;
- self-consistent maps of `SCF` type.

## Article

- Main English source: `article/Contractive Propagation of Perturbations as a Principle of Stability in Iterative Reconstruction and Self-Consistent Maps.tex`
- Associated PDF, when available: `article/Contractive Propagation of Perturbations as a Principle of Stability in Iterative Reconstruction and Self-Consistent Maps.pdf`
- Zenodo record: `https://zenodo.org/records/19289615`

## Structure

- `article/`: the main manuscript in English, with the canonical title for publication.
- `experiments/`: experimental notes and protocol summary associated with the article.
- `simulations/`: simulation notes for this theory.
- `results/`: compact result summaries extracted from the confirmed evidence reported in the manuscript.

## Status

This theory is already organized as a publication unit. The English manuscript is the primary article in this folder.

## Initial validation suite

This theory now includes the first executable validation seed of the archive for `CPP`:

- `experiments/t02_cpp_contraction_test.py`
- `simulations/t02_cpp_regime_comparison.py`

These files start the broader plan of building ten tests and ten simulations for this theory.
