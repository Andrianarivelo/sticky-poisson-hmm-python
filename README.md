# Sticky Poisson HMM Python

Python utilities for discovering latent states in neural population activity with Hidden Markov Models.

The package includes Python counterparts of the original MATLAB demo families:

- standard Poisson HMM
- sticky Poisson HMM
- Poisson HMM with Dirichlet prior over transition rows
- Multinoulli HMM
- sticky Gaussian HMM for continuous signals

The Poisson and Multinoulli models are designed for count-like observations:

- binned spike-count arrays
- arrays of spike trains
- a single spike-time array
- fiber photometry signals after converting transients to event counts

Brutally honest modeling rule: sticky-Poisson HMM is a Poisson count model. It is principled for spike counts and detected event counts. Do not feed raw continuous fiber photometry fluorescence directly into the Poisson model unless you first convert it to event counts. If you want to model raw continuous amplitudes, use `fit_sticky_gaussian_hmm` instead.

## Install

From this repository:

```powershell
python -m pip install -e .
```

Then any script in the same Python environment can import:

```python
from hmm_spikes import fit_sticky_poisson_hmm, fit_sticky_gaussian_hmm
```

## Minimal Use With A Count Array

```python
from hmm_spikes import fit_sticky_poisson_hmm, state_probabilities, viterbi_decode

dt = 0.05
counts = your_counts_array  # shape: n_neurons x n_time_bins

result = fit_sticky_poisson_hmm(
    [counts],
    n_states=3,
    bin_size=dt,
    threshold=0.8,
    max_iter=1000,
    random_state=3456,
)

posterior = state_probabilities(counts, result.means, result.gamma)
posterior_states = posterior.argmax(axis=0)
viterbi_states = viterbi_decode(counts, result.means, result.gamma)
```

## Algorithm Map

```python
from hmm_spikes import (
    fit_poisson_hmm,
    fit_sticky_poisson_hmm,
    fit_dirichlet_poisson_hmm,
    fit_multinoulli_hmm,
    fit_sticky_gaussian_hmm,
)
```

Use `fit_poisson_hmm` for the standard PHMM, `fit_sticky_poisson_hmm` for the recommended sticky count model, `fit_dirichlet_poisson_hmm` when you want a soft transition prior rather than a hard sticky reset, `fit_multinoulli_hmm` for categorical symbols, and `fit_sticky_gaussian_hmm` for continuous traces such as raw photometry.

## Reliability Note

The implementation does not depend on an external HMM backend. The forward-backward and EM updates are implemented in this package. Transition expectations are accumulated without constructing a large `n_states x n_states x n_time_bins` tensor, which is important for long recordings.

For unstable model-selection sweeps, use the crash-isolated scanner:

```powershell
python scripts\run_bic_scan.py `
  --dataset data\python\exampledata_dt50ms.npz `
  --output-dir figures\bic_scan `
  --states 2..6 `
  --restarts 20 `
  --max-iter 1000
```

Each restart runs in a fresh Python subprocess. A native crash is recorded as `crashed` and the scan continues. Fits that satisfy the sticky threshold but do not hit strict EM convergence are labeled `diagnostic`.

## Citation

This package is based on the sticky Poisson HMM methodology from:

Li T, La Camera G (2025) A sticky Poisson Hidden Markov Model for solving the problem of over-segmentation and rapid state switching in cortical datasets. PLOS One 20(7): e0325979. https://doi.org/10.1371/journal.pone.0325979



## Tutorial

See [tutorial.md](tutorial.md) for spike trains, single spike-time arrays, binned count arrays, fiber photometry event counts, and state-number selection.
