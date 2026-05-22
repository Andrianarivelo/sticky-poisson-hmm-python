# Sticky Poisson HMM Python

Python utilities for discovering latent states in neural population activity with a sticky Poisson Hidden Markov Model.

This package is designed for count-like observations:

- binned spike-count arrays
- arrays of spike trains
- a single spike-time array
- fiber photometry signals after converting transients to event counts

Brutally honest modeling rule: sticky-Poisson HMM is a Poisson count model. It is principled for spike counts and detected event counts. Do not feed raw continuous fiber photometry fluorescence directly into this model unless you first convert it to event counts. If you want to model raw continuous amplitudes, use a sticky Gaussian HMM instead.

## Install

From this repository:

```powershell
python -m pip install -e .
```

Then any script in the same Python environment can import:

```python
from hmm_spikes import fit_sticky_poisson_hmm
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

## Citation

This package is based on the sticky Poisson HMM methodology from:

Li T, La Camera G (2025) A sticky Poisson Hidden Markov Model for solving the problem of over-segmentation and rapid state switching in cortical datasets. PLOS One 20(7): e0325979. https://doi.org/10.1371/journal.pone.0325979



## Tutorial

See [tutorial.md](tutorial.md) for spike trains, single spike-time arrays, binned count arrays, fiber photometry event counts, and state-number selection.
