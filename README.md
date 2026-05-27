# Sticky Poisson HMM Python

Python utilities for discovering latent states in neural population activity with Hidden Markov Models.

The package includes Python counterparts of the original MATLAB demo families plus graph-informed extensions:

- standard Poisson HMM
- sticky Poisson HMM
- Poisson HMM with Dirichlet prior over transition rows
- graph-informed Poisson, Gaussian, and Multinoulli HMMs with complex-network smoothing
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
    fit_sticky_graph_poisson_hmm,
    fit_sticky_graph_gaussian_hmm,
    fit_graph_multinoulli_hmm,
    fit_multinoulli_hmm,
    fit_sticky_gaussian_hmm,
)
```

Use `fit_poisson_hmm` for the standard PHMM, `fit_sticky_poisson_hmm` for the recommended sticky count model, `fit_dirichlet_poisson_hmm` when you want a soft transition prior rather than a hard sticky reset, `fit_multinoulli_hmm` for categorical symbols, and `fit_sticky_gaussian_hmm` for continuous traces such as raw photometry. Use the graph-informed variants when a unit, channel, feature, or symbol graph should regularize the emission parameters.

## Graph-Informed HMM Family

The graph-informed family uses a weighted graph over observed units, continuous features, or categorical symbols. During each M-step, the ordinary emission estimate is smoothed with a graph Laplacian. For Poisson this smooths state count-rate maps. For Gaussian this smooths state mean maps, with optional variance smoothing. For Multinoulli this smooths categorical emission probabilities over a symbol graph.

This is useful when the graph represents physical proximity, known connectivity, channel layout, anatomical grouping, symbol similarity, or a functional connectivity estimate.

```python
from hmm_spikes import (
    fit_sticky_graph_poisson_hmm,
    fit_sticky_graph_gaussian_hmm,
    fit_graph_multinoulli_hmm,
    infer_functional_connectivity_graph,
    infer_observation_graph,
    infer_symbol_transition_graph,
)

graph = infer_functional_connectivity_graph(
    trial_counts,
    threshold_quantile=0.75,
    top_k=6,
)

result = fit_sticky_graph_poisson_hmm(
    trial_counts,
    n_states=3,
    bin_size=0.05,
    adjacency=graph,
    graph_strength=0.2,
    threshold=0.8,
    max_iter=1000,
    random_state=3456,
)

gaussian_graph = infer_observation_graph(continuous_trials, top_k=6)
gaussian_result = fit_sticky_graph_gaussian_hmm(
    continuous_trials,
    n_states=3,
    adjacency=gaussian_graph,
    graph_strength=0.2,
    threshold=0.8,
    max_iter=1000,
)

symbol_graph = infer_symbol_transition_graph(symbol_trials, n_symbols=n_symbols)
symbol_result = fit_graph_multinoulli_hmm(
    symbol_trials,
    n_states=3,
    n_symbols=n_symbols,
    adjacency=symbol_graph,
    graph_strength=0.1,
    max_iter=1000,
)
```

Brutal honesty: this is a structured regularizer, not proof of causal connectivity. If the inferred graph is garbage, the model will faithfully regularize toward garbage. Use held-out likelihood, BIC, posterior diagnostics, and biological interpretability before trusting it.

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
