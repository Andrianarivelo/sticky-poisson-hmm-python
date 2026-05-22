# Tutorial: Sticky Poisson HMM For Neural State Discovery

The sticky Poisson HMM is for segmenting neural activity into discrete latent states while discouraging unrealistically fast state switching. The key idea is simple: each hidden state has a vector of expected event counts across neurons, units, or channels, and the transition matrix is constrained to keep self-transition probabilities high.

Use the Poisson and Multinoulli models when your observations are counts or symbols in time bins.

Do not use raw continuous signals as Poisson observations. For fiber photometry, either convert transients to detected events and use sticky-Poisson HMM, or model the raw continuous trace with sticky Gaussian HMM.

## Install

From the package folder:

```powershell
python -m pip install -e .
```

Then your other scripts can call:

```python
from hmm_spikes import fit_sticky_poisson_hmm
```

## 1. Binned Spike-Count Array

If you already have a count matrix:

```python
import numpy as np
from hmm_spikes import fit_sticky_poisson_hmm, state_probabilities, viterbi_decode

dt = 0.05  # seconds
counts = np.asarray(your_counts)  # shape: n_neurons x n_time_bins

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

Outputs:

- `posterior`: probability of each state in each time bin, shape `n_states x n_time_bins`
- `posterior_states`: most probable state in each bin
- `viterbi_states`: most likely complete state sequence
- `result.rates_hz`: state firing rates or event rates, shape `n_neurons x n_states`
- `result.gamma`: transition probability matrix

State labels are zero-based in Python. Add 1 if you want MATLAB-style labels.

## 1b. The Four MATLAB-Style HMM Families

The Python package exposes direct public entry points for the four MATLAB demo families.

Standard Poisson HMM:

```python
from hmm_spikes import fit_poisson_hmm

standard = fit_poisson_hmm(
    trial_counts,
    n_states=3,
    bin_size=0.05,
    max_iter=1000,
)
```

Sticky Poisson HMM:

```python
from hmm_spikes import fit_sticky_poisson_hmm

sticky = fit_sticky_poisson_hmm(
    trial_counts,
    n_states=3,
    bin_size=0.05,
    threshold=0.8,
    max_iter=1000,
)
```

Poisson HMM with Dirichlet transition prior:

```python
from hmm_spikes import fit_dirichlet_poisson_hmm

dirichlet = fit_dirichlet_poisson_hmm(
    trial_counts,
    n_states=3,
    bin_size=0.05,
    mode_self_transition=0.9,
    offdiag_alpha=1.1,
    max_iter=1000,
)
```

Multinoulli HMM:

```python
from hmm_spikes import counts_to_multinoulli_symbols, fit_multinoulli_hmm

symbols = [
    counts_to_multinoulli_symbols(counts, random_state=0)
    for counts in trial_counts
]

multi = fit_multinoulli_hmm(
    symbols,
    n_states=3,
    n_symbols=trial_counts[0].shape[0] + 1,
    max_iter=1000,
)
```

Multinoulli warning: use short bins. The model assumes one categorical symbol per time bin: one selected unit fired, or no unit fired. If many neurons fire in the same bin, the symbol representation is throwing information away.

## 2. Multiple Spike Trains

Use this when each neuron has its own spike-time array.

```python
import numpy as np
from hmm_spikes import spike_trains_to_counts, fit_sticky_poisson_hmm

dt = 0.05
time_edges = np.arange(0.0, 30.0 + dt, dt)

spike_trains = [
    neuron_1_spike_times,
    neuron_2_spike_times,
    neuron_3_spike_times,
]

counts = spike_trains_to_counts(spike_trains, time_edges)

result = fit_sticky_poisson_hmm(
    [counts],
    n_states=3,
    bin_size=dt,
    threshold=0.8,
)
```

This gives a count matrix with shape:

```python
n_neurons x n_time_bins
```

## 3. One Single Spike-Time Array

You can train and decode one unit, although it is statistically weak compared with population data.

```python
import numpy as np
from hmm_spikes import spike_times_to_counts, fit_sticky_poisson_hmm, viterbi_decode

dt = 0.05
time_edges = np.arange(0.0, 60.0 + dt, dt)
spike_times = single_unit_spike_times

counts = spike_times_to_counts(spike_times, time_edges)

result = fit_sticky_poisson_hmm(
    [counts],
    n_states=2,
    bin_size=dt,
    threshold=0.8,
)

states = viterbi_decode(counts, result.means, result.gamma)
```

Brutal honesty: a single spike train can support simple low-rate versus high-rate segmentation. It cannot reliably support a large number of hidden states.

## 4. Two-Column Firing Matrix

If your spikes are stored as `[spike_time, neuron_id]`, with one-based neuron IDs:

```python
import numpy as np
from hmm_spikes import firings_to_counts, fit_sticky_poisson_hmm

dt = 0.05
time_edges = np.arange(-2.0, 5.0 + dt, dt)

counts = firings_to_counts(
    firings,
    time_edges,
    n_neurons=12,
)

result = fit_sticky_poisson_hmm(
    [counts],
    n_states=3,
    bin_size=dt,
    threshold=0.8,
)
```

## 5. Multiple Trials

Training is much better when you provide multiple trials:

```python
trial_counts = [
    counts_trial_1,
    counts_trial_2,
    counts_trial_3,
]

result = fit_sticky_poisson_hmm(
    trial_counts,
    n_states=3,
    bin_size=0.05,
    threshold=0.8,
    max_iter=1000,
)
```

Then decode each trial:

```python
from hmm_spikes import state_probabilities, viterbi_decode

for counts in trial_counts:
    posterior = state_probabilities(counts, result.means, result.gamma)
    states = viterbi_decode(counts, result.means, result.gamma)
```

## 6. Fiber Photometry

Fiber photometry is continuous. A Poisson HMM does not model raw fluorescence values. To use sticky-Poisson HMM, detect transients and turn them into event counts.

Single photometry signal:

```python
from hmm_spikes import signal_to_event_counts, fit_sticky_poisson_hmm, viterbi_decode

counts, time_edges, event_times = signal_to_event_counts(
    photometry_signal,
    sampling_rate=1000.0,
    bin_size=0.1,
    threshold_z=2.5,
    refractory=0.2,
)

result = fit_sticky_poisson_hmm(
    [counts],
    n_states=2,
    bin_size=0.1,
    threshold=0.8,
)

states = viterbi_decode(counts, result.means, result.gamma)
```

Array of photometry signals:

```python
counts, time_edges, event_times = signal_to_event_counts(
    photometry_signals,  # shape: n_signals x n_samples
    sampling_rate=1000.0,
    bin_size=0.1,
    threshold_z=2.5,
    refractory=0.2,
)

result = fit_sticky_poisson_hmm(
    [counts],
    n_states=3,
    bin_size=0.1,
    threshold=0.8,
)
```

Interpretation:

- each photometry channel becomes one row
- each detected transient contributes event counts
- `result.rates_hz` is an event rate, not a fluorescence amplitude

This is a modeling compromise. It is useful when transients are the event-like objects of interest. It is not a replacement for modeling raw fluorescence dynamics.

## 6b. Sticky Gaussian HMM For Raw Photometry Or Continuous Signals

If you want to model the raw continuous signal itself, use the sticky Gaussian HMM. This uses a diagonal Gaussian emission model, so each state has a mean and variance for each signal.

Single continuous signal:

```python
from hmm_spikes import fit_sticky_gaussian_hmm, gaussian_viterbi_decode

result = fit_sticky_gaussian_hmm(
    [photometry_signal],
    n_states=3,
    threshold=0.8,
    max_iter=1000,
    random_state=3456,
)

states = gaussian_viterbi_decode(
    photometry_signal,
    result.means,
    result.variances,
    result.gamma,
)
```

Array of continuous signals:

```python
from hmm_spikes import fit_sticky_gaussian_hmm, gaussian_state_probabilities

signals = photometry_signals  # shape: n_signals x n_time_bins

result = fit_sticky_gaussian_hmm(
    [signals],
    n_states=3,
    threshold=0.8,
    max_iter=1000,
)

posterior = gaussian_state_probabilities(
    signals,
    result.means,
    result.variances,
    result.gamma,
)
```

Interpretation:

- `result.means`: state-dependent signal amplitude means
- `result.variances`: state-dependent signal variances
- `result.gamma`: transition matrix
- `result.threshold_satisfied`: whether the final sticky constraint holds

Use sticky Gaussian HMM for raw continuous photometry. Use sticky Poisson HMM only after event detection and binning.

## 7. Choosing The Number Of States

Do not choose the state number by raw likelihood. Raw likelihood usually rewards too many states.

Use:

1. Candidate state numbers, for example `m = 2, 3, ..., 12`
2. Many random restarts per `m`, ideally 20 to 100
3. Keep only converged models with `threshold_satisfied == True`
4. Compute BIC
5. Inspect rasters, posterior confidence, state rates, and dwell times

```python
import numpy as np
from hmm_spikes import fit_sticky_poisson_hmm

def bic(log_likelihood, m, n_neurons, total_bins):
    n_params = m * (m - 1) + n_neurons * m
    return -2 * log_likelihood + n_params * np.log(total_bins)

results = []

for m in range(2, 13):
    best = None

    for seed in range(30):
        result = fit_sticky_poisson_hmm(
            trial_counts,
            n_states=m,
            bin_size=0.05,
            threshold=0.8,
            max_iter=1000,
            random_state=seed,
        )

        if not result.converged:
            continue
        if not result.threshold_satisfied:
            continue

        if best is None or result.log_likelihood > best.log_likelihood:
            best = result

    if best is None:
        print(f"m={m}: no valid model")
        continue

    n_neurons = trial_counts[0].shape[0]
    total_bins = sum(x.shape[1] for x in trial_counts)
    score = bic(best.log_likelihood, m, n_neurons, total_bins)
    results.append((m, score, best))
    print(f"m={m}, BIC={score:.2f}, LL={best.log_likelihood:.2f}")

best_m, best_bic, best_model = min(results, key=lambda x: x[1])
print("Best state number:", best_m)
```

## 8. Citation

This package implements the sticky Poisson HMM workflow introduced by:

Li T, La Camera G (2025) A sticky Poisson Hidden Markov Model for solving the problem of over-segmentation and rapid state switching in cortical datasets. PLOS One 20(7): e0325979. https://doi.org/10.1371/journal.pone.0325979

PLOS article page:

https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0325979

## Wrap-Up

Use sticky-Poisson HMM when your observations are event counts. Spike trains naturally become counts. Single spike-time arrays can be counted too, but support only simple models. Fiber photometry must be converted into event counts first. The sticky transition constraint protects you from one of the most common HMM failures in neural data: fake rapid switching caused by weak self-transition probabilities.
