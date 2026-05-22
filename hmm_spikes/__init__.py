"""Sticky Poisson HMM tools for neural spike-count data."""

from .data import (
    SpikeDataset,
    firings_to_counts,
    load_mat_dataset,
    load_npz_dataset,
    save_npz_dataset,
    signal_to_event_counts,
    spike_count,
    spike_times_to_counts,
    spike_trains_to_counts,
)
from .phmm import (
    PHMMHistory,
    PHMMResult,
    fit_sticky_poisson_hmm,
    model_log_likelihood,
    state_probabilities,
    stationary_distribution,
    viterbi_decode,
)

__all__ = [
    "PHMMHistory",
    "PHMMResult",
    "SpikeDataset",
    "fit_sticky_poisson_hmm",
    "load_mat_dataset",
    "load_npz_dataset",
    "model_log_likelihood",
    "save_npz_dataset",
    "firings_to_counts",
    "signal_to_event_counts",
    "spike_count",
    "spike_times_to_counts",
    "spike_trains_to_counts",
    "state_probabilities",
    "stationary_distribution",
    "viterbi_decode",
]
