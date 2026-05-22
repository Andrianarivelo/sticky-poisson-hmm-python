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
    fit_dirichlet_poisson_hmm,
    fit_poisson_hmm,
    fit_sticky_poisson_hmm,
    model_log_likelihood,
    state_probabilities,
    stationary_distribution,
    viterbi_decode,
)
from .multinoulli import (
    MultinoulliHMMResult,
    counts_to_multinoulli_symbols,
    fit_multinoulli_hmm,
    multinoulli_state_probabilities,
    multinoulli_viterbi_decode,
)
from .gaussian import (
    GaussianHMMResult,
    fit_gaussian_hmm,
    fit_sticky_gaussian_hmm,
    gaussian_state_probabilities,
    gaussian_viterbi_decode,
)
from .selection import (
    BICScanResult,
    RestartRecord,
    poisson_hmm_bic,
    run_sticky_poisson_bic_scan_isolated,
)
from .analysis import (
    benjamini_hochberg_fdr,
    exact_sign_flip_pvalue,
    paired_condition_table,
    state_sequence_metrics,
)

__all__ = [
    "PHMMHistory",
    "PHMMResult",
    "SpikeDataset",
    "GaussianHMMResult",
    "MultinoulliHMMResult",
    "BICScanResult",
    "RestartRecord",
    "benjamini_hochberg_fdr",
    "counts_to_multinoulli_symbols",
    "exact_sign_flip_pvalue",
    "fit_dirichlet_poisson_hmm",
    "fit_gaussian_hmm",
    "fit_multinoulli_hmm",
    "fit_poisson_hmm",
    "fit_sticky_gaussian_hmm",
    "fit_sticky_poisson_hmm",
    "gaussian_state_probabilities",
    "gaussian_viterbi_decode",
    "load_mat_dataset",
    "load_npz_dataset",
    "model_log_likelihood",
    "multinoulli_state_probabilities",
    "multinoulli_viterbi_decode",
    "paired_condition_table",
    "poisson_hmm_bic",
    "run_sticky_poisson_bic_scan_isolated",
    "save_npz_dataset",
    "firings_to_counts",
    "signal_to_event_counts",
    "spike_count",
    "spike_times_to_counts",
    "spike_trains_to_counts",
    "state_probabilities",
    "state_sequence_metrics",
    "stationary_distribution",
    "viterbi_decode",
]
