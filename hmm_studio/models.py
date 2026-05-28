"""Model registry: a uniform interface over every ``hmm_spikes`` HMM family.

Each :class:`MethodSpec` knows how to:
  * declare the data kind it consumes (counts / continuous / symbols),
  * prepare a dataset's canonical trials into model-ready inputs,
  * fit the model,
  * decode a single trial (posterior + Viterbi),
  * expose the per-state emission profile and a BIC parameter count.

This keeps the GUI completely generic: add a method here and it shows up
everywhere (fit panel, BIC scan, comparison) with no further wiring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

import hmm_spikes as H


# ----------------------------------------------------------------------------
# Parameter schema
# ----------------------------------------------------------------------------
@dataclass
class ParamSpec:
    name: str
    label: str
    kind: str                # 'int' | 'float' | 'bool'
    default: Any
    minimum: float = 0.0
    maximum: float = 1.0
    step: float = 0.1
    decimals: int = 3
    help: str = ""


# ----------------------------------------------------------------------------
# Prepared model inputs (canonical trials -> model representation)
# ----------------------------------------------------------------------------
@dataclass
class Prepared:
    trials: list[np.ndarray]          # model-ready per-trial arrays
    n_units: int                      # neurons / features / symbols for BIC
    bin_size: float
    n_symbols: int | None = None
    extra: dict = field(default_factory=dict)


DataKind = str  # 'counts' | 'continuous' | 'symbols'


@dataclass
class MethodSpec:
    key: str
    label: str
    family: str                       # 'poisson' | 'gaussian' | 'multinoulli'
    requires: DataKind                # canonical data kind needed
    description: str
    params: list[ParamSpec]
    sticky: bool = False
    _fit: Callable = None
    _decode: Callable = None
    _profile: Callable = None
    _nparams: Callable = None

    # -- preparation -------------------------------------------------------
    def prepare(self, trials: list[np.ndarray], kind: str, bin_size: float,
                rng: np.random.Generator) -> Prepared:
        if self.family == "multinoulli":
            if kind != "counts":
                raise ValueError("Multinoulli models need count data.")
            n_units = int(trials[0].shape[0])
            n_symbols = n_units + 1
            syms = [H.counts_to_multinoulli_symbols(np.asarray(t), random_state=rng)
                    for t in trials]
            return Prepared(trials=syms, n_units=n_symbols, bin_size=bin_size,
                            n_symbols=n_symbols)
        # poisson / gaussian operate on the 2D arrays directly
        clean = [np.asarray(t, dtype=float) for t in trials]
        return Prepared(trials=clean, n_units=int(clean[0].shape[0]),
                        bin_size=bin_size)

    # -- fit ---------------------------------------------------------------
    def fit(self, prep: Prepared, n_states: int, params: dict,
            random_state: int):
        return self._fit(self, prep, n_states, params, random_state)

    # -- decode a single prepared trial -----------------------------------
    def decode(self, result, trial_array: np.ndarray):
        return self._decode(result, trial_array)

    # -- per-state emission profile ---------------------------------------
    def profile(self, result, bin_size: float):
        return self._profile(result, bin_size)

    # -- free parameter count for BIC -------------------------------------
    def n_params(self, n_states: int, n_units: int) -> int:
        return self._nparams(n_states, n_units)

    def compatible(self, kind: str) -> bool:
        if self.requires == "counts":
            return kind == "counts"
        if self.requires == "symbols":
            return kind == "counts"
        return True  # gaussian works on any numeric series


# ----------------------------------------------------------------------------
# Shared fit / decode / profile helpers per family
# ----------------------------------------------------------------------------
def _poisson_decode(result, x):
    post = H.state_probabilities(x, result.means, result.gamma)
    vit = H.viterbi_decode(x, result.means, result.gamma)
    return post, vit


def _poisson_profile(result, bin_size):
    return result.rates_hz, "Rate (Hz)"


def _gaussian_decode(result, x):
    post = H.gaussian_state_probabilities(x, result.means, result.variances, result.gamma)
    vit = H.gaussian_viterbi_decode(x, result.means, result.variances, result.gamma)
    return post, vit


def _gaussian_profile(result, bin_size):
    return result.means, "State mean"


def _multinoulli_decode(result, s):
    post = H.multinoulli_state_probabilities(s, result.emissions, result.gamma)
    vit = H.multinoulli_viterbi_decode(s, result.emissions, result.gamma)
    return post, vit


def _multinoulli_profile(result, bin_size):
    return result.emissions.T, "Emission prob."


def _np_poisson(K, n):
    return K * (K - 1) + K * n


def _np_gaussian(K, n):
    return K * (K - 1) + 2 * K * n


def _np_multinoulli(K, n_symbols):
    return K * (K - 1) + K * (n_symbols - 1)


# ----------------------------------------------------------------------------
# Concrete fit wrappers (translate generic params -> hmm_spikes kwargs)
# ----------------------------------------------------------------------------
def _fit_poisson(spec, prep, K, p, rs):
    fn = H.fit_sticky_poisson_hmm if spec.sticky else H.fit_poisson_hmm
    kw = dict(bin_size=prep.bin_size, max_iter=int(p["max_iter"]), random_state=rs)
    if spec.sticky:
        kw["threshold"] = float(p["threshold"])
    return fn(prep.trials, K, **kw)


def _fit_dirichlet(spec, prep, K, p, rs):
    return H.fit_dirichlet_poisson_hmm(
        prep.trials, K, bin_size=prep.bin_size,
        mode_self_transition=float(p["mode_self_transition"]),
        offdiag_alpha=float(p["offdiag_alpha"]),
        max_iter=int(p["max_iter"]), random_state=rs,
    )


def _fit_gaussian(spec, prep, K, p, rs):
    fn = H.fit_sticky_gaussian_hmm if spec.sticky else H.fit_gaussian_hmm
    kw = dict(max_iter=int(p["max_iter"]), random_state=rs)
    if spec.sticky:
        kw["threshold"] = float(p["threshold"])
    return fn(prep.trials, K, **kw)


def _fit_multinoulli(spec, prep, K, p, rs):
    return H.fit_multinoulli_hmm(
        prep.trials, K, n_symbols=prep.n_symbols,
        max_iter=int(p["max_iter"]), random_state=rs,
    )


def _fit_graph_poisson(spec, prep, K, p, rs):
    top_k = int(p["graph_top_k"]) or None
    return H.fit_sticky_graph_poisson_hmm(
        prep.trials, K, bin_size=prep.bin_size,
        graph_strength=float(p["graph_strength"]),
        graph_top_k=top_k, threshold=float(p["threshold"]),
        max_iter=int(p["max_iter"]), random_state=rs, sticky=spec.sticky,
    )


def _fit_graph_gaussian(spec, prep, K, p, rs):
    top_k = int(p["graph_top_k"]) or None
    return H.fit_sticky_graph_gaussian_hmm(
        prep.trials, K, graph_strength=float(p["graph_strength"]),
        variance_graph_strength=float(p["variance_graph_strength"]),
        graph_top_k=top_k, threshold=float(p["threshold"]),
        max_iter=int(p["max_iter"]), random_state=rs, sticky=spec.sticky,
    )


def _fit_graph_multinoulli(spec, prep, K, p, rs):
    return H.fit_graph_multinoulli_hmm(
        prep.trials, K, n_symbols=prep.n_symbols,
        graph_strength=float(p["graph_strength"]),
        graph_window=int(p["graph_window"]),
        max_iter=int(p["max_iter"]), random_state=rs,
    )


# ----------------------------------------------------------------------------
# Common parameter atoms
# ----------------------------------------------------------------------------
def _p_maxiter(default=300):
    return ParamSpec("max_iter", "Max EM iterations", "int", default, 20, 5000, 10, 0,
                     "Upper bound on Baum-Welch iterations per fit.")


def _p_threshold():
    return ParamSpec("threshold", "Sticky threshold", "float", 0.8, 0.0, 0.999, 0.01, 3,
                     "Minimum self-transition probability the model must keep.")


def _p_graph_strength():
    return ParamSpec("graph_strength", "Graph strength α", "float", 0.15, 0.0, 5.0, 0.05, 3,
                     "Laplacian smoothing weight on the emission parameters.")


def _p_topk():
    return ParamSpec("graph_top_k", "Graph top-k (0=auto)", "int", 6, 0, 64, 1, 0,
                     "Per-node neighbor pruning when inferring the unit graph.")


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------
def _make() -> dict[str, MethodSpec]:
    specs: list[MethodSpec] = [
        MethodSpec(
            "sticky_poisson", "Sticky Poisson HMM", "poisson", "counts",
            "Recommended count model. High self-transitions stop fake rapid switching.",
            [_p_threshold(), _p_maxiter()], sticky=True,
            _fit=_fit_poisson, _decode=_poisson_decode, _profile=_poisson_profile,
            _nparams=_np_poisson),
        MethodSpec(
            "poisson", "Standard Poisson HMM", "poisson", "counts",
            "Classic Poisson HMM with no stickiness constraint.",
            [_p_maxiter()], sticky=False,
            _fit=_fit_poisson, _decode=_poisson_decode, _profile=_poisson_profile,
            _nparams=_np_poisson),
        MethodSpec(
            "dirichlet_poisson", "Dirichlet Poisson HMM", "poisson", "counts",
            "Soft Dirichlet prior nudges self-transitions up without a hard reset.",
            [ParamSpec("mode_self_transition", "Mode self-transition", "float", 0.9,
                       0.01, 0.99, 0.01, 3, "Prior mode for the diagonal."),
             ParamSpec("offdiag_alpha", "Off-diagonal α", "float", 1.1, 1.001, 5.0,
                       0.05, 3, "Dirichlet concentration off the diagonal (>1)."),
             _p_maxiter()],
            _fit=_fit_dirichlet, _decode=_poisson_decode, _profile=_poisson_profile,
            _nparams=_np_poisson),
        MethodSpec(
            "sticky_graph_poisson", "Sticky graph-Poisson HMM", "poisson", "counts",
            "Sticky Poisson with a unit graph that smooths state rate maps.",
            [_p_threshold(), _p_graph_strength(), _p_topk(), _p_maxiter()], sticky=True,
            _fit=_fit_graph_poisson, _decode=_poisson_decode, _profile=_poisson_profile,
            _nparams=_np_poisson),
        MethodSpec(
            "sticky_gaussian", "Sticky Gaussian HMM", "gaussian", "continuous",
            "For raw continuous signals (e.g. fiber photometry traces).",
            [_p_threshold(), _p_maxiter()], sticky=True,
            _fit=_fit_gaussian, _decode=_gaussian_decode, _profile=_gaussian_profile,
            _nparams=_np_gaussian),
        MethodSpec(
            "gaussian", "Standard Gaussian HMM", "gaussian", "continuous",
            "Diagonal-covariance Gaussian HMM, no stickiness.",
            [_p_maxiter()], sticky=False,
            _fit=_fit_gaussian, _decode=_gaussian_decode, _profile=_gaussian_profile,
            _nparams=_np_gaussian),
        MethodSpec(
            "sticky_graph_gaussian", "Sticky graph-Gaussian HMM", "gaussian", "continuous",
            "Sticky Gaussian with a feature graph smoothing state mean maps.",
            [_p_threshold(), _p_graph_strength(),
             ParamSpec("variance_graph_strength", "Variance graph β", "float", 0.0,
                       0.0, 5.0, 0.05, 3, "Optional Laplacian smoothing on variances."),
             _p_topk(), _p_maxiter()], sticky=True,
            _fit=_fit_graph_gaussian, _decode=_gaussian_decode, _profile=_gaussian_profile,
            _nparams=_np_gaussian),
        MethodSpec(
            "multinoulli", "Multinoulli HMM", "multinoulli", "symbols",
            "Categorical model: one symbol per bin (which unit fired, or none).",
            [_p_maxiter()],
            _fit=_fit_multinoulli, _decode=_multinoulli_decode, _profile=_multinoulli_profile,
            _nparams=_np_multinoulli),
        MethodSpec(
            "graph_multinoulli", "Graph-Multinoulli HMM", "multinoulli", "symbols",
            "Multinoulli with a symbol graph smoothing emission probabilities.",
            [_p_graph_strength(),
             ParamSpec("graph_window", "Symbol lag window", "int", 1, 1, 10, 1, 0,
                       "Temporal lag used to infer the symbol adjacency."),
             _p_maxiter()],
            _fit=_fit_graph_multinoulli, _decode=_multinoulli_decode,
            _profile=_multinoulli_profile, _nparams=_np_multinoulli),
    ]
    return {s.key: s for s in specs}


REGISTRY: dict[str, MethodSpec] = _make()


def methods_for_kind(kind: str) -> list[MethodSpec]:
    return [s for s in REGISTRY.values() if s.compatible(kind)]


def get(key: str) -> MethodSpec:
    return REGISTRY[key]
