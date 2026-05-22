from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .phmm import PHMMHistory, _as_rng, _initialize_transition, _normalize_rows, forward_backward_scaled, stationary_distribution


@dataclass
class MultinoulliHMMResult:
    emissions: np.ndarray
    gamma: np.ndarray
    deltas: np.ndarray
    converged: bool
    n_iter: int
    log_likelihood: float
    history: PHMMHistory


def counts_to_multinoulli_symbols(
    counts: np.ndarray,
    *,
    random_state: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Convert binned counts to zero-based Multinoulli symbols.

    Symbols ``0`` through ``n_units - 1`` mean one selected unit fired in the
    bin. Symbol ``n_units`` means no unit fired. If multiple spikes occur in a
    bin, one unit is sampled with probability proportional to its count.
    """

    rng = _as_rng(random_state)
    counts = np.asarray(counts)
    if counts.ndim != 2:
        raise ValueError("counts must have shape (n_units, n_time_bins)")
    n_units, n_bins = counts.shape
    symbols = np.full(n_bins, n_units, dtype=np.int32)
    totals = counts.sum(axis=0)
    for t in np.flatnonzero(totals > 0):
        weights = counts[:, t].astype(float)
        weights = weights / weights.sum()
        symbols[t] = int(rng.choice(n_units, p=weights))
    return symbols


def log_multinoulli_emissions(symbols: np.ndarray, emissions: np.ndarray, floor_prob: float = 1e-12) -> np.ndarray:
    symbols = np.asarray(symbols, dtype=np.int32).ravel()
    emissions = np.maximum(np.asarray(emissions, dtype=float), floor_prob)
    emissions = emissions / emissions.sum(axis=1, keepdims=True)
    if np.any(symbols < 0) or np.any(symbols >= emissions.shape[1]):
        raise ValueError("symbols contain values outside the emission alphabet")
    return np.log(emissions[:, symbols])


def _initialize_emissions(symbols: list[np.ndarray], n_states: int, n_symbols: int, rng: np.random.Generator) -> np.ndarray:
    pooled = np.concatenate(symbols)
    base = np.bincount(pooled, minlength=n_symbols).astype(float) + 1.0
    base = base / base.sum()
    emissions = rng.dirichlet(10.0 * base, size=n_states)
    return emissions / emissions.sum(axis=1, keepdims=True)


def fit_multinoulli_hmm(
    symbols: list[np.ndarray],
    n_states: int,
    *,
    n_symbols: int | None = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
    loglik_tol: float = 1e-6,
    floor_prob: float = 1e-12,
    init_emissions: np.ndarray | None = None,
    init_gamma: np.ndarray | None = None,
    random_state: int | np.random.Generator | None = None,
    verbose: bool = False,
) -> MultinoulliHMMResult:
    """Fit a categorical or Multinoulli HMM.

    This is the Python counterpart of the MATLAB mHMM demo.
    """

    if not symbols:
        raise ValueError("symbols must contain at least one trial")
    symbols = [np.asarray(s, dtype=np.int32).ravel() for s in symbols]
    if n_symbols is None:
        n_symbols = int(max(s.max(initial=0) for s in symbols) + 1)
    rng = _as_rng(random_state)

    emissions = (
        np.asarray(init_emissions, dtype=float).copy()
        if init_emissions is not None
        else _initialize_emissions(symbols, n_states, n_symbols, rng)
    )
    emissions = np.maximum(emissions, floor_prob)
    emissions = emissions / emissions.sum(axis=1, keepdims=True)
    gamma = (
        np.asarray(init_gamma, dtype=float).copy()
        if init_gamma is not None
        else _initialize_transition(n_states, 0.8, rng)
    )
    gamma = _normalize_rows(gamma, np.full_like(gamma, 1.0 / n_states))

    deltas = np.full((len(symbols), n_states), 1.0 / n_states)
    logliks: list[float] = []
    crits: list[float] = []
    gamma_diag: list[np.ndarray] = [np.diag(gamma).copy()]
    old_loglik = 1.0
    converged = False
    final_loglik = float("-inf")
    final_iter = 0

    for iteration in range(1, max_iter + 1):
        old_emissions = emissions.copy()
        old_gamma = gamma.copy()
        loglik = 0.0
        gamma_num = np.zeros((n_states, n_states))
        emission_num = np.zeros((n_states, n_symbols))
        emission_den = np.zeros(n_states)

        for trial, seq in enumerate(symbols):
            log_em = log_multinoulli_emissions(seq, old_emissions, floor_prob)
            _, _, q, xi_sum, trial_ll = forward_backward_scaled(log_em, old_gamma, deltas[trial])
            loglik += trial_ll
            gamma_num += xi_sum
            emission_den += q.sum(axis=1)
            for state in range(n_states):
                np.add.at(emission_num[state], seq, q[state])
            deltas[trial] = q[:, 0] / np.maximum(q[:, 0].sum(), np.finfo(float).tiny)

        gamma_next = _normalize_rows(gamma_num, old_gamma)
        emissions_next = emission_num / np.maximum(emission_den[:, None], np.finfo(float).tiny)
        unused = emission_den <= np.finfo(float).tiny
        if np.any(unused):
            emissions_next[unused] = old_emissions[unused]
        emissions_next = np.maximum(emissions_next, floor_prob)
        emissions_next = emissions_next / emissions_next.sum(axis=1, keepdims=True)

        crit = float(np.linalg.norm(old_gamma - gamma_next) + np.linalg.norm(old_emissions - emissions_next))
        gamma = gamma_next
        emissions = emissions_next
        logliks.append(float(loglik))
        crits.append(crit)
        gamma_diag.append(np.diag(gamma).copy())
        final_loglik = float(loglik)
        final_iter = iteration

        if verbose and (iteration == 1 or iteration % 25 == 0):
            print(f"iter={iteration:04d} loglik={loglik:.3f} crit={crit:.3g}")

        if abs(loglik - old_loglik) < loglik_tol and crit < tol:
            converged = True
            break
        old_loglik = loglik

    return MultinoulliHMMResult(
        emissions=emissions,
        gamma=gamma,
        deltas=deltas,
        converged=converged,
        n_iter=final_iter,
        log_likelihood=final_loglik,
        history=PHMMHistory(
            log_likelihood=np.asarray(logliks),
            gamma_diag=np.asarray(gamma_diag).T,
            crit=np.asarray(crits),
            reset_hits=np.empty((0, 2), dtype=int),
        ),
    )


def multinoulli_state_probabilities(symbols: np.ndarray, emissions: np.ndarray, gamma: np.ndarray, delta: np.ndarray | None = None) -> np.ndarray:
    log_em = log_multinoulli_emissions(symbols, emissions)
    _, _, probs, _, _ = forward_backward_scaled(log_em, gamma, delta)
    return probs


def multinoulli_viterbi_decode(symbols: np.ndarray, emissions: np.ndarray, gamma: np.ndarray, delta: np.ndarray | None = None) -> np.ndarray:
    symbols = np.asarray(symbols, dtype=np.int32).ravel()
    log_em = log_multinoulli_emissions(symbols, emissions)
    n_states, n_bins = log_em.shape
    if delta is None:
        delta = stationary_distribution(gamma)
    tiny = np.finfo(float).tiny
    log_gamma = np.log(np.maximum(gamma, tiny))
    log_delta = np.log(np.maximum(delta / np.sum(delta), tiny))
    score = np.empty((n_states, n_bins))
    back = np.zeros((n_states, n_bins), dtype=np.int32)
    score[:, 0] = log_delta + log_em[:, 0]
    for t in range(1, n_bins):
        candidates = score[:, t - 1][:, None] + log_gamma
        back[:, t] = np.argmax(candidates, axis=0)
        score[:, t] = log_em[:, t] + np.max(candidates, axis=0)
    states = np.zeros(n_bins, dtype=np.int32)
    states[-1] = int(np.argmax(score[:, -1]))
    for t in range(n_bins - 2, -1, -1):
        states[t] = back[states[t + 1], t + 1]
    return states
