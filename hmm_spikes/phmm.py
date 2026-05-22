from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln, logsumexp


@dataclass
class PHMMHistory:
    log_likelihood: np.ndarray
    gamma_diag: np.ndarray
    crit: np.ndarray
    reset_hits: np.ndarray


@dataclass
class PHMMResult:
    means: np.ndarray
    gamma: np.ndarray
    deltas: np.ndarray
    bin_size: float
    converged: bool
    n_iter: int
    log_likelihood: float
    history: PHMMHistory
    threshold: float
    threshold_satisfied: bool
    restored_on_exit: bool

    @property
    def rates_hz(self) -> np.ndarray:
        return self.means / self.bin_size


def stationary_distribution(gamma: np.ndarray) -> np.ndarray:
    """Return the stationary row distribution for a row-stochastic matrix."""

    gamma = np.asarray(gamma, dtype=float)
    m = gamma.shape[0]
    a = np.eye(m) - gamma.T
    a[-1] = 1.0
    b = np.zeros(m)
    b[-1] = 1.0
    try:
        delta = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        delta = np.linalg.lstsq(a, b, rcond=None)[0]
    delta = np.maximum(delta, 0.0)
    total = delta.sum()
    if total <= 0:
        return np.full(m, 1.0 / m)
    return delta / total


def _as_rng(random_state: int | np.random.Generator | None) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _initialize_transition(n_states: int, threshold: float, rng: np.random.Generator) -> np.ndarray:
    raw = rng.random((n_states, n_states))
    raw = raw / raw.sum(axis=1, keepdims=True)
    gamma = threshold * np.eye(n_states) + (1.0 - threshold) * raw
    return gamma / gamma.sum(axis=1, keepdims=True)


def _initialize_means(
    counts: list[np.ndarray],
    n_states: int,
    floor_mean: float,
    rng: np.random.Generator,
) -> np.ndarray:
    all_counts = np.concatenate([np.asarray(x, dtype=float) for x in counts], axis=1)
    max_mean = np.maximum(all_counts.max(axis=1), all_counts.mean(axis=1))
    max_mean = np.maximum(max_mean, floor_mean * 10.0)
    means = rng.random((all_counts.shape[0], n_states)) * max_mean[:, None]
    return np.maximum(means, floor_mean)


def log_poisson_emissions(x: np.ndarray, means: np.ndarray, floor_mean: float = 1e-12) -> np.ndarray:
    """Compute log p(x_t | state) for a matrix of spike counts."""

    x = np.asarray(x, dtype=float)
    means = np.maximum(np.asarray(means, dtype=float), floor_mean)
    return (x.T @ np.log(means)).T - means.sum(axis=0)[:, None] - gammaln(x + 1.0).sum(axis=0)[None, :]


def forward_backward(
    log_emissions: np.ndarray,
    gamma: np.ndarray,
    delta: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run log-space forward and backward recursions."""

    log_emissions = np.asarray(log_emissions, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    n_states, n_bins = log_emissions.shape
    if delta is None:
        delta = stationary_distribution(gamma)
    delta = np.asarray(delta, dtype=float)
    delta = delta / delta.sum()

    tiny = np.finfo(float).tiny
    log_gamma = np.log(np.maximum(gamma, tiny))
    log_delta = np.log(np.maximum(delta, tiny))

    log_alpha = np.empty((n_states, n_bins), dtype=float)
    log_beta = np.zeros((n_states, n_bins), dtype=float)
    log_alpha[:, 0] = log_delta + log_emissions[:, 0]
    for t in range(1, n_bins):
        log_alpha[:, t] = log_emissions[:, t] + logsumexp(
            log_alpha[:, t - 1][:, None] + log_gamma,
            axis=0,
        )

    for t in range(n_bins - 2, -1, -1):
        log_beta[:, t] = logsumexp(
            log_gamma + log_emissions[:, t + 1][None, :] + log_beta[:, t + 1][None, :],
            axis=1,
        )

    log_likelihood = float(logsumexp(log_alpha[:, -1]))
    return log_alpha, log_beta, log_likelihood


def forward_backward_scaled(
    log_emissions: np.ndarray,
    gamma: np.ndarray,
    delta: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Scaled forward-backward pass for EM.

    Emissions are scaled independently at each time bin before exponentiation.
    This preserves posteriors while avoiding the heavy cost of full log-space EM.
    """

    log_emissions = np.asarray(log_emissions, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    n_states, n_bins = log_emissions.shape
    if delta is None:
        delta = stationary_distribution(gamma)
    delta = np.asarray(delta, dtype=float)
    delta = delta / delta.sum()

    tiny = np.finfo(float).tiny
    emission_offsets = np.max(log_emissions, axis=0)
    emissions = np.exp(log_emissions - emission_offsets[None, :])
    emissions = np.maximum(emissions, tiny)

    alpha = np.empty((n_states, n_bins), dtype=float)
    beta = np.empty((n_states, n_bins), dtype=float)
    scales = np.empty(n_bins, dtype=float)

    alpha[:, 0] = delta * emissions[:, 0]
    scales[0] = max(alpha[:, 0].sum(), tiny)
    alpha[:, 0] /= scales[0]
    for t in range(1, n_bins):
        alpha[:, t] = (alpha[:, t - 1] @ gamma) * emissions[:, t]
        scales[t] = max(alpha[:, t].sum(), tiny)
        alpha[:, t] /= scales[t]

    beta[:, -1] = 1.0
    for t in range(n_bins - 2, -1, -1):
        beta[:, t] = gamma @ (emissions[:, t + 1] * beta[:, t + 1])
        beta[:, t] /= scales[t + 1]

    q = alpha * beta
    q = q / np.maximum(q.sum(axis=0, keepdims=True), tiny)

    if n_bins < 2:
        xi_sum = np.zeros((n_states, n_states), dtype=float)
    else:
        future = emissions[:, 1:] * beta[:, 1:]
        xi = alpha[:, None, :-1] * gamma[:, :, None] * future[None, :, :]
        xi = xi / np.maximum(xi.sum(axis=(0, 1), keepdims=True), tiny)
        xi_sum = xi.sum(axis=2)

    log_likelihood = float(emission_offsets.sum() + np.log(scales).sum())
    return alpha, beta, q, xi_sum, log_likelihood


def _expected_transitions(
    log_alpha: np.ndarray,
    log_beta: np.ndarray,
    log_emissions: np.ndarray,
    gamma: np.ndarray,
    log_likelihood: float,
) -> np.ndarray:
    n_states, n_bins = log_alpha.shape
    if n_bins < 2:
        return np.zeros((n_states, n_states), dtype=float)

    tiny = np.finfo(float).tiny
    log_gamma = np.log(np.maximum(gamma, tiny))
    log_future = log_emissions[:, 1:] + log_beta[:, 1:]
    log_xi = (
        log_alpha[:, None, :-1]
        + log_gamma[:, :, None]
        + log_future[None, :, :]
        - log_likelihood
    )
    return np.exp(log_xi).sum(axis=2)


def _normalize_rows(matrix: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    out = np.asarray(matrix, dtype=float).copy()
    row_sums = out.sum(axis=1, keepdims=True)
    bad = row_sums[:, 0] <= 0
    row_sums[bad] = 1.0
    out = out / row_sums
    if np.any(bad):
        out[bad] = fallback[bad]
    return out / out.sum(axis=1, keepdims=True)


def fit_sticky_poisson_hmm(
    counts: list[np.ndarray],
    n_states: int,
    *,
    bin_size: float,
    threshold: float = 0.8,
    max_iter: int = 1000,
    tol: float = 1e-4,
    loglik_tol: float = 1e-6,
    reset_tol: float = 5e-5,
    floor_mean: float = 1e-3,
    init_means: np.ndarray | None = None,
    init_gamma: np.ndarray | None = None,
    random_state: int | np.random.Generator | None = None,
    sticky: bool = True,
    verbose: bool = False,
) -> PHMMResult:
    """Fit a sticky Poisson HMM with Baum-Welch EM.

    ``means`` are Poisson means per time bin. Divide by ``bin_size`` to get Hz.
    Setting ``sticky=False`` gives the standard Poisson HMM.
    """

    if not counts:
        raise ValueError("counts must contain at least one trial")
    counts = [np.asarray(x, dtype=float) for x in counts]
    n_neurons = counts[0].shape[0]
    if any(x.shape[0] != n_neurons for x in counts):
        raise ValueError("all count matrices must have the same neuron count")

    rng = _as_rng(random_state)
    means = (
        np.asarray(init_means, dtype=float).copy()
        if init_means is not None
        else _initialize_means(counts, n_states, floor_mean, rng)
    )
    means = np.maximum(means, floor_mean)
    gamma = (
        np.asarray(init_gamma, dtype=float).copy()
        if init_gamma is not None
        else _initialize_transition(n_states, threshold if sticky else 0.8, rng)
    )
    gamma = _normalize_rows(gamma, np.full_like(gamma, 1.0 / n_states))
    if sticky and np.any(np.diag(gamma) < threshold):
        raise ValueError("initial transition diagonal must be at least the sticky threshold")

    n_trials = len(counts)
    deltas = np.full((n_trials, n_states), 1.0 / n_states, dtype=float)
    latest_good_gamma = gamma.copy()
    latest_good_means = means.copy()
    latest_good_deltas = deltas.copy()

    logliks: list[float] = []
    crits: list[float] = []
    gamma_diag: list[np.ndarray] = [np.diag(gamma).copy()]
    reset_hits: list[tuple[int, int]] = []

    old_loglik = 1.0
    converged = False
    final_loglik = float("-inf")
    final_iter = 0

    for iteration in range(1, max_iter + 1):
        old_means = means.copy()
        old_gamma = gamma.copy()
        loglik = 0.0
        gamma_num = np.zeros((n_states, n_states), dtype=float)
        means_num = np.zeros((n_neurons, n_states), dtype=float)
        means_den = np.zeros(n_states, dtype=float)

        for trial, x in enumerate(counts):
            log_em = log_poisson_emissions(x, old_means, floor_mean)
            _, _, q, xi_sum, trial_ll = forward_backward_scaled(log_em, old_gamma, deltas[trial])
            loglik += trial_ll

            gamma_num += xi_sum
            means_num += x @ q.T
            means_den += q.sum(axis=1)

            delta_next = q[:, 0]
            deltas[trial] = delta_next / np.maximum(delta_next.sum(), np.finfo(float).tiny)

        gamma_next = _normalize_rows(gamma_num, old_gamma)
        means_next = np.divide(
            means_num,
            np.maximum(means_den[None, :], np.finfo(float).tiny),
        )
        unused = means_den <= np.finfo(float).tiny
        if np.any(unused):
            means_next[:, unused] = old_means[:, unused]
        means_next = np.maximum(means_next, floor_mean)

        flag = 0.0
        bad_diag = np.flatnonzero(np.diag(gamma_next) < threshold) if sticky else np.array([], dtype=int)
        diag_has_stalled = np.all(np.abs(np.diag(gamma_next) - np.diag(old_gamma)) < reset_tol)
        if sticky and bad_diag.size and diag_has_stalled:
            gamma_next = latest_good_gamma.copy()
            means_next = latest_good_means.copy()
            for state in bad_diag:
                means_next[:, state] = latest_good_means[rng.permutation(n_neurons), state]
                reset_hits.append((iteration, int(state)))
            flag = 1.0
        elif sticky and np.all(np.diag(gamma_next) >= threshold):
            latest_good_gamma = gamma_next.copy()
            latest_good_means = means_next.copy()
            latest_good_deltas = deltas.copy()

        crit = float(np.linalg.norm(old_means - means_next) + np.linalg.norm(old_gamma - gamma_next) + flag)
        means = means_next
        gamma = gamma_next

        logliks.append(float(loglik))
        crits.append(crit)
        gamma_diag.append(np.diag(gamma).copy())
        final_loglik = float(loglik)
        final_iter = iteration

        if verbose and (iteration == 1 or iteration % 25 == 0):
            print(
                f"iter={iteration:04d} loglik={loglik:.3f} "
                f"crit={crit:.3g} min_diag={np.diag(gamma).min():.3f}"
            )

        if abs(loglik - old_loglik) < loglik_tol and crit < tol:
            if not sticky or np.all(np.diag(gamma) >= threshold):
                converged = True
                break
        old_loglik = loglik

    restored_on_exit = False
    if sticky and np.any(np.diag(gamma) < threshold):
        gamma = latest_good_gamma.copy()
        means = latest_good_means.copy()
        deltas = latest_good_deltas.copy()
        restored_on_exit = True
        final_loglik = 0.0
        for trial, x in enumerate(counts):
            log_em = log_poisson_emissions(x, means, floor_mean)
            _, _, _, _, trial_ll = forward_backward_scaled(log_em, gamma, deltas[trial])
            final_loglik += trial_ll

    history = PHMMHistory(
        log_likelihood=np.asarray(logliks, dtype=float),
        gamma_diag=np.asarray(gamma_diag, dtype=float).T,
        crit=np.asarray(crits, dtype=float),
        reset_hits=np.asarray(reset_hits, dtype=int) if reset_hits else np.empty((0, 2), dtype=int),
    )
    return PHMMResult(
        means=means,
        gamma=gamma,
        deltas=deltas,
        bin_size=bin_size,
        converged=converged,
        n_iter=final_iter,
        log_likelihood=final_loglik,
        history=history,
        threshold=threshold if sticky else 0.0,
        threshold_satisfied=bool((not sticky) or np.all(np.diag(gamma) >= threshold)),
        restored_on_exit=restored_on_exit,
    )


def fit_poisson_hmm(
    counts: list[np.ndarray],
    n_states: int,
    *,
    bin_size: float,
    max_iter: int = 1000,
    tol: float = 1e-4,
    loglik_tol: float = 1e-6,
    floor_mean: float = 1e-3,
    init_means: np.ndarray | None = None,
    init_gamma: np.ndarray | None = None,
    random_state: int | np.random.Generator | None = None,
    verbose: bool = False,
) -> PHMMResult:
    """Fit the standard Poisson HMM.

    This is the Python counterpart of the MATLAB standard PHMM demo.
    """

    return fit_sticky_poisson_hmm(
        counts,
        n_states,
        bin_size=bin_size,
        threshold=0.0,
        max_iter=max_iter,
        tol=tol,
        loglik_tol=loglik_tol,
        floor_mean=floor_mean,
        init_means=init_means,
        init_gamma=init_gamma,
        random_state=random_state,
        sticky=False,
        verbose=verbose,
    )


def _dirichlet_alpha(n_states: int, mode_self_transition: float, offdiag_alpha: float) -> np.ndarray:
    if not 0.0 < mode_self_transition < 1.0:
        raise ValueError("mode_self_transition must be between 0 and 1")
    if offdiag_alpha <= 1.0:
        raise ValueError("offdiag_alpha must be greater than 1")

    alpha = offdiag_alpha * np.ones((n_states, n_states), dtype=float)
    diag_alpha = 1.0 + mode_self_transition * (n_states - 1) * (offdiag_alpha - 1.0) / (1.0 - mode_self_transition)
    np.fill_diagonal(alpha, diag_alpha)
    return alpha


def fit_dirichlet_poisson_hmm(
    counts: list[np.ndarray],
    n_states: int,
    *,
    bin_size: float,
    mode_self_transition: float = 0.9,
    offdiag_alpha: float = 1.1,
    max_iter: int = 1000,
    tol: float = 1e-4,
    loglik_tol: float = 1e-6,
    floor_mean: float = 1e-3,
    init_means: np.ndarray | None = None,
    init_gamma: np.ndarray | None = None,
    random_state: int | np.random.Generator | None = None,
    verbose: bool = False,
) -> PHMMResult:
    """Fit a Poisson HMM with a Dirichlet prior over transition rows.

    The prior nudges self-transition probabilities upward but does not impose
    the hard sticky threshold used by ``fit_sticky_poisson_hmm``.
    """

    if not counts:
        raise ValueError("counts must contain at least one trial")
    counts = [np.asarray(x, dtype=float) for x in counts]
    n_neurons = counts[0].shape[0]
    if any(x.shape[0] != n_neurons for x in counts):
        raise ValueError("all count matrices must have the same neuron count")

    rng = _as_rng(random_state)
    alpha = _dirichlet_alpha(n_states, mode_self_transition, offdiag_alpha)
    means = (
        np.asarray(init_means, dtype=float).copy()
        if init_means is not None
        else _initialize_means(counts, n_states, floor_mean, rng)
    )
    means = np.maximum(means, floor_mean)
    gamma = (
        np.asarray(init_gamma, dtype=float).copy()
        if init_gamma is not None
        else _initialize_transition(n_states, min(mode_self_transition, 0.95), rng)
    )
    gamma = _normalize_rows(gamma, np.full_like(gamma, 1.0 / n_states))

    n_trials = len(counts)
    deltas = np.full((n_trials, n_states), 1.0 / n_states, dtype=float)
    logliks: list[float] = []
    crits: list[float] = []
    gamma_diag: list[np.ndarray] = [np.diag(gamma).copy()]
    old_loglik = 1.0
    converged = False
    final_loglik = float("-inf")
    final_iter = 0

    prior_num = alpha - 1.0
    prior_den = prior_num.sum(axis=1, keepdims=True)

    for iteration in range(1, max_iter + 1):
        old_means = means.copy()
        old_gamma = gamma.copy()
        loglik = 0.0
        gamma_num = np.zeros((n_states, n_states), dtype=float)
        means_num = np.zeros((n_neurons, n_states), dtype=float)
        means_den = np.zeros(n_states, dtype=float)

        for trial, x in enumerate(counts):
            log_em = log_poisson_emissions(x, old_means, floor_mean)
            _, _, q, xi_sum, trial_ll = forward_backward_scaled(log_em, old_gamma, deltas[trial])
            loglik += trial_ll
            gamma_num += xi_sum
            means_num += x @ q.T
            means_den += q.sum(axis=1)
            deltas[trial] = q[:, 0] / np.maximum(q[:, 0].sum(), np.finfo(float).tiny)

        gamma_next = (gamma_num + prior_num) / np.maximum(gamma_num.sum(axis=1, keepdims=True) + prior_den, np.finfo(float).tiny)
        gamma_next = _normalize_rows(gamma_next, old_gamma)
        means_next = means_num / np.maximum(means_den[None, :], np.finfo(float).tiny)
        unused = means_den <= np.finfo(float).tiny
        if np.any(unused):
            means_next[:, unused] = old_means[:, unused]
        means_next = np.maximum(means_next, floor_mean)

        crit = float(np.linalg.norm(old_means - means_next) + np.linalg.norm(old_gamma - gamma_next))
        means = means_next
        gamma = gamma_next
        logliks.append(float(loglik))
        crits.append(crit)
        gamma_diag.append(np.diag(gamma).copy())
        final_loglik = float(loglik)
        final_iter = iteration

        if verbose and (iteration == 1 or iteration % 25 == 0):
            print(
                f"iter={iteration:04d} loglik={loglik:.3f} "
                f"crit={crit:.3g} min_diag={np.diag(gamma).min():.3f}"
            )

        if abs(loglik - old_loglik) < loglik_tol and crit < tol:
            converged = True
            break
        old_loglik = loglik

    history = PHMMHistory(
        log_likelihood=np.asarray(logliks, dtype=float),
        gamma_diag=np.asarray(gamma_diag, dtype=float).T,
        crit=np.asarray(crits, dtype=float),
        reset_hits=np.empty((0, 2), dtype=int),
    )
    return PHMMResult(
        means=means,
        gamma=gamma,
        deltas=deltas,
        bin_size=bin_size,
        converged=converged,
        n_iter=final_iter,
        log_likelihood=final_loglik,
        history=history,
        threshold=0.0,
        threshold_satisfied=True,
        restored_on_exit=False,
    )


def state_probabilities(
    x: np.ndarray,
    means: np.ndarray,
    gamma: np.ndarray,
    delta: np.ndarray | None = None,
) -> np.ndarray:
    """Posterior state probabilities with shape ``(n_states, n_bins)``."""

    log_em = log_poisson_emissions(x, means)
    _, _, probs, _, _ = forward_backward_scaled(log_em, gamma, delta)
    return probs


def viterbi_decode(
    x: np.ndarray,
    means: np.ndarray,
    gamma: np.ndarray,
    delta: np.ndarray | None = None,
) -> np.ndarray:
    """Return the most likely zero-based state sequence."""

    log_em = log_poisson_emissions(x, means)
    n_states, n_bins = log_em.shape
    if delta is None:
        delta = stationary_distribution(gamma)
    tiny = np.finfo(float).tiny
    log_gamma = np.log(np.maximum(gamma, tiny))
    log_delta = np.log(np.maximum(delta / np.sum(delta), tiny))

    score = np.empty((n_states, n_bins), dtype=float)
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


def model_log_likelihood(
    counts: list[np.ndarray],
    means: np.ndarray,
    gamma: np.ndarray,
    deltas: np.ndarray | None = None,
) -> float:
    """Compute total log-likelihood across trials."""

    total = 0.0
    for trial, x in enumerate(counts):
        delta = None if deltas is None else deltas[trial]
        log_em = log_poisson_emissions(x, means)
        _, _, _, _, ll = forward_backward_scaled(log_em, gamma, delta)
        total += ll
    return float(total)
