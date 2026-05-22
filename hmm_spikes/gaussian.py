from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .phmm import PHMMHistory, _as_rng, _initialize_transition, _normalize_rows, forward_backward_scaled, stationary_distribution


@dataclass
class GaussianHMMResult:
    means: np.ndarray
    variances: np.ndarray
    gamma: np.ndarray
    deltas: np.ndarray
    converged: bool
    n_iter: int
    log_likelihood: float
    history: PHMMHistory
    threshold: float
    threshold_satisfied: bool
    restored_on_exit: bool


def _as_observations(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError("observations must have shape (n_features, n_time_bins)")
    return x


def log_gaussian_emissions(x: np.ndarray, means: np.ndarray, variances: np.ndarray, floor_variance: float = 1e-6) -> np.ndarray:
    x = _as_observations(x)
    means = np.asarray(means, dtype=float)
    variances = np.maximum(np.asarray(variances, dtype=float), floor_variance)
    diff = x[:, None, :] - means[:, :, None]
    return -0.5 * (
        np.log(2.0 * np.pi * variances).sum(axis=0)[:, None]
        + ((diff * diff) / variances[:, :, None]).sum(axis=0)
    )


def _initialize_gaussian_parameters(observations: list[np.ndarray], n_states: int, floor_variance: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    pooled = np.concatenate([_as_observations(x) for x in observations], axis=1)
    n_features, n_bins = pooled.shape
    choices = rng.choice(n_bins, size=n_states, replace=n_bins < n_states)
    means = pooled[:, choices].copy()
    global_var = np.maximum(np.var(pooled, axis=1), floor_variance)
    variances = np.tile(global_var[:, None], (1, n_states))
    noise = rng.normal(scale=np.sqrt(global_var)[:, None] * 0.01, size=(n_features, n_states))
    return means + noise, variances


def fit_sticky_gaussian_hmm(
    observations: list[np.ndarray],
    n_states: int,
    *,
    threshold: float = 0.8,
    max_iter: int = 1000,
    tol: float = 1e-4,
    loglik_tol: float = 1e-6,
    reset_tol: float = 5e-5,
    floor_variance: float = 1e-6,
    init_means: np.ndarray | None = None,
    init_variances: np.ndarray | None = None,
    init_gamma: np.ndarray | None = None,
    random_state: int | np.random.Generator | None = None,
    sticky: bool = True,
    verbose: bool = False,
) -> GaussianHMMResult:
    """Fit a diagonal-covariance Gaussian HMM with an optional sticky reset.

    Use this for continuous signals such as raw fiber photometry traces.
    """

    if not observations:
        raise ValueError("observations must contain at least one trial")
    observations = [_as_observations(x) for x in observations]
    n_features = observations[0].shape[0]
    if any(x.shape[0] != n_features for x in observations):
        raise ValueError("all observations must have the same feature count")

    rng = _as_rng(random_state)
    if init_means is None or init_variances is None:
        means0, vars0 = _initialize_gaussian_parameters(observations, n_states, floor_variance, rng)
    means = np.asarray(init_means, dtype=float).copy() if init_means is not None else means0
    variances = np.asarray(init_variances, dtype=float).copy() if init_variances is not None else vars0
    variances = np.maximum(variances, floor_variance)
    gamma = (
        np.asarray(init_gamma, dtype=float).copy()
        if init_gamma is not None
        else _initialize_transition(n_states, threshold if sticky else 0.8, rng)
    )
    gamma = _normalize_rows(gamma, np.full_like(gamma, 1.0 / n_states))
    if sticky and np.any(np.diag(gamma) < threshold):
        raise ValueError("initial transition diagonal must be at least the sticky threshold")

    deltas = np.full((len(observations), n_states), 1.0 / n_states)
    latest_good_gamma = gamma.copy()
    latest_good_means = means.copy()
    latest_good_variances = variances.copy()
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
        old_variances = variances.copy()
        old_gamma = gamma.copy()
        loglik = 0.0
        gamma_num = np.zeros((n_states, n_states))
        weight_sum = np.zeros(n_states)
        mean_num = np.zeros((n_features, n_states))
        second_num = np.zeros((n_features, n_states))

        for trial, x in enumerate(observations):
            log_em = log_gaussian_emissions(x, old_means, old_variances, floor_variance)
            _, _, q, xi_sum, trial_ll = forward_backward_scaled(log_em, old_gamma, deltas[trial])
            loglik += trial_ll
            gamma_num += xi_sum
            weight_sum += q.sum(axis=1)
            mean_num += x @ q.T
            second_num += (x * x) @ q.T
            deltas[trial] = q[:, 0] / np.maximum(q[:, 0].sum(), np.finfo(float).tiny)

        gamma_next = _normalize_rows(gamma_num, old_gamma)
        means_next = mean_num / np.maximum(weight_sum[None, :], np.finfo(float).tiny)
        variances_next = second_num / np.maximum(weight_sum[None, :], np.finfo(float).tiny) - means_next * means_next
        unused = weight_sum <= np.finfo(float).tiny
        if np.any(unused):
            means_next[:, unused] = old_means[:, unused]
            variances_next[:, unused] = old_variances[:, unused]
        variances_next = np.maximum(variances_next, floor_variance)

        flag = 0.0
        bad_diag = np.flatnonzero(np.diag(gamma_next) < threshold) if sticky else np.array([], dtype=int)
        diag_has_stalled = np.all(np.abs(np.diag(gamma_next) - np.diag(old_gamma)) < reset_tol)
        if sticky and bad_diag.size and diag_has_stalled:
            gamma_next = latest_good_gamma.copy()
            means_next = latest_good_means.copy()
            variances_next = latest_good_variances.copy()
            for state in bad_diag:
                perm = rng.permutation(n_features)
                means_next[:, state] = latest_good_means[perm, state]
                variances_next[:, state] = latest_good_variances[perm, state]
                reset_hits.append((iteration, int(state)))
            flag = 1.0
        elif sticky and np.all(np.diag(gamma_next) >= threshold):
            latest_good_gamma = gamma_next.copy()
            latest_good_means = means_next.copy()
            latest_good_variances = variances_next.copy()
            latest_good_deltas = deltas.copy()

        crit = float(
            np.linalg.norm(old_gamma - gamma_next)
            + np.linalg.norm(old_means - means_next)
            + np.linalg.norm(old_variances - variances_next)
            + flag
        )
        gamma = gamma_next
        means = means_next
        variances = variances_next
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
        variances = latest_good_variances.copy()
        deltas = latest_good_deltas.copy()
        restored_on_exit = True
        final_loglik = 0.0
        for trial, x in enumerate(observations):
            log_em = log_gaussian_emissions(x, means, variances, floor_variance)
            _, _, _, _, trial_ll = forward_backward_scaled(log_em, gamma, deltas[trial])
            final_loglik += trial_ll

    return GaussianHMMResult(
        means=means,
        variances=variances,
        gamma=gamma,
        deltas=deltas,
        converged=converged,
        n_iter=final_iter,
        log_likelihood=final_loglik,
        history=PHMMHistory(
            log_likelihood=np.asarray(logliks),
            gamma_diag=np.asarray(gamma_diag).T,
            crit=np.asarray(crits),
            reset_hits=np.asarray(reset_hits, dtype=int) if reset_hits else np.empty((0, 2), dtype=int),
        ),
        threshold=threshold if sticky else 0.0,
        threshold_satisfied=bool((not sticky) or np.all(np.diag(gamma) >= threshold)),
        restored_on_exit=restored_on_exit,
    )


def fit_gaussian_hmm(observations: list[np.ndarray], n_states: int, **kwargs) -> GaussianHMMResult:
    """Fit a standard diagonal-covariance Gaussian HMM."""

    kwargs["sticky"] = False
    kwargs.setdefault("threshold", 0.0)
    return fit_sticky_gaussian_hmm(observations, n_states, **kwargs)


def gaussian_state_probabilities(x: np.ndarray, means: np.ndarray, variances: np.ndarray, gamma: np.ndarray, delta: np.ndarray | None = None) -> np.ndarray:
    log_em = log_gaussian_emissions(x, means, variances)
    _, _, probs, _, _ = forward_backward_scaled(log_em, gamma, delta)
    return probs


def gaussian_viterbi_decode(x: np.ndarray, means: np.ndarray, variances: np.ndarray, gamma: np.ndarray, delta: np.ndarray | None = None) -> np.ndarray:
    log_em = log_gaussian_emissions(x, means, variances)
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
