from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gaussian import _as_observations, _initialize_gaussian_parameters, log_gaussian_emissions
from .multinoulli import _initialize_emissions, log_multinoulli_emissions
from .phmm import (
    PHMMHistory,
    _as_rng,
    _initialize_means,
    _initialize_transition,
    _normalize_rows,
    forward_backward_scaled,
    log_poisson_emissions,
)


@dataclass
class GraphPoissonHMMResult:
    """Result for a graph-informed Poisson HMM.

    The graph lives over observed units, not over hidden states. It regularizes
    each hidden state's Poisson mean vector so connected units have smoother
    state-dependent count rates.
    """

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
    adjacency: np.ndarray
    laplacian: np.ndarray
    graph_strength: float
    normalized_laplacian: bool
    inferred_graph: bool
    smoothing_delta: np.ndarray

    @property
    def rates_hz(self) -> np.ndarray:
        return self.means / self.bin_size

    @property
    def graph_smoothness(self) -> float:
        return graph_smoothness(self.means, self.laplacian)

    @property
    def graph_summary(self) -> dict[str, np.ndarray | float]:
        return graph_network_summary(self.adjacency)


@dataclass
class GraphGaussianHMMResult:
    """Result for a graph-informed diagonal-covariance Gaussian HMM."""

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
    adjacency: np.ndarray
    laplacian: np.ndarray
    graph_strength: float
    variance_graph_strength: float
    normalized_laplacian: bool
    inferred_graph: bool
    smoothing_delta: np.ndarray

    @property
    def graph_smoothness(self) -> float:
        return graph_smoothness(self.means, self.laplacian)

    @property
    def graph_summary(self) -> dict[str, np.ndarray | float]:
        return graph_network_summary(self.adjacency)


@dataclass
class GraphMultinoulliHMMResult:
    """Result for a graph-informed categorical or Multinoulli HMM."""

    emissions: np.ndarray
    gamma: np.ndarray
    deltas: np.ndarray
    converged: bool
    n_iter: int
    log_likelihood: float
    history: PHMMHistory
    adjacency: np.ndarray
    laplacian: np.ndarray
    graph_strength: float
    normalized_laplacian: bool
    inferred_graph: bool
    smoothing_delta: np.ndarray

    @property
    def graph_smoothness(self) -> float:
        return graph_smoothness(self.emissions.T, self.laplacian)

    @property
    def graph_summary(self) -> dict[str, np.ndarray | float]:
        return graph_network_summary(self.adjacency)


def _validate_counts(counts: list[np.ndarray]) -> tuple[list[np.ndarray], int]:
    if not counts:
        raise ValueError("counts must contain at least one trial")
    clean = [np.asarray(x, dtype=float) for x in counts]
    if any(x.ndim != 2 for x in clean):
        raise ValueError("each count matrix must have shape (n_units, n_time_bins)")
    n_units = clean[0].shape[0]
    if any(x.shape[0] != n_units for x in clean):
        raise ValueError("all count matrices must have the same unit count")
    return clean, n_units


def _validate_observations(observations: list[np.ndarray]) -> tuple[list[np.ndarray], int]:
    if not observations:
        raise ValueError("observations must contain at least one trial")
    clean = [_as_observations(x) for x in observations]
    n_features = clean[0].shape[0]
    if any(x.shape[0] != n_features for x in clean):
        raise ValueError("all observations must have the same feature count")
    return clean, n_features


def _validate_symbol_sequences(
    symbols: list[np.ndarray],
    n_symbols: int | None,
) -> tuple[list[np.ndarray], int]:
    if not symbols:
        raise ValueError("symbols must contain at least one trial")
    clean = [np.asarray(s, dtype=np.int32).ravel() for s in symbols]
    if n_symbols is None:
        n_symbols = int(max(s.max(initial=0) for s in clean) + 1)
    if n_symbols <= 0:
        raise ValueError("n_symbols must be positive")
    for seq in clean:
        if np.any(seq < 0) or np.any(seq >= n_symbols):
            raise ValueError("symbols contain values outside the emission alphabet")
    return clean, int(n_symbols)


def _prepare_adjacency(adjacency: np.ndarray, n_units: int | None = None) -> np.ndarray:
    graph = np.asarray(adjacency, dtype=float)
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    if n_units is not None and graph.shape != (n_units, n_units):
        raise ValueError("adjacency shape must match the number of observed units")
    if not np.all(np.isfinite(graph)):
        raise ValueError("adjacency must contain only finite values")
    if np.any(graph < 0.0):
        raise ValueError("adjacency must be nonnegative")

    graph = 0.5 * (graph + graph.T)
    graph = graph.copy()
    np.fill_diagonal(graph, 0.0)
    return graph


def graph_laplacian(adjacency: np.ndarray, *, normalized: bool = True) -> np.ndarray:
    """Return the combinatorial or symmetric normalized graph Laplacian."""

    graph = _prepare_adjacency(adjacency)
    degree = graph.sum(axis=1)
    if not normalized:
        return np.diag(degree) - graph

    lap = np.eye(graph.shape[0], dtype=float)
    connected = degree > 0.0
    lap[~connected, ~connected] = 0.0
    inv_sqrt = np.zeros_like(degree)
    inv_sqrt[connected] = 1.0 / np.sqrt(degree[connected])
    lap -= inv_sqrt[:, None] * graph * inv_sqrt[None, :]
    return 0.5 * (lap + lap.T)


def infer_functional_connectivity_graph(
    counts: list[np.ndarray],
    *,
    method: str = "correlation",
    absolute: bool = True,
    threshold_quantile: float | None = 0.75,
    top_k: int | None = None,
) -> np.ndarray:
    """Infer a weighted unit graph from trial count matrices.

    Parameters
    ----------
    counts:
        Count matrices with shape ``(n_units, n_time_bins)``.
    method:
        Currently ``"correlation"`` or ``"cosine"``.
    absolute:
        If true, strong negative correlations become strong edges. If false,
        only positive associations survive.
    threshold_quantile:
        Optional quantile threshold applied to positive edge weights.
    top_k:
        Optional per-node top-k pruning. The final graph is symmetrized by
        union of each node's retained neighbors.
    """

    counts, n_units = _validate_counts(counts)
    if n_units == 1:
        return np.zeros((1, 1), dtype=float)
    if threshold_quantile is not None and not 0.0 <= threshold_quantile <= 1.0:
        raise ValueError("threshold_quantile must be between 0 and 1")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive")

    pooled = np.concatenate(counts, axis=1)
    if method == "correlation":
        centered = pooled - pooled.mean(axis=1, keepdims=True)
        denom_source = centered
    elif method == "cosine":
        denom_source = pooled
        centered = pooled
    else:
        raise ValueError("method must be 'correlation' or 'cosine'")

    norms = np.linalg.norm(denom_source, axis=1)
    denom = norms[:, None] * norms[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        assoc = (centered @ centered.T) / denom
    assoc[~np.isfinite(assoc)] = 0.0

    graph = np.abs(assoc) if absolute else np.maximum(assoc, 0.0)
    np.fill_diagonal(graph, 0.0)

    if threshold_quantile is not None:
        positive = graph[graph > 0.0]
        if positive.size:
            cutoff = float(np.quantile(positive, threshold_quantile))
            graph[graph < cutoff] = 0.0

    if top_k is not None and top_k < n_units:
        mask = np.zeros_like(graph, dtype=bool)
        for row in range(n_units):
            candidates = np.flatnonzero(graph[row] > 0.0)
            if candidates.size > top_k:
                strongest = candidates[np.argsort(graph[row, candidates])[-top_k:]]
            else:
                strongest = candidates
            mask[row, strongest] = True
        graph = np.where(mask | mask.T, graph, 0.0)

    return _prepare_adjacency(graph, n_units)


def infer_observation_graph(
    observations: list[np.ndarray],
    *,
    method: str = "correlation",
    absolute: bool = True,
    threshold_quantile: float | None = 0.75,
    top_k: int | None = None,
) -> np.ndarray:
    """Infer a weighted graph from count or continuous observation matrices."""

    clean, _ = _validate_observations(observations)
    return infer_functional_connectivity_graph(
        clean,
        method=method,
        absolute=absolute,
        threshold_quantile=threshold_quantile,
        top_k=top_k,
    )


def infer_symbol_transition_graph(
    symbols: list[np.ndarray],
    *,
    n_symbols: int | None = None,
    window: int = 1,
    isolate_last_symbol: bool = True,
) -> np.ndarray:
    """Infer a symbol graph from short-lag temporal adjacency in sequences.

    For the package's count-to-Multinoulli convention, the last symbol is
    usually the no-event symbol. It is isolated by default because smoothing
    no-event probability into unit-firing symbols is usually a bad model.
    """

    if window < 1:
        raise ValueError("window must be positive")
    sequences, n_symbols = _validate_symbol_sequences(symbols, n_symbols)
    graph = np.zeros((n_symbols, n_symbols), dtype=float)
    for seq in sequences:
        for lag in range(1, window + 1):
            if seq.size <= lag:
                continue
            weight = 1.0 / lag
            left = seq[:-lag]
            right = seq[lag:]
            np.add.at(graph, (left, right), weight)
            np.add.at(graph, (right, left), weight)

    np.fill_diagonal(graph, 0.0)
    if isolate_last_symbol and n_symbols > 1:
        graph[-1, :] = 0.0
        graph[:, -1] = 0.0
    max_weight = graph.max(initial=0.0)
    if max_weight > 0.0:
        graph = graph / max_weight
    return _prepare_adjacency(graph, n_symbols)


def _prepare_symbol_adjacency(
    adjacency: np.ndarray,
    n_symbols: int,
    *,
    isolate_last_symbol: bool,
) -> np.ndarray:
    graph = np.asarray(adjacency, dtype=float)
    if graph.shape == (n_symbols - 1, n_symbols - 1):
        padded = np.zeros((n_symbols, n_symbols), dtype=float)
        padded[:-1, :-1] = graph
        graph = padded
    graph = _prepare_adjacency(graph, n_symbols)
    if isolate_last_symbol and n_symbols > 1:
        graph[-1, :] = 0.0
        graph[:, -1] = 0.0
    return graph


def graph_network_summary(adjacency: np.ndarray) -> dict[str, np.ndarray | float]:
    """Compute lightweight complex-network descriptors for an adjacency matrix."""

    graph = _prepare_adjacency(adjacency)
    n_units = graph.shape[0]
    binary = graph > 0.0
    degree = binary.sum(axis=1).astype(float)
    strength = graph.sum(axis=1)
    possible_edges = n_units * (n_units - 1) / 2.0
    density = 0.0 if possible_edges == 0.0 else float(np.triu(binary, 1).sum() / possible_edges)

    clustering = np.zeros(n_units, dtype=float)
    for node in range(n_units):
        neighbors = np.flatnonzero(binary[node])
        k = neighbors.size
        if k < 2:
            continue
        subgraph = binary[np.ix_(neighbors, neighbors)]
        clustering[node] = float(subgraph.sum() / (k * (k - 1)))

    eigenvector = np.full(n_units, 1.0 / max(n_units, 1), dtype=float)
    if graph.sum() > 0.0 and n_units > 0:
        vector = np.full(n_units, 1.0 / np.sqrt(n_units), dtype=float)
        for _ in range(200):
            next_vector = graph @ vector
            norm = np.linalg.norm(next_vector)
            if norm <= np.finfo(float).tiny:
                break
            next_vector = next_vector / norm
            if np.linalg.norm(next_vector - vector) < 1e-10:
                vector = next_vector
                break
            vector = next_vector
        eigenvector = np.abs(vector)
        total = eigenvector.sum()
        if total > 0.0:
            eigenvector = eigenvector / total

    return {
        "degree": degree,
        "strength": strength,
        "clustering": clustering,
        "eigenvector_centrality": eigenvector,
        "density": density,
        "mean_clustering": float(clustering.mean()) if n_units else 0.0,
        "mean_strength": float(strength.mean()) if n_units else 0.0,
    }


def graph_smoothness(values: np.ndarray, laplacian: np.ndarray) -> float:
    """Return ``sum_k values[:, k].T @ laplacian @ values[:, k]``."""

    values = np.asarray(values, dtype=float)
    laplacian = np.asarray(laplacian, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if laplacian.shape != (values.shape[0], values.shape[0]):
        raise ValueError("laplacian shape must match the first dimension of values")
    return float(np.trace(values.T @ laplacian @ values))


def _smooth_columns(values: np.ndarray, smoothing_matrix: np.ndarray) -> np.ndarray:
    try:
        smoothed = np.linalg.solve(smoothing_matrix, values)
    except np.linalg.LinAlgError:
        smoothed = np.linalg.lstsq(smoothing_matrix, values, rcond=None)[0]
    return smoothed


def _smooth_nonnegative_columns(
    values: np.ndarray,
    smoothing_matrix: np.ndarray,
    floor_value: float,
) -> np.ndarray:
    return np.maximum(_smooth_columns(values, smoothing_matrix), floor_value)


def _smooth_probability_rows(
    probabilities: np.ndarray,
    smoothing_matrix: np.ndarray,
    floor_prob: float,
) -> np.ndarray:
    smoothed = _smooth_columns(probabilities.T, smoothing_matrix).T
    smoothed = np.maximum(smoothed, floor_prob)
    return smoothed / smoothed.sum(axis=1, keepdims=True)


def _smooth_poisson_means(
    means: np.ndarray,
    smoothing_matrix: np.ndarray,
    floor_mean: float,
) -> np.ndarray:
    return _smooth_nonnegative_columns(means, smoothing_matrix, floor_mean)


def _model_log_likelihood(
    counts: list[np.ndarray],
    means: np.ndarray,
    gamma: np.ndarray,
    deltas: np.ndarray,
    floor_mean: float,
) -> float:
    total = 0.0
    for trial, x in enumerate(counts):
        log_em = log_poisson_emissions(x, means, floor_mean)
        _, _, _, _, trial_ll = forward_backward_scaled(log_em, gamma, deltas[trial])
        total += trial_ll
    return float(total)


def _gaussian_model_log_likelihood(
    observations: list[np.ndarray],
    means: np.ndarray,
    variances: np.ndarray,
    gamma: np.ndarray,
    deltas: np.ndarray,
    floor_variance: float,
) -> float:
    total = 0.0
    for trial, x in enumerate(observations):
        log_em = log_gaussian_emissions(x, means, variances, floor_variance)
        _, _, _, _, trial_ll = forward_backward_scaled(log_em, gamma, deltas[trial])
        total += trial_ll
    return float(total)


def _multinoulli_model_log_likelihood(
    symbols: list[np.ndarray],
    emissions: np.ndarray,
    gamma: np.ndarray,
    deltas: np.ndarray,
    floor_prob: float,
) -> float:
    total = 0.0
    for trial, seq in enumerate(symbols):
        log_em = log_multinoulli_emissions(seq, emissions, floor_prob)
        _, _, _, _, trial_ll = forward_backward_scaled(log_em, gamma, deltas[trial])
        total += trial_ll
    return float(total)


def fit_sticky_graph_poisson_hmm(
    counts: list[np.ndarray],
    n_states: int,
    *,
    bin_size: float,
    adjacency: np.ndarray | None = None,
    graph_strength: float = 0.15,
    normalized_laplacian: bool = True,
    infer_graph: bool = True,
    graph_method: str = "correlation",
    graph_absolute: bool = True,
    graph_threshold_quantile: float | None = 0.75,
    graph_top_k: int | None = None,
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
) -> GraphPoissonHMMResult:
    """Fit a graph-informed Poisson HMM.

    This is a generalized EM model. The E-step is the ordinary Poisson HMM
    forward-backward recursion. The M-step first computes the Poisson mean MLE,
    then smooths each hidden state's mean vector with ``I + alpha L``, where
    ``L`` is the unit graph Laplacian and ``alpha`` is ``graph_strength``.
    """

    if graph_strength < 0.0:
        raise ValueError("graph_strength must be nonnegative")
    counts, n_units = _validate_counts(counts)

    inferred_graph = adjacency is None
    if adjacency is None:
        if not infer_graph:
            raise ValueError("adjacency must be provided when infer_graph is false")
        adjacency = infer_functional_connectivity_graph(
            counts,
            method=graph_method,
            absolute=graph_absolute,
            threshold_quantile=graph_threshold_quantile,
            top_k=graph_top_k,
        )
    else:
        adjacency = _prepare_adjacency(adjacency, n_units)
    laplacian = graph_laplacian(adjacency, normalized=normalized_laplacian)
    smoothing_matrix = np.eye(n_units, dtype=float) + graph_strength * laplacian

    rng = _as_rng(random_state)
    means = (
        np.asarray(init_means, dtype=float).copy()
        if init_means is not None
        else _initialize_means(counts, n_states, floor_mean, rng)
    )
    means = np.maximum(means, floor_mean)
    if graph_strength > 0.0:
        means = _smooth_poisson_means(means, smoothing_matrix, floor_mean)

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
    smoothing_delta: list[float] = []

    old_loglik = 1.0
    converged = False
    final_iter = 0

    for iteration in range(1, max_iter + 1):
        old_means = means.copy()
        old_gamma = gamma.copy()
        loglik = 0.0
        gamma_num = np.zeros((n_states, n_states), dtype=float)
        means_num = np.zeros((n_units, n_states), dtype=float)
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
        mle_means = np.divide(
            means_num,
            np.maximum(means_den[None, :], np.finfo(float).tiny),
        )
        unused = means_den <= np.finfo(float).tiny
        if np.any(unused):
            mle_means[:, unused] = old_means[:, unused]
        mle_means = np.maximum(mle_means, floor_mean)
        means_next = (
            _smooth_poisson_means(mle_means, smoothing_matrix, floor_mean)
            if graph_strength > 0.0
            else mle_means
        )

        flag = 0.0
        bad_diag = np.flatnonzero(np.diag(gamma_next) < threshold) if sticky else np.array([], dtype=int)
        diag_has_stalled = np.all(np.abs(np.diag(gamma_next) - np.diag(old_gamma)) < reset_tol)
        if sticky and bad_diag.size and diag_has_stalled:
            gamma_next = latest_good_gamma.copy()
            means_next = latest_good_means.copy()
            for state in bad_diag:
                means_next[:, state] = latest_good_means[rng.permutation(n_units), state]
                reset_hits.append((iteration, int(state)))
            flag = 1.0
        elif sticky and np.all(np.diag(gamma_next) >= threshold):
            latest_good_gamma = gamma_next.copy()
            latest_good_means = means_next.copy()
            latest_good_deltas = deltas.copy()

        smooth_delta = float(np.linalg.norm(mle_means - means_next))
        crit = float(np.linalg.norm(old_means - means_next) + np.linalg.norm(old_gamma - gamma_next) + flag)
        means = means_next
        gamma = gamma_next

        logliks.append(float(loglik))
        crits.append(crit)
        gamma_diag.append(np.diag(gamma).copy())
        smoothing_delta.append(smooth_delta)
        final_iter = iteration

        if verbose and (iteration == 1 or iteration % 25 == 0):
            print(
                f"iter={iteration:04d} loglik={loglik:.3f} "
                f"crit={crit:.3g} smooth={smooth_delta:.3g} min_diag={np.diag(gamma).min():.3f}"
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

    final_loglik = _model_log_likelihood(counts, means, gamma, deltas, floor_mean)
    history = PHMMHistory(
        log_likelihood=np.asarray(logliks, dtype=float),
        gamma_diag=np.asarray(gamma_diag, dtype=float).T,
        crit=np.asarray(crits, dtype=float),
        reset_hits=np.asarray(reset_hits, dtype=int) if reset_hits else np.empty((0, 2), dtype=int),
    )
    return GraphPoissonHMMResult(
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
        adjacency=adjacency,
        laplacian=laplacian,
        graph_strength=float(graph_strength),
        normalized_laplacian=bool(normalized_laplacian),
        inferred_graph=bool(inferred_graph),
        smoothing_delta=np.asarray(smoothing_delta, dtype=float),
    )


def fit_graph_poisson_hmm(
    counts: list[np.ndarray],
    n_states: int,
    *,
    bin_size: float,
    **kwargs,
) -> GraphPoissonHMMResult:
    """Fit a non-sticky graph-informed Poisson HMM."""

    kwargs["sticky"] = False
    kwargs.setdefault("threshold", 0.0)
    return fit_sticky_graph_poisson_hmm(counts, n_states, bin_size=bin_size, **kwargs)


def fit_sticky_graph_gaussian_hmm(
    observations: list[np.ndarray],
    n_states: int,
    *,
    adjacency: np.ndarray | None = None,
    graph_strength: float = 0.15,
    variance_graph_strength: float = 0.0,
    normalized_laplacian: bool = True,
    infer_graph: bool = True,
    graph_method: str = "correlation",
    graph_absolute: bool = True,
    graph_threshold_quantile: float | None = 0.75,
    graph_top_k: int | None = None,
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
) -> GraphGaussianHMMResult:
    """Fit a graph-informed diagonal-covariance Gaussian HMM."""

    if graph_strength < 0.0:
        raise ValueError("graph_strength must be nonnegative")
    if variance_graph_strength < 0.0:
        raise ValueError("variance_graph_strength must be nonnegative")
    observations, n_features = _validate_observations(observations)

    inferred_graph = adjacency is None
    if adjacency is None:
        if not infer_graph:
            raise ValueError("adjacency must be provided when infer_graph is false")
        adjacency = infer_observation_graph(
            observations,
            method=graph_method,
            absolute=graph_absolute,
            threshold_quantile=graph_threshold_quantile,
            top_k=graph_top_k,
        )
    else:
        adjacency = _prepare_adjacency(adjacency, n_features)
    laplacian = graph_laplacian(adjacency, normalized=normalized_laplacian)
    mean_smoothing_matrix = np.eye(n_features, dtype=float) + graph_strength * laplacian
    variance_smoothing_matrix = np.eye(n_features, dtype=float) + variance_graph_strength * laplacian

    rng = _as_rng(random_state)
    if init_means is None or init_variances is None:
        means0, vars0 = _initialize_gaussian_parameters(observations, n_states, floor_variance, rng)
    means = np.asarray(init_means, dtype=float).copy() if init_means is not None else means0
    variances = np.asarray(init_variances, dtype=float).copy() if init_variances is not None else vars0
    if graph_strength > 0.0:
        means = _smooth_columns(means, mean_smoothing_matrix)
    variances = np.maximum(variances, floor_variance)
    if variance_graph_strength > 0.0:
        variances = _smooth_nonnegative_columns(variances, variance_smoothing_matrix, floor_variance)

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
    smoothing_delta: list[float] = []
    old_loglik = 1.0
    converged = False
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
        mle_means = mean_num / np.maximum(weight_sum[None, :], np.finfo(float).tiny)
        mle_variances = second_num / np.maximum(weight_sum[None, :], np.finfo(float).tiny) - mle_means * mle_means
        unused = weight_sum <= np.finfo(float).tiny
        if np.any(unused):
            mle_means[:, unused] = old_means[:, unused]
            mle_variances[:, unused] = old_variances[:, unused]
        mle_variances = np.maximum(mle_variances, floor_variance)

        means_next = _smooth_columns(mle_means, mean_smoothing_matrix) if graph_strength > 0.0 else mle_means
        variances_next = (
            _smooth_nonnegative_columns(mle_variances, variance_smoothing_matrix, floor_variance)
            if variance_graph_strength > 0.0
            else mle_variances
        )

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

        smooth_delta = float(np.linalg.norm(mle_means - means_next) + np.linalg.norm(mle_variances - variances_next))
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
        smoothing_delta.append(smooth_delta)
        final_iter = iteration

        if verbose and (iteration == 1 or iteration % 25 == 0):
            print(
                f"iter={iteration:04d} loglik={loglik:.3f} "
                f"crit={crit:.3g} smooth={smooth_delta:.3g} min_diag={np.diag(gamma).min():.3f}"
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

    final_loglik = _gaussian_model_log_likelihood(observations, means, variances, gamma, deltas, floor_variance)
    return GraphGaussianHMMResult(
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
        adjacency=adjacency,
        laplacian=laplacian,
        graph_strength=float(graph_strength),
        variance_graph_strength=float(variance_graph_strength),
        normalized_laplacian=bool(normalized_laplacian),
        inferred_graph=bool(inferred_graph),
        smoothing_delta=np.asarray(smoothing_delta, dtype=float),
    )


def fit_graph_gaussian_hmm(
    observations: list[np.ndarray],
    n_states: int,
    **kwargs,
) -> GraphGaussianHMMResult:
    """Fit a non-sticky graph-informed diagonal-covariance Gaussian HMM."""

    kwargs["sticky"] = False
    kwargs.setdefault("threshold", 0.0)
    return fit_sticky_graph_gaussian_hmm(observations, n_states, **kwargs)


def fit_graph_multinoulli_hmm(
    symbols: list[np.ndarray],
    n_states: int,
    *,
    n_symbols: int | None = None,
    adjacency: np.ndarray | None = None,
    graph_strength: float = 0.15,
    normalized_laplacian: bool = True,
    infer_graph: bool = True,
    graph_window: int = 1,
    isolate_last_symbol: bool = True,
    max_iter: int = 1000,
    tol: float = 1e-6,
    loglik_tol: float = 1e-6,
    floor_prob: float = 1e-12,
    init_emissions: np.ndarray | None = None,
    init_gamma: np.ndarray | None = None,
    random_state: int | np.random.Generator | None = None,
    verbose: bool = False,
) -> GraphMultinoulliHMMResult:
    """Fit a graph-informed categorical or Multinoulli HMM."""

    if graph_strength < 0.0:
        raise ValueError("graph_strength must be nonnegative")
    symbols, n_symbols = _validate_symbol_sequences(symbols, n_symbols)

    inferred_graph = adjacency is None
    if adjacency is None:
        if not infer_graph:
            raise ValueError("adjacency must be provided when infer_graph is false")
        adjacency = infer_symbol_transition_graph(
            symbols,
            n_symbols=n_symbols,
            window=graph_window,
            isolate_last_symbol=isolate_last_symbol,
        )
    else:
        adjacency = _prepare_symbol_adjacency(
            adjacency,
            n_symbols,
            isolate_last_symbol=isolate_last_symbol,
        )
    laplacian = graph_laplacian(adjacency, normalized=normalized_laplacian)
    smoothing_matrix = np.eye(n_symbols, dtype=float) + graph_strength * laplacian

    rng = _as_rng(random_state)
    emissions = (
        np.asarray(init_emissions, dtype=float).copy()
        if init_emissions is not None
        else _initialize_emissions(symbols, n_states, n_symbols, rng)
    )
    emissions = np.maximum(emissions, floor_prob)
    emissions = emissions / emissions.sum(axis=1, keepdims=True)
    if graph_strength > 0.0:
        emissions = _smooth_probability_rows(emissions, smoothing_matrix, floor_prob)

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
    smoothing_delta: list[float] = []
    old_loglik = 1.0
    converged = False
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
        mle_emissions = emission_num / np.maximum(emission_den[:, None], np.finfo(float).tiny)
        unused = emission_den <= np.finfo(float).tiny
        if np.any(unused):
            mle_emissions[unused] = old_emissions[unused]
        mle_emissions = np.maximum(mle_emissions, floor_prob)
        mle_emissions = mle_emissions / mle_emissions.sum(axis=1, keepdims=True)
        emissions_next = (
            _smooth_probability_rows(mle_emissions, smoothing_matrix, floor_prob)
            if graph_strength > 0.0
            else mle_emissions
        )

        smooth_delta = float(np.linalg.norm(mle_emissions - emissions_next))
        crit = float(np.linalg.norm(old_gamma - gamma_next) + np.linalg.norm(old_emissions - emissions_next))
        gamma = gamma_next
        emissions = emissions_next
        logliks.append(float(loglik))
        crits.append(crit)
        gamma_diag.append(np.diag(gamma).copy())
        smoothing_delta.append(smooth_delta)
        final_iter = iteration

        if verbose and (iteration == 1 or iteration % 25 == 0):
            print(f"iter={iteration:04d} loglik={loglik:.3f} crit={crit:.3g} smooth={smooth_delta:.3g}")

        if abs(loglik - old_loglik) < loglik_tol and crit < tol:
            converged = True
            break
        old_loglik = loglik

    final_loglik = _multinoulli_model_log_likelihood(symbols, emissions, gamma, deltas, floor_prob)
    return GraphMultinoulliHMMResult(
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
        adjacency=adjacency,
        laplacian=laplacian,
        graph_strength=float(graph_strength),
        normalized_laplacian=bool(normalized_laplacian),
        inferred_graph=bool(inferred_graph),
        smoothing_delta=np.asarray(smoothing_delta, dtype=float),
    )
