from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from .phmm import PHMMResult, state_probabilities, viterbi_decode


def default_state_colors(n_states: int) -> np.ndarray:
    base = np.array(
        [
            [0.9961, 0.6445, 0.0000],
            [0.8500, 0.3250, 0.0980],
            [0.0000, 0.0000, 1.0000],
            [0.1198, 0.3620, 0.2266],
            [0.1172, 0.5625, 0.9961],
            [0.8000, 0.0000, 0.0000],
            [0.7520, 0.8750, 0.0000],
            [0.7194, 0.5208, 0.7194],
        ],
        dtype=float,
    )
    if n_states <= len(base):
        return base[:n_states]
    extra = plt.get_cmap("tab20")(np.linspace(0, 1, n_states - len(base)))[:, :3]
    return np.vstack([base, extra])


def _save(fig: plt.Figure, path: str | Path | None) -> None:
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")


def _plot_raster(ax: plt.Axes, firings: np.ndarray | None, n_neurons: int, start: float, end: float) -> None:
    if firings is None or np.asarray(firings).size == 0:
        ax.set_ylim(0.5, n_neurons + 0.5)
        return
    firings = np.asarray(firings, dtype=float)
    mask = (firings[:, 0] >= start) & (firings[:, 0] <= end)
    f = firings[mask]
    ax.vlines(f[:, 0], f[:, 1] - 0.38, f[:, 1] + 0.38, color="black", linewidth=0.6, alpha=0.9)
    ax.set_ylim(0.5, n_neurons + 0.5)


def _segments_from_labels(labels: np.ndarray, valid: np.ndarray) -> list[tuple[int, int, int]]:
    segments: list[tuple[int, int, int]] = []
    start = None
    current = None
    for i, label in enumerate(labels):
        if not valid[i]:
            if start is not None:
                segments.append((start, i, int(current)))
                start = None
                current = None
            continue
        if start is None:
            start = i
            current = label
        elif label != current:
            segments.append((start, i, int(current)))
            start = i
            current = label
    if start is not None:
        segments.append((start, len(labels), int(current)))
    return segments


def plot_training_diagnostics(
    result: PHMMResult,
    *,
    path: str | Path | None = None,
    colors: np.ndarray | None = None,
) -> plt.Figure:
    n_states = result.gamma.shape[0]
    colors = default_state_colors(n_states) if colors is None else colors
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=False)

    x = np.arange(result.history.gamma_diag.shape[1])
    for state in range(n_states):
        axes[0].plot(x, result.history.gamma_diag[state], color=colors[state], lw=2, label=f"State {state + 1}")
    axes[0].axhline(result.threshold, color="0.2", lw=1.2, ls=":")
    axes[0].set_ylabel("self transition probability")
    axes[0].set_title("Self transition probability during sticky Poisson HMM training")
    axes[0].legend(ncol=min(n_states, 5), fontsize=8, frameon=False)
    axes[0].set_ylim(0, 1.02)

    axes[1].plot(np.arange(1, len(result.history.log_likelihood) + 1), result.history.log_likelihood, color="0.1", lw=2)
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("log-likelihood")
    axes[1].set_title("Training log-likelihood")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, axis="y", color="0.9", lw=0.8)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_trial_decoding(
    firings: np.ndarray | None,
    counts: np.ndarray,
    result: PHMMResult,
    time_edges: np.ndarray,
    *,
    trial_number: int = 1,
    probability_threshold: float = 0.8,
    path: str | Path | None = None,
    colors: np.ndarray | None = None,
) -> plt.Figure:
    n_states = result.gamma.shape[0]
    n_neurons = counts.shape[0]
    colors = default_state_colors(n_states) if colors is None else colors
    start, end = float(time_edges[0]), float(time_edges[-1])
    times = time_edges[:-1]
    probs = state_probabilities(counts, result.means, result.gamma)
    max_probs = probs.max(axis=0)
    labels = probs.argmax(axis=0)
    viterbi = viterbi_decode(counts, result.means, result.gamma)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    for ax, title, state_labels, valid in [
        (
            axes[0],
            f"Posterior decoding, trial {trial_number}",
            labels,
            max_probs >= probability_threshold,
        ),
        (
            axes[1],
            "Viterbi decoding",
            viterbi,
            np.ones_like(viterbi, dtype=bool),
        ),
    ]:
        for a, b, state in _segments_from_labels(state_labels, valid):
            ax.axvspan(time_edges[a], time_edges[b], color=colors[state], alpha=0.32, lw=0)
        _plot_raster(ax, firings, n_neurons, start, end)
        ax.set_ylabel("neuron index")
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)

    prob_ax = axes[0].twinx()
    for state in range(n_states):
        prob_ax.plot(times, probs[state], color=colors[state], lw=1.3, alpha=0.95)
    prob_ax.set_ylim(0, 1)
    prob_ax.set_ylabel("state probability")
    prob_ax.spines["top"].set_visible(False)

    axes[1].set_xlabel("time (sec)")
    axes[1].set_xlim(start, end)
    fig.suptitle(
        f"sticky Poisson HMM, N={n_neurons}, m={n_states}, dt={result.bin_size * 1000:.0f} ms",
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_all_trials_summary(
    counts: list[np.ndarray],
    result: PHMMResult,
    time_edges: np.ndarray,
    *,
    train_cut: int | None = None,
    probability_threshold: float = 0.8,
    path: str | Path | None = None,
    colors: np.ndarray | None = None,
) -> plt.Figure:
    n_states = result.gamma.shape[0]
    n_neurons = counts[0].shape[0]
    colors = default_state_colors(n_states) if colors is None else colors

    fig = plt.figure(figsize=(max(12, n_states * 1.6), 8))
    grid = fig.add_gridspec(3, n_states, height_ratios=[2.5, 0.08, 1.2], hspace=0.35, wspace=0.35)
    ax = fig.add_subplot(grid[0, :])

    for trial, x in enumerate(counts):
        probs = state_probabilities(x, result.means, result.gamma)
        max_probs = probs.max(axis=0)
        labels = probs.argmax(axis=0)
        valid = max_probs >= probability_threshold
        for a, b, state in _segments_from_labels(labels, valid):
            rect = Rectangle(
                (time_edges[a], trial + 1),
                time_edges[b] - time_edges[a],
                0.9,
                facecolor=colors[state],
                edgecolor="none",
                alpha=0.55,
            )
            ax.add_patch(rect)

    if train_cut is not None:
        ax.axhline(train_cut + 1, color="black", lw=1.5)
    ax.set_xlim(float(time_edges[0]), float(time_edges[-1]))
    ax.set_ylim(1, len(counts) + 1)
    ax.set_xlabel("time (sec)")
    ax.set_ylabel("trial index")
    ax.set_title("Posterior decoding across trials")
    ax.spines[["top", "right"]].set_visible(False)

    max_rate = max(float(result.rates_hz.max()), 1.0)
    for state in range(n_states):
        rate_ax = fig.add_subplot(grid[2, state])
        rate_ax.barh(np.arange(1, n_neurons + 1), result.rates_hz[:, state], color=colors[state], alpha=0.75)
        rate_ax.set_xlim(0, max_rate * 1.08)
        rate_ax.invert_yaxis()
        rate_ax.set_title(f"State {state + 1}", fontsize=10)
        if state == 0:
            rate_ax.set_ylabel("neuron")
        else:
            rate_ax.set_yticklabels([])
        rate_ax.set_xlabel("Hz")
        rate_ax.tick_params(labelsize=8)
        rate_ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("sticky Poisson HMM trial summary and state firing rates", fontweight="bold")
    _save(fig, path)
    return fig
