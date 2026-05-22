from __future__ import annotations

from itertools import product

import numpy as np


def state_sequence_metrics(states: np.ndarray, bin_size: float) -> dict[str, float]:
    """Summarize a decoded state sequence."""

    states = np.asarray(states, dtype=np.int64).ravel()
    if states.size == 0:
        return {
            "switches_per_min": float("nan"),
            "mean_state_duration_s": float("nan"),
            "state_entropy_bits": float("nan"),
        }

    switches = int(np.sum(states[1:] != states[:-1]))
    duration_min = states.size * float(bin_size) / 60.0
    switches_per_min = switches / duration_min if duration_min > 0 else float("nan")

    boundaries = np.r_[0, np.flatnonzero(states[1:] != states[:-1]) + 1, states.size]
    segment_lengths = np.diff(boundaries) * float(bin_size)
    mean_duration = float(np.mean(segment_lengths)) if segment_lengths.size else float("nan")

    _, counts = np.unique(states, return_counts=True)
    probs = counts / counts.sum()
    entropy = float(-np.sum(probs * np.log2(probs + np.finfo(float).tiny)))

    return {
        "switches_per_min": float(switches_per_min),
        "mean_state_duration_s": mean_duration,
        "state_entropy_bits": entropy,
    }


def exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    """Two-sided exact paired sign-flip p value for small n."""

    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    differences = differences[differences != 0]
    n = differences.size
    if n == 0:
        return 1.0

    observed = abs(float(np.mean(differences)))
    signs = np.asarray(list(product([-1.0, 1.0], repeat=n)))
    null = np.abs((signs * differences[None, :]).mean(axis=1))
    return float(np.mean(null >= observed - 1e-15))


def benjamini_hochberg_fdr(p_values: np.ndarray) -> np.ndarray:
    """Return Benjamini-Hochberg q values."""

    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not np.any(valid):
        return q

    p_valid = p[valid]
    order = np.argsort(p_valid)
    ranked = p_valid[order]
    m = ranked.size
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    q_valid = np.empty_like(adjusted)
    q_valid[order] = adjusted
    q[valid] = q_valid
    return q


def paired_condition_table(condition_a: dict[str, np.ndarray], condition_b: dict[str, np.ndarray]) -> list[dict[str, float | str]]:
    """Compare paired metrics between two conditions.

    Inputs map metric names to paired arrays with one value per subject.
    """

    rows = []
    shared = sorted(set(condition_a) & set(condition_b))
    p_values = []
    for metric in shared:
        a = np.asarray(condition_a[metric], dtype=float)
        b = np.asarray(condition_b[metric], dtype=float)
        if a.shape != b.shape:
            raise ValueError(f"metric {metric!r} has mismatched paired shapes")
        diff = b - a
        p = exact_sign_flip_pvalue(diff)
        p_values.append(p)
        rows.append(
            {
                "metric": metric,
                "mean_a": float(np.nanmean(a)),
                "mean_b": float(np.nanmean(b)),
                "mean_difference_b_minus_a": float(np.nanmean(diff)),
                "p_sign_flip": float(p),
                "q_fdr": float("nan"),
                "n": int(np.sum(np.isfinite(diff))),
            }
        )

    q_values = benjamini_hochberg_fdr(np.asarray(p_values, dtype=float))
    for row, q in zip(rows, q_values):
        row["q_fdr"] = float(q)
    return rows
