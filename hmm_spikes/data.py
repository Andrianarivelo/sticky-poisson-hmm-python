from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat
from scipy.signal import find_peaks


@dataclass
class SpikeDataset:
    """Spike-count dataset arranged as one count matrix per trial.

    Each count matrix has shape ``(n_neurons, n_time_bins)``.
    Firing arrays, when present, use two columns: spike time in seconds and
    one-based neuron id.
    """

    counts: list[np.ndarray]
    time_edges: np.ndarray
    bin_size: float
    n_neurons: int
    name: str
    firings: list[np.ndarray] | None = None
    neuron_ids: np.ndarray | None = None
    true_state_sequences: list[np.ndarray] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_trials(self) -> int:
        return len(self.counts)

    @property
    def n_bins(self) -> int:
        return int(self.counts[0].shape[1]) if self.counts else 0


def _as_trial_list(value: Any) -> list[Any]:
    return list(np.atleast_1d(value).ravel())


def _safe_scalar(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.ravel()[0].item()
    return value


def _make_time_edges(start: float, end: float, bin_size: float) -> np.ndarray:
    n_bins = int(round((end - start) / bin_size))
    edges = start + bin_size * np.arange(n_bins + 1, dtype=float)
    edges[-1] = end
    return edges


def spike_count(
    firings: np.ndarray,
    time_edges: np.ndarray,
    neuron_ids: np.ndarray,
) -> np.ndarray:
    """Count spikes for selected neurons in fixed time bins."""

    n_neurons = len(neuron_ids)
    counts = np.zeros((n_neurons, len(time_edges) - 1), dtype=np.uint16)
    if firings is None or np.asarray(firings).size == 0:
        return counts

    firings = np.asarray(firings, dtype=float)
    for row, neuron_id in enumerate(neuron_ids):
        spike_times = firings[firings[:, 1] == neuron_id, 0]
        counts[row] = np.histogram(spike_times, bins=time_edges)[0].astype(np.uint16)
    return counts


def spike_times_to_counts(spike_times: np.ndarray, time_edges: np.ndarray) -> np.ndarray:
    """Convert one spike-time array into a ``(1, n_time_bins)`` count matrix."""

    spike_times = np.asarray(spike_times, dtype=float).ravel()
    counts = np.histogram(spike_times, bins=np.asarray(time_edges, dtype=float))[0]
    return counts[None, :].astype(np.uint16)


def spike_trains_to_counts(spike_trains: list[np.ndarray] | np.ndarray, time_edges: np.ndarray) -> np.ndarray:
    """Convert one or more spike trains into a count matrix.

    ``spike_trains`` may be a list of one-dimensional spike-time arrays or a
    single one-dimensional spike-time array. Rows in the output correspond to
    neurons or units.
    """

    if isinstance(spike_trains, np.ndarray) and spike_trains.ndim == 1:
        return spike_times_to_counts(spike_trains, time_edges)

    trains = [np.asarray(train, dtype=float).ravel() for train in spike_trains]
    rows = [np.histogram(train, bins=time_edges)[0] for train in trains]
    return np.asarray(rows, dtype=np.uint16)


def firings_to_counts(
    firings: np.ndarray,
    time_edges: np.ndarray,
    *,
    n_neurons: int | None = None,
    neuron_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Convert a two-column ``[spike_time, neuron_id]`` array to counts."""

    firings = np.asarray(firings, dtype=float)
    if neuron_ids is None:
        if n_neurons is None:
            n_neurons = int(np.max(firings[:, 1])) if firings.size else 1
        neuron_ids = np.arange(1, n_neurons + 1, dtype=int)
    return spike_count(firings, np.asarray(time_edges, dtype=float), np.asarray(neuron_ids, dtype=int))


def signal_to_event_counts(
    signal: np.ndarray,
    sampling_rate: float,
    bin_size: float,
    *,
    threshold_z: float = 2.5,
    refractory: float = 0.2,
    start_time: float = 0.0,
    polarity: str = "positive",
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Convert one or more continuous signals to event-count observations.

    This is intended for data such as fiber photometry only after accepting a
    modeling choice: the sticky-Poisson HMM is being applied to detected event
    counts, not to the raw continuous fluorescence signal.

    Parameters
    ----------
    signal:
        One signal with shape ``(n_samples,)`` or multiple channels/signals with
        shape ``(n_signals, n_samples)``.
    sampling_rate:
        Samples per second.
    bin_size:
        Output HMM bin size in seconds.
    threshold_z:
        Robust z-score threshold used for event detection.
    refractory:
        Minimum distance between detected events in seconds.
    polarity:
        ``"positive"`` for positive transients, ``"negative"`` for negative
        transients.
    """

    y = np.asarray(signal, dtype=float)
    if y.ndim == 1:
        y = y[None, :]
    if y.ndim != 2:
        raise ValueError("signal must have shape (n_samples,) or (n_signals, n_samples)")
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive")
    if bin_size <= 0:
        raise ValueError("bin_size must be positive")

    n_samples = y.shape[1]
    duration = n_samples / float(sampling_rate)
    n_bins = int(np.ceil(duration / bin_size))
    time_edges = start_time + bin_size * np.arange(n_bins + 1, dtype=float)
    time_edges[-1] = start_time + duration
    min_distance = max(1, int(round(refractory * sampling_rate)))

    event_times: list[np.ndarray] = []
    for row in y:
        center = np.nanmedian(row)
        mad = np.nanmedian(np.abs(row - center))
        scale = 1.4826 * mad if mad > 0 else np.nanstd(row)
        if not np.isfinite(scale) or scale <= 0:
            event_times.append(np.array([], dtype=float))
            continue
        z = (row - center) / scale
        if polarity == "negative":
            z = -z
        elif polarity != "positive":
            raise ValueError("polarity must be 'positive' or 'negative'")
        peaks, _ = find_peaks(z, height=threshold_z, distance=min_distance)
        event_times.append(start_time + peaks / float(sampling_rate))

    return spike_trains_to_counts(event_times, time_edges), time_edges, event_times


def load_mat_dataset(
    path: str | Path,
    *,
    bin_size: float | None = None,
    neuron_ids: np.ndarray | None = None,
) -> SpikeDataset:
    """Load one of the MATLAB datasets supplied by the original package.

    Supported formats:
    - ``exampledata.mat`` with ``data(k).firings``.
    - MMPP files with precomputed ``spkc`` cells and optional ``MMPP`` firings.
    """

    path = Path(path)
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)

    if "spkc" in mat:
        counts = [np.asarray(x, dtype=np.uint16) for x in _as_trial_list(mat["spkc"])]
        dt = float(_safe_scalar(mat.get("pHMMdt", bin_size or 1.0)))
        if "pHMMtimev" in mat:
            time_edges = np.asarray(mat["pHMMtimev"], dtype=float).ravel()
        else:
            time_edges = dt * np.arange(counts[0].shape[1] + 1, dtype=float)
        n_neurons = int(counts[0].shape[0])
        ids = np.arange(1, n_neurons + 1, dtype=int)
        firings = None
        true_states = None
        if "MMPP" in mat:
            trials = _as_trial_list(mat["MMPP"])
            firings = [np.asarray(t.firings, dtype=float) for t in trials]
            true_states = [np.asarray(t.stateSeq, dtype=float) for t in trials]
        metadata = {
            "source": str(path),
            "description": str(mat.get("MMPPdata_description", "")),
        }
        return SpikeDataset(
            counts=counts,
            time_edges=time_edges,
            bin_size=dt,
            n_neurons=n_neurons,
            name=path.stem,
            firings=firings,
            neuron_ids=ids,
            true_state_sequences=true_states,
            metadata=metadata,
        )

    if "data" not in mat:
        raise ValueError(f"Unsupported MATLAB dataset: {path}")

    n_neurons = int(_safe_scalar(mat["N"]))
    start = float(_safe_scalar(mat["starttime"]))
    end = float(_safe_scalar(mat["endtime"]))
    dt = 0.05 if bin_size is None else float(bin_size)
    time_edges = _make_time_edges(start, end, dt)
    ids = np.arange(1, n_neurons + 1, dtype=int) if neuron_ids is None else np.asarray(neuron_ids, dtype=int)

    data_trials = _as_trial_list(mat["data"])
    firings = [np.asarray(t.firings, dtype=float) for t in data_trials]
    counts = [spike_count(f, time_edges, ids) for f in firings]
    return SpikeDataset(
        counts=counts,
        time_edges=time_edges,
        bin_size=dt,
        n_neurons=len(ids),
        name=path.stem,
        firings=firings,
        neuron_ids=ids,
        metadata={"source": str(path), "starttime": start, "endtime": end},
    )


def save_npz_dataset(dataset: SpikeDataset, path: str | Path) -> Path:
    """Save a ``SpikeDataset`` to a portable NumPy archive."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = np.stack(dataset.counts, axis=0)

    arrays: dict[str, Any] = {
        "counts": counts,
        "time_edges": dataset.time_edges,
        "bin_size": np.array(dataset.bin_size),
        "n_neurons": np.array(dataset.n_neurons),
        "neuron_ids": dataset.neuron_ids
        if dataset.neuron_ids is not None
        else np.arange(1, dataset.n_neurons + 1, dtype=int),
        "name": np.array(dataset.name),
    }

    if dataset.firings is not None:
        trial_index = []
        spike_time = []
        spike_neuron = []
        for trial, firings in enumerate(dataset.firings):
            if firings is None or np.asarray(firings).size == 0:
                continue
            f = np.asarray(firings, dtype=float)
            trial_index.append(np.full(f.shape[0], trial, dtype=np.int32))
            spike_time.append(f[:, 0].astype(float))
            spike_neuron.append(f[:, 1].astype(np.int32))
        if trial_index:
            arrays["spike_trial"] = np.concatenate(trial_index)
            arrays["spike_time"] = np.concatenate(spike_time)
            arrays["spike_neuron"] = np.concatenate(spike_neuron)

    if dataset.true_state_sequences is not None:
        trial_index = []
        state_time = []
        state_label = []
        for trial, seq in enumerate(dataset.true_state_sequences):
            if seq is None or np.asarray(seq).size == 0:
                continue
            seq = np.asarray(seq, dtype=float)
            trial_index.append(np.full(seq.shape[1], trial, dtype=np.int32))
            state_time.append(seq[0].astype(float))
            state_label.append(seq[1].astype(np.int32))
        if trial_index:
            arrays["state_trial"] = np.concatenate(trial_index)
            arrays["state_time"] = np.concatenate(state_time)
            arrays["state_label"] = np.concatenate(state_label)

    np.savez_compressed(path, **arrays)
    return path


def load_npz_dataset(path: str | Path) -> SpikeDataset:
    """Load a dataset saved by ``save_npz_dataset``."""

    path = Path(path)
    archive = np.load(path, allow_pickle=False)
    counts = [np.asarray(x, dtype=np.uint16) for x in archive["counts"]]
    n_trials = len(counts)
    firings = None
    if {"spike_trial", "spike_time", "spike_neuron"}.issubset(archive.files):
        firings = []
        spike_trial = archive["spike_trial"]
        spike_time = archive["spike_time"]
        spike_neuron = archive["spike_neuron"]
        for trial in range(n_trials):
            mask = spike_trial == trial
            f = np.column_stack([spike_time[mask], spike_neuron[mask]])
            firings.append(f.astype(float))

    true_states = None
    if {"state_trial", "state_time", "state_label"}.issubset(archive.files):
        true_states = []
        state_trial = archive["state_trial"]
        state_time = archive["state_time"]
        state_label = archive["state_label"]
        for trial in range(n_trials):
            mask = state_trial == trial
            true_states.append(np.vstack([state_time[mask], state_label[mask]]))

    return SpikeDataset(
        counts=counts,
        time_edges=np.asarray(archive["time_edges"], dtype=float),
        bin_size=float(archive["bin_size"]),
        n_neurons=int(archive["n_neurons"]),
        name=str(archive["name"]),
        firings=firings,
        neuron_ids=np.asarray(archive["neuron_ids"], dtype=int),
        true_state_sequences=true_states,
        metadata={"source": str(path)},
    )
