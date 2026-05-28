"""Data IO: canonical dataset, multi-format loaders, demos, exporters.

A :class:`Dataset` is a list of 2D trial arrays oriented as
``(n_channels, n_time_bins)``, tagged with a ``kind`` of ``"counts"`` or
``"continuous"``, plus a ``bin_size`` in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime as _dt
import gzip
from pathlib import Path
import pickle
from typing import Any

import numpy as np


PROJECT_FORMAT = "hmm_studio_project"
PROJECT_VERSION = 1


# ----------------------------------------------------------------------------
# Canonical dataset
# ----------------------------------------------------------------------------
@dataclass
class Dataset:
    name: str
    trials: list[np.ndarray]          # each (n_channels, n_bins)
    kind: str                         # 'counts' | 'continuous'
    bin_size: float
    channel_labels: list[str] = field(default_factory=list)
    source: str = ""
    sampling_rate: float | None = None

    @property
    def n_trials(self) -> int:
        return len(self.trials)

    @property
    def n_channels(self) -> int:
        return int(self.trials[0].shape[0]) if self.trials else 0

    @property
    def n_bins(self) -> int:
        return int(self.trials[0].shape[1]) if self.trials else 0

    @property
    def duration_s(self) -> float:
        return self.n_bins * float(self.bin_size)

    def time_axis(self, trial_idx: int = 0) -> np.ndarray:
        n = int(self.trials[trial_idx].shape[1])
        return np.arange(n, dtype=float) * self.bin_size

    def time_edges(self, trial_idx: int = 0) -> np.ndarray:
        n = int(self.trials[trial_idx].shape[1])
        return np.arange(n + 1, dtype=float) * self.bin_size

    def info(self) -> str:
        return (f"{self.n_trials} trial(s) · {self.n_channels} channels · "
                f"{self.n_bins} bins · Δt={self.bin_size*1000:.1f} ms · "
                f"{self.duration_s:.1f} s · kind={self.kind}")


# ----------------------------------------------------------------------------
# Raw-load result before orientation / kind are finalized
# ----------------------------------------------------------------------------
@dataclass
class RawLoad:
    arrays: list[np.ndarray]
    name: str
    suggested_kind: str
    suggested_bin_size: float | None
    note: str = ""


def _guess_kind(a: np.ndarray) -> str:
    a = np.asarray(a)
    if not np.issubdtype(a.dtype, np.number):
        return "continuous"
    if a.size == 0:
        return "continuous"
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return "continuous"
    if np.all(finite >= 0) and np.all(np.equal(np.mod(finite, 1.0), 0.0)) and np.max(finite) < 1e5:
        return "counts"
    return "continuous"


def _read_csv(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    # Pick delimiter from the first non-empty line.
    line = next((ln for ln in text.splitlines() if ln.strip()), "")
    for d in [",", "\t", ";", " "]:
        if d in line:
            delim = d
            break
    else:
        delim = ","
    # Detect a header row by trying to parse the first row.
    rows = [ln for ln in text.splitlines() if ln.strip()]
    skip = 0
    try:
        [float(x) for x in rows[0].replace(delim, " ").split()]
    except ValueError:
        skip = 1
    arr = np.genfromtxt(path, delimiter=delim, skip_header=skip)
    if arr.ndim == 1:
        arr = arr[None, :]
    return np.asarray(arr, dtype=float)


def read_raw(path: str | Path) -> RawLoad:
    """Read a file and return raw arrays + heuristic suggestions."""

    path = Path(path)
    name = path.stem
    suffix = path.suffix.lower()

    if suffix in {".npz"}:
        arch = np.load(path, allow_pickle=False)
        # hmm_spikes SpikeDataset npz
        if "counts" in arch.files:
            counts = np.asarray(arch["counts"])
            trials = [np.asarray(c, dtype=float) for c in counts]
            dt = float(arch["bin_size"]) if "bin_size" in arch.files else 0.05
            return RawLoad(trials, name=name, suggested_kind="counts",
                           suggested_bin_size=dt,
                           note="Recognised hmm_spikes count archive.")
        arrays = []
        for key in sorted(arch.files):
            a = np.asarray(arch[key])
            if a.dtype.kind in "fiub" and a.ndim in (1, 2):
                arrays.append(a.astype(float))
        if not arrays:
            raise ValueError("No numeric 1D/2D arrays found in npz.")
        return RawLoad(arrays, name=name,
                       suggested_kind=_guess_kind(arrays[0]),
                       suggested_bin_size=None,
                       note=f"Loaded {len(arrays)} array(s) from npz.")

    if suffix == ".npy":
        a = np.asarray(np.load(path, allow_pickle=False), dtype=float)
        return RawLoad([a], name=name, suggested_kind=_guess_kind(a),
                       suggested_bin_size=None, note="Loaded npy array.")

    if suffix == ".mat":
        try:
            from hmm_spikes.data import load_mat_dataset

            ds = load_mat_dataset(path)
            trials = [np.asarray(c, dtype=float) for c in ds.counts]
            return RawLoad(trials, name=ds.name or name, suggested_kind="counts",
                           suggested_bin_size=ds.bin_size,
                           note="Recognised hmm_spikes MATLAB layout.")
        except Exception as exc:
            from scipy.io import loadmat

            mat = loadmat(path, squeeze_me=True, struct_as_record=False)
            arrays = []
            for key, val in mat.items():
                if key.startswith("__"):
                    continue
                a = np.asarray(val)
                if a.dtype.kind in "fiub" and a.ndim in (1, 2):
                    arrays.append(a.astype(float))
            if not arrays:
                raise ValueError(f"Could not extract numeric arrays from {path}: {exc}")
            return RawLoad(arrays, name=name,
                           suggested_kind=_guess_kind(arrays[0]),
                           suggested_bin_size=None,
                           note=f"Loaded {len(arrays)} numeric array(s) from MAT.")

    if suffix in {".csv", ".tsv", ".txt", ".dat"}:
        a = _read_csv(path)
        return RawLoad([a], name=name, suggested_kind=_guess_kind(a),
                       suggested_bin_size=None,
                       note=f"Loaded CSV array shape={a.shape}.")

    raise ValueError(f"Unsupported file extension: {suffix}")


# ----------------------------------------------------------------------------
# Orientation and dataset assembly
# ----------------------------------------------------------------------------
def _orient(a: np.ndarray, time_axis: str) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return a[None, :]
    if a.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got shape {a.shape}.")
    if time_axis == "rows":
        return a.T
    if time_axis == "columns":
        return a
    # auto: channels = shorter axis
    return a if a.shape[0] <= a.shape[1] else a.T


def build_dataset(raw: RawLoad, *, kind: str, bin_size: float,
                  time_axis: str = "auto",
                  channel_labels: list[str] | None = None,
                  name: str | None = None,
                  source: str = "") -> Dataset:
    trials = [_orient(a, time_axis) for a in raw.arrays]
    n_channels = trials[0].shape[0]
    # Force uniform channel counts across trials by trimming the trailing axis if needed.
    bin_counts = [t.shape[1] for t in trials]
    if any(t.shape[0] != n_channels for t in trials):
        raise ValueError("All trials must have the same number of channels.")
    if kind == "counts":
        trials = [np.round(t).clip(min=0).astype(float) for t in trials]
    labels = channel_labels or [f"Ch {i+1}" for i in range(n_channels)]
    return Dataset(name=name or raw.name, trials=trials, kind=kind,
                   bin_size=float(bin_size), channel_labels=labels, source=source)


# ----------------------------------------------------------------------------
# Demo data generators
# ----------------------------------------------------------------------------
def demo_spike_counts(*, n_neurons: int = 12, n_states: int = 3,
                      n_bins: int = 1600, bin_size: float = 0.05,
                      n_trials: int = 3, dwell_p: float = 0.92,
                      seed: int = 7) -> Dataset:
    """Synthetic sticky Poisson dataset with clear latent states."""

    rng = np.random.default_rng(seed)
    # State firing-rate maps (Hz), each state highlights a different sub-population.
    rates = np.zeros((n_neurons, n_states), dtype=float)
    for k in range(n_states):
        base = rng.uniform(1.0, 4.0, size=n_neurons)
        hot = rng.choice(n_neurons, size=max(2, n_neurons // n_states), replace=False)
        base[hot] = rng.uniform(15.0, 35.0, size=hot.size)
        rates[:, k] = base
    means = rates * bin_size  # expected counts per bin

    # Build a sticky transition matrix.
    off = (1.0 - dwell_p) / max(n_states - 1, 1)
    gamma = np.full((n_states, n_states), off, dtype=float)
    np.fill_diagonal(gamma, dwell_p)

    trials = []
    for _ in range(n_trials):
        state = rng.integers(0, n_states)
        seq = np.empty(n_bins, dtype=np.int32)
        for t in range(n_bins):
            seq[t] = state
            state = int(rng.choice(n_states, p=gamma[state]))
        # Sample Poisson counts conditional on the latent state.
        lam = means[:, seq]  # (n_neurons, n_bins)
        counts = rng.poisson(lam).astype(float)
        trials.append(counts)

    return Dataset(name="demo_counts", trials=trials, kind="counts",
                   bin_size=bin_size,
                   channel_labels=[f"N{i+1:02d}" for i in range(n_neurons)],
                   source="demo")


def demo_continuous(*, n_channels: int = 4, n_states: int = 3,
                    n_bins: int = 4000, bin_size: float = 0.01,
                    n_trials: int = 2, dwell_p: float = 0.985,
                    seed: int = 11) -> Dataset:
    """Synthetic continuous (photometry-like) dataset."""

    rng = np.random.default_rng(seed)
    means = np.zeros((n_channels, n_states))
    sigmas = np.zeros((n_channels, n_states))
    for k in range(n_states):
        means[:, k] = rng.uniform(-1.5, 1.5, size=n_channels)
        sigmas[:, k] = rng.uniform(0.25, 0.7, size=n_channels)

    off = (1.0 - dwell_p) / max(n_states - 1, 1)
    gamma = np.full((n_states, n_states), off, dtype=float)
    np.fill_diagonal(gamma, dwell_p)

    trials = []
    for _ in range(n_trials):
        state = rng.integers(0, n_states)
        seq = np.empty(n_bins, dtype=np.int32)
        for t in range(n_bins):
            seq[t] = state
            state = int(rng.choice(n_states, p=gamma[state]))
        signal = means[:, seq] + sigmas[:, seq] * rng.standard_normal((n_channels, n_bins))
        # smooth slightly for that photometry look
        kernel = np.exp(-np.linspace(-2, 2, 9) ** 2)
        kernel = kernel / kernel.sum()
        for c in range(n_channels):
            signal[c] = np.convolve(signal[c], kernel, mode="same")
        trials.append(signal)
    return Dataset(name="demo_photometry", trials=trials, kind="continuous",
                   bin_size=bin_size,
                   channel_labels=[f"Ch {i+1}" for i in range(n_channels)],
                   source="demo")


# ----------------------------------------------------------------------------
# Continuous -> event counts preprocessing
# ----------------------------------------------------------------------------
def events_from_continuous(dataset: Dataset, *, threshold_z: float = 2.5,
                           refractory_s: float = 0.2,
                           polarity: str = "positive") -> Dataset:
    """Convert a continuous Dataset into a count Dataset via transient detection."""

    if dataset.kind != "continuous":
        raise ValueError("events_from_continuous requires a continuous dataset.")
    from hmm_spikes.data import signal_to_event_counts

    sampling_rate = dataset.sampling_rate or (1.0 / dataset.bin_size)
    out_trials = []
    for trial in dataset.trials:
        counts, _edges, _events = signal_to_event_counts(
            trial, sampling_rate=sampling_rate,
            bin_size=dataset.bin_size,
            threshold_z=threshold_z,
            refractory=refractory_s, polarity=polarity,
        )
        out_trials.append(counts.astype(float))
    return Dataset(name=dataset.name + " (events)", trials=out_trials,
                   kind="counts", bin_size=dataset.bin_size,
                   channel_labels=list(dataset.channel_labels),
                   source=dataset.source + " (events)")


# ----------------------------------------------------------------------------
# Exporters
# ----------------------------------------------------------------------------
def export_states_csv(path: str | Path, run, decodes: list[tuple[np.ndarray, np.ndarray]],
                      dataset: Dataset) -> Path:
    """Per-trial / per-bin posterior state, max-prob and Viterbi state."""

    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["trial", "bin", "time_s", "posterior_state",
                    "max_posterior", "viterbi_state"])
        for ti, (post, vit) in enumerate(decodes):
            t_axis = dataset.time_axis(ti)
            labels = post.argmax(axis=0)
            maxp = post.max(axis=0)
            for b in range(post.shape[1]):
                w.writerow([ti + 1, b + 1, f"{t_axis[b]:.6f}",
                            int(labels[b]) + 1, f"{maxp[b]:.6f}",
                            int(vit[b]) + 1])
    return path


def export_posteriors_npz(path: str | Path,
                          decodes: list[tuple[np.ndarray, np.ndarray]]) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for ti, (post, vit) in enumerate(decodes):
        payload[f"posterior_trial_{ti+1}"] = post.astype(np.float32)
        payload[f"viterbi_trial_{ti+1}"] = vit.astype(np.int16)
    np.savez_compressed(path, **payload)
    return path


def export_model_npz(path: str | Path, run) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    res = run.result
    payload: dict[str, np.ndarray] = {
        "gamma": np.asarray(res.gamma, dtype=np.float64),
        "log_likelihood": np.asarray([res.log_likelihood], dtype=np.float64),
        "n_states": np.asarray([run.n_states], dtype=np.int32),
    }
    if hasattr(res, "means"): payload["means"] = np.asarray(res.means, dtype=np.float64)
    if hasattr(res, "variances"): payload["variances"] = np.asarray(res.variances, dtype=np.float64)
    if hasattr(res, "emissions"): payload["emissions"] = np.asarray(res.emissions, dtype=np.float64)
    if hasattr(res, "deltas"): payload["deltas"] = np.asarray(res.deltas, dtype=np.float64)
    if hasattr(res, "bin_size"): payload["bin_size"] = np.asarray([float(res.bin_size)], dtype=np.float64)
    if hasattr(res, "history"):
        h = res.history
        payload["history_loglik"] = np.asarray(h.log_likelihood, dtype=np.float64)
        payload["history_gamma_diag"] = np.asarray(h.gamma_diag, dtype=np.float64)
    np.savez_compressed(path, **payload)
    return path


# ----------------------------------------------------------------------------
# Project files
# ----------------------------------------------------------------------------
def save_project(path: str | Path, *, dataset: Dataset, runs: list,
                 active_run_id: str | None = None, bic_result: Any = None,
                 trial_index: int = 0, ui_state: dict | None = None) -> Path:
    """Save a complete Studio project, including fitted model objects.

    The project format is intentionally a trusted local workspace format. It
    uses pickle inside gzip so large fitted arrays survive a round trip without
    forcing the user to recompute EM fits.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": PROJECT_FORMAT,
        "version": PROJECT_VERSION,
        "saved_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset,
        "runs": list(runs),
        "active_run_id": active_run_id,
        "bic_result": bic_result,
        "trial_index": int(trial_index),
        "ui_state": dict(ui_state or {}),
    }
    with gzip.open(path, "wb", compresslevel=5) as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_project(path: str | Path) -> dict:
    """Load a trusted HMM Studio project file."""

    path = Path(path)
    with gzip.open(path, "rb") as fh:
        payload = pickle.load(fh)
    if not isinstance(payload, dict) or payload.get("format") != PROJECT_FORMAT:
        raise ValueError("This is not an HMM Studio project file.")
    version = int(payload.get("version", 0))
    if version > PROJECT_VERSION:
        raise ValueError(
            f"Project version {version} is newer than this HMM Studio build."
        )
    if not isinstance(payload.get("dataset"), Dataset):
        raise ValueError("Project file does not contain a valid dataset.")
    payload.setdefault("runs", [])
    payload.setdefault("active_run_id", None)
    payload.setdefault("bic_result", None)
    payload.setdefault("trial_index", 0)
    payload.setdefault("ui_state", {})
    return payload
