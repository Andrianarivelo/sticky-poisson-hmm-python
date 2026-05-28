"""Background workers: model fitting and BIC scans run off the GUI thread.

The :class:`FitWorker` runs ``n_restarts`` random-seed restarts and picks the
best converged + threshold-satisfied fit. The :class:`BICScanWorker` does the
same across a range of candidate state numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime as _dt
import os
import sys
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from .models import MethodSpec, Prepared


# ---------------------------------------------------------------------------
def _set_subprocess_pythonpath() -> None:
    """Ensure spawned ProcessPoolExecutor workers can import hmm_studio + hmm_spikes."""
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    parts = [str(repo_root), str(repo_root / "sticky-poisson-hmm-python")]
    existing = os.environ.get("PYTHONPATH", "")
    if existing:
        parts.append(existing)
    os.environ["PYTHONPATH"] = os.pathsep.join(parts)


def _suggested_workers(restarts: int) -> int:
    cpu = os.cpu_count() or 1
    return max(1, min(restarts, max(1, cpu // 2)))


def _safe_worker_count(max_workers: int, total_tasks: int) -> int:
    """Clamp a user-supplied worker count to something the OS will tolerate."""
    cpu = os.cpu_count() or 1
    hard_ceiling = max(1, cpu)              # never spawn more than CPU count
    return max(1, min(int(max_workers), int(total_tasks), hard_ceiling))


# ----------------------------------------------------------------------------
# Run record
# ----------------------------------------------------------------------------
@dataclass
class Run:
    run_id: str
    name: str
    method_key: str
    method_label: str
    family: str
    dataset_name: str
    dataset_kind: str
    n_states: int
    bin_size: float
    n_units: int
    params: dict
    seed: int
    result: object
    prepared_trials: list                 # model-ready arrays for decoding
    log_likelihood: float
    bic: float
    n_params: int
    total_bins: int
    converged: bool
    threshold_satisfied: bool
    n_iter: int
    elapsed_s: float
    timestamp: str
    attempts: list[dict] = field(default_factory=list)

    @property
    def restored_on_exit(self) -> bool:
        return bool(getattr(self.result, "restored_on_exit", False))


@dataclass
class BICScanResult:
    by_k: dict[int, dict]                 # K -> {"best_run", "attempts", "bic", "ll"}
    best_k: int | None = None


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _total_bins(prep: Prepared) -> int:
    total = 0
    for t in prep.trials:
        if np.ndim(t) == 1:
            total += int(np.asarray(t).shape[0])
        else:
            total += int(np.asarray(t).shape[1])
    return total


def _is_valid(result, spec: MethodSpec) -> tuple[bool, bool]:
    converged = bool(getattr(result, "converged", False))
    threshold_satisfied = bool(getattr(result, "threshold_satisfied", True))
    return converged, threshold_satisfied


def _score_record(record: dict) -> tuple[int, int, float, float]:
    """Lower is better. Prefer threshold_satisfied, then converged, then BIC."""
    return (
        0 if record["threshold_satisfied"] else 1,
        0 if record["converged"] else 1,
        record["bic"],
        -record["log_likelihood"],
    )


def _make_run(spec: MethodSpec, prep: Prepared, n_states: int, params: dict,
              seed: int, result, dataset_name: str, dataset_kind: str,
              elapsed_s: float, attempts: list[dict]) -> Run:
    total_bins = _total_bins(prep)
    n_params = spec.n_params(n_states, prep.n_units)
    ll = float(result.log_likelihood)
    bic = -2.0 * ll + n_params * float(np.log(max(total_bins, 1)))
    converged, threshold_satisfied = _is_valid(result, spec)
    return Run(
        run_id=str(uuid.uuid4())[:8],
        name=f"{spec.label} · K={n_states} · LL={ll:.0f}",
        method_key=spec.key,
        method_label=spec.label,
        family=spec.family,
        dataset_name=dataset_name,
        dataset_kind=dataset_kind,
        n_states=n_states,
        bin_size=prep.bin_size,
        n_units=prep.n_units,
        params=dict(params),
        seed=seed,
        result=result,
        prepared_trials=list(prep.trials),
        log_likelihood=ll,
        bic=bic,
        n_params=n_params,
        total_bins=total_bins,
        converged=converged,
        threshold_satisfied=threshold_satisfied,
        n_iter=int(getattr(result, "n_iter", 0)),
        elapsed_s=elapsed_s,
        timestamp=_dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        attempts=attempts,
    )


# ----------------------------------------------------------------------------
# Fit worker
# ----------------------------------------------------------------------------
class FitWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(object)             # Run
    failed = Signal(str)

    def __init__(self, spec: MethodSpec, prep: Prepared, n_states: int,
                 params: dict, restarts: int, base_seed: int,
                 dataset_name: str, dataset_kind: str):
        super().__init__()
        self.spec = spec
        self.prep = prep
        self.n_states = int(n_states)
        self.params = dict(params)
        self.restarts = max(int(restarts), 1)
        self.base_seed = int(base_seed)
        self.dataset_name = dataset_name
        self.dataset_kind = dataset_kind
        self._cancel = False

    @Slot()
    def cancel(self):
        self._cancel = True

    @Slot()
    def run(self):
        try:
            attempts: list[dict] = []
            best_record = None
            best_run = None
            total_bins = _total_bins(self.prep)
            n_params = self.spec.n_params(self.n_states, self.prep.n_units)

            for r in range(self.restarts):
                if self._cancel:
                    self.failed.emit("Cancelled by user.")
                    return
                seed = self.base_seed + 1000 * self.n_states + r
                self.progress.emit(r, self.restarts,
                                   f"Restart {r+1}/{self.restarts} · seed={seed}")
                t0 = time.perf_counter()
                try:
                    result = self.spec.fit(self.prep, self.n_states, self.params, seed)
                except Exception as exc:
                    attempts.append({
                        "restart": r, "seed": seed, "status": "error",
                        "error": str(exc), "log_likelihood": float("-inf"),
                        "bic": float("inf"), "converged": False,
                        "threshold_satisfied": False, "n_iter": 0,
                    })
                    continue
                elapsed = time.perf_counter() - t0
                ll = float(result.log_likelihood)
                bic = -2.0 * ll + n_params * float(np.log(max(total_bins, 1)))
                converged, threshold_satisfied = _is_valid(result, self.spec)
                rec = {
                    "restart": r, "seed": seed, "status": "ok",
                    "log_likelihood": ll, "bic": bic,
                    "converged": converged,
                    "threshold_satisfied": threshold_satisfied,
                    "n_iter": int(getattr(result, "n_iter", 0)),
                    "elapsed_s": elapsed,
                }
                attempts.append(rec)
                if best_record is None or _score_record(rec) < _score_record(best_record):
                    best_record = rec
                    best_run = _make_run(
                        self.spec, self.prep, self.n_states, self.params, seed,
                        result, self.dataset_name, self.dataset_kind, elapsed,
                        attempts=[],
                    )

            if best_run is None:
                self.failed.emit("All restarts failed. Check parameters and data.")
                return
            best_run.attempts = attempts
            self.progress.emit(self.restarts, self.restarts,
                               f"Done · best LL={best_run.log_likelihood:.1f} · BIC={best_run.bic:.1f}")
            self.finished.emit(best_run)
        except Exception:
            self.failed.emit(traceback.format_exc())


# ----------------------------------------------------------------------------
# BIC scan worker
# ----------------------------------------------------------------------------
class BICScanWorker(QObject):
    progress = Signal(int, int, str)
    k_done = Signal(int, object)          # K, Run
    finished = Signal(object)             # BICScanResult
    failed = Signal(str)

    def __init__(self, spec: MethodSpec, prep: Prepared, k_values: list[int],
                 params: dict, restarts: int, base_seed: int,
                 dataset_name: str, dataset_kind: str):
        super().__init__()
        self.spec = spec
        self.prep = prep
        self.k_values = list(k_values)
        self.params = dict(params)
        self.restarts = max(int(restarts), 1)
        self.base_seed = int(base_seed)
        self.dataset_name = dataset_name
        self.dataset_kind = dataset_kind
        self._cancel = False

    @Slot()
    def cancel(self):
        self._cancel = True

    @Slot()
    def run(self):
        try:
            total = len(self.k_values) * self.restarts
            step = 0
            by_k: dict[int, dict] = {}
            total_bins = _total_bins(self.prep)
            for K in self.k_values:
                attempts: list[dict] = []
                best_record = None
                best_run = None
                n_params = self.spec.n_params(K, self.prep.n_units)
                for r in range(self.restarts):
                    if self._cancel:
                        self.failed.emit("Cancelled by user.")
                        return
                    seed = self.base_seed + 1000 * K + r
                    step += 1
                    self.progress.emit(step, total,
                                       f"K={K} · restart {r+1}/{self.restarts}")
                    t0 = time.perf_counter()
                    try:
                        result = self.spec.fit(self.prep, K, self.params, seed)
                    except Exception as exc:
                        attempts.append({
                            "restart": r, "seed": seed, "status": "error",
                            "error": str(exc), "log_likelihood": float("-inf"),
                            "bic": float("inf"), "converged": False,
                            "threshold_satisfied": False, "n_iter": 0,
                        })
                        continue
                    elapsed = time.perf_counter() - t0
                    ll = float(result.log_likelihood)
                    bic = -2.0 * ll + n_params * float(np.log(max(total_bins, 1)))
                    converged, threshold_satisfied = _is_valid(result, self.spec)
                    rec = {
                        "restart": r, "seed": seed, "status": "ok",
                        "log_likelihood": ll, "bic": bic,
                        "converged": converged,
                        "threshold_satisfied": threshold_satisfied,
                        "n_iter": int(getattr(result, "n_iter", 0)),
                        "elapsed_s": elapsed,
                    }
                    attempts.append(rec)
                    if best_record is None or _score_record(rec) < _score_record(best_record):
                        best_record = rec
                        best_run = _make_run(
                            self.spec, self.prep, K, self.params, seed, result,
                            self.dataset_name, self.dataset_kind, elapsed,
                            attempts=[],
                        )
                if best_run is not None:
                    best_run.attempts = attempts
                    by_k[K] = {
                        "best_run": best_run,
                        "attempts": attempts,
                        "bic": best_run.bic,
                        "ll": best_run.log_likelihood,
                    }
                    self.k_done.emit(K, best_run)
                else:
                    by_k[K] = {"best_run": None, "attempts": attempts,
                               "bic": float("inf"), "ll": float("-inf")}

            valid = {K: d for K, d in by_k.items() if d["best_run"] is not None}
            best_k = min(valid, key=lambda K: valid[K]["bic"]) if valid else None
            self.finished.emit(BICScanResult(by_k=by_k, best_k=best_k))
        except Exception:
            self.failed.emit(traceback.format_exc())


# ---------------------------------------------------------------------------
# Parallel workers (process pool)
# ---------------------------------------------------------------------------
def _make_run_from_envelope(env: dict, spec: MethodSpec, prep: Prepared,
                            params: dict, dataset_name: str,
                            dataset_kind: str, attempts: list) -> Run:
    return _make_run(spec, prep, env["K"], params, env["seed"],
                     env["result"], dataset_name, dataset_kind,
                     float(env.get("elapsed_s", 0.0)), attempts=attempts)


def _bic_from_envelope(env: dict, spec: MethodSpec, prep: Prepared) -> float:
    total_bins = _total_bins(prep)
    n_params = spec.n_params(env["K"], prep.n_units)
    return -2.0 * float(env["log_likelihood"]) + n_params * float(np.log(max(total_bins, 1)))


class ParallelFitWorker(QObject):
    """Run multi-restart fits across a process pool."""

    progress = Signal(int, int, str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, spec: MethodSpec, prep: Prepared, n_states: int,
                 params: dict, restarts: int, base_seed: int,
                 dataset_name: str, dataset_kind: str,
                 max_workers: int | None = None):
        super().__init__()
        self.spec = spec
        self.prep = prep
        self.n_states = int(n_states)
        self.params = dict(params)
        self.restarts = max(int(restarts), 1)
        self.base_seed = int(base_seed)
        self.dataset_name = dataset_name
        self.dataset_kind = dataset_kind
        self.max_workers = max_workers or _suggested_workers(self.restarts)
        self._cancel = False
        self._executor: ProcessPoolExecutor | None = None

    @Slot()
    def cancel(self):
        self._cancel = True
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

    @Slot()
    def run(self):
        try:
            _set_subprocess_pythonpath()
            from ._fit_task import _init_worker, run_fit_task

            initargs = (self.spec.key, self.prep.trials, self.prep.n_units,
                        self.prep.bin_size, self.prep.n_symbols, self.dataset_kind)
            attempts: list[dict] = []
            best_run: Run | None = None
            best_record = None

            n_workers = _safe_worker_count(self.max_workers, self.restarts)
            with ProcessPoolExecutor(max_workers=n_workers,
                                     initializer=_init_worker,
                                     initargs=initargs) as ex:
                self._executor = ex
                futures = {}
                for r in range(self.restarts):
                    seed = self.base_seed + 1000 * self.n_states + r
                    fut = ex.submit(run_fit_task, self.n_states, self.params, seed)
                    futures[fut] = (r, seed)
                self.progress.emit(0, self.restarts,
                                   f"Started {self.restarts} restart(s) across {n_workers} worker(s)…")

                done = 0
                for fut in as_completed(futures):
                    if self._cancel:
                        self.failed.emit("Cancelled by user.")
                        return
                    r, seed = futures[fut]
                    try:
                        env = fut.result()
                    except Exception as exc:
                        attempts.append({"restart": r, "seed": seed, "status": "error",
                                         "error": str(exc),
                                         "log_likelihood": float("-inf"),
                                         "bic": float("inf"),
                                         "converged": False,
                                         "threshold_satisfied": False, "n_iter": 0})
                        done += 1
                        self.progress.emit(done, self.restarts,
                                           f"Restart {r+1} crashed: {exc}")
                        continue

                    if not env.get("ok", False):
                        attempts.append({"restart": r, "seed": seed, "status": "error",
                                         "error": env.get("error", "unknown"),
                                         "log_likelihood": float("-inf"),
                                         "bic": float("inf"),
                                         "converged": False,
                                         "threshold_satisfied": False, "n_iter": 0})
                        done += 1
                        self.progress.emit(done, self.restarts,
                                           f"Restart {r+1} failed.")
                        continue

                    bic = _bic_from_envelope(env, self.spec, self.prep)
                    rec = {
                        "restart": r, "seed": seed, "status": "ok",
                        "log_likelihood": env["log_likelihood"], "bic": bic,
                        "converged": env["converged"],
                        "threshold_satisfied": env["threshold_satisfied"],
                        "n_iter": env["n_iter"],
                        "elapsed_s": env["elapsed_s"],
                    }
                    attempts.append(rec)
                    if best_record is None or _score_record(rec) < _score_record(best_record):
                        best_record = rec
                        best_run = _make_run_from_envelope(
                            env, self.spec, self.prep, self.params,
                            self.dataset_name, self.dataset_kind, attempts=[])
                    done += 1
                    self.progress.emit(
                        done, self.restarts,
                        f"Restart {r+1}/{self.restarts}  ·  "
                        f"LL={env['log_likelihood']:.0f}  "
                        f"BIC={bic:.0f}  "
                        f"{'conv' if env['converged'] else 'no-conv'}  "
                        f"({env['elapsed_s']:.1f}s)"
                    )

            self._executor = None
            if best_run is None:
                self.failed.emit("All restarts failed. Check parameters and data.")
                return
            best_run.attempts = attempts
            self.finished.emit(best_run)
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            self._executor = None


class ParallelBICScanWorker(QObject):
    """Run a BIC scan across (K × restarts) in a process pool."""

    progress = Signal(int, int, str)
    k_done = Signal(int, object)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, spec: MethodSpec, prep: Prepared, k_values: list,
                 params: dict, restarts: int, base_seed: int,
                 dataset_name: str, dataset_kind: str,
                 max_workers: int | None = None):
        super().__init__()
        self.spec = spec
        self.prep = prep
        self.k_values = list(k_values)
        self.params = dict(params)
        self.restarts = max(int(restarts), 1)
        self.base_seed = int(base_seed)
        self.dataset_name = dataset_name
        self.dataset_kind = dataset_kind
        total_tasks = max(1, len(self.k_values) * self.restarts)
        self.max_workers = max_workers or _suggested_workers(total_tasks)
        self._cancel = False
        self._executor: ProcessPoolExecutor | None = None

    @Slot()
    def cancel(self):
        self._cancel = True
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

    @Slot()
    def run(self):
        try:
            _set_subprocess_pythonpath()
            from ._fit_task import _init_worker, run_fit_task

            initargs = (self.spec.key, self.prep.trials, self.prep.n_units,
                        self.prep.bin_size, self.prep.n_symbols, self.dataset_kind)
            total = len(self.k_values) * self.restarts
            n_workers = _safe_worker_count(self.max_workers, total)

            # Per-K aggregation
            attempts_by_k: dict[int, list[dict]] = {K: [] for K in self.k_values}
            best_record_by_k: dict[int, dict | None] = {K: None for K in self.k_values}
            best_run_by_k: dict[int, Run | None] = {K: None for K in self.k_values}

            with ProcessPoolExecutor(max_workers=n_workers,
                                     initializer=_init_worker,
                                     initargs=initargs) as ex:
                self._executor = ex
                futures = {}
                for K in self.k_values:
                    for r in range(self.restarts):
                        seed = self.base_seed + 1000 * K + r
                        fut = ex.submit(run_fit_task, K, self.params, seed)
                        futures[fut] = (K, r, seed)
                self.progress.emit(0, total,
                                   f"Started {total} fits across {n_workers} worker(s)…")

                done = 0
                k_progress = {K: 0 for K in self.k_values}
                k_total = self.restarts

                for fut in as_completed(futures):
                    if self._cancel:
                        self.failed.emit("Cancelled by user.")
                        return
                    K, r, seed = futures[fut]
                    try:
                        env = fut.result()
                    except Exception as exc:
                        attempts_by_k[K].append({"restart": r, "seed": seed,
                                                 "status": "error",
                                                 "error": str(exc),
                                                 "log_likelihood": float("-inf"),
                                                 "bic": float("inf"),
                                                 "converged": False,
                                                 "threshold_satisfied": False,
                                                 "n_iter": 0})
                        done += 1; k_progress[K] += 1
                        self.progress.emit(done, total, f"K={K} restart {r+1} crashed")
                        continue

                    if not env.get("ok", False):
                        attempts_by_k[K].append({"restart": r, "seed": seed,
                                                 "status": "error",
                                                 "error": env.get("error", "unknown"),
                                                 "log_likelihood": float("-inf"),
                                                 "bic": float("inf"),
                                                 "converged": False,
                                                 "threshold_satisfied": False,
                                                 "n_iter": 0})
                        done += 1; k_progress[K] += 1
                        self.progress.emit(done, total, f"K={K} restart {r+1} failed")
                        continue

                    bic = _bic_from_envelope(env, self.spec, self.prep)
                    rec = {
                        "restart": r, "seed": seed, "status": "ok",
                        "log_likelihood": env["log_likelihood"], "bic": bic,
                        "converged": env["converged"],
                        "threshold_satisfied": env["threshold_satisfied"],
                        "n_iter": env["n_iter"],
                        "elapsed_s": env["elapsed_s"],
                    }
                    attempts_by_k[K].append(rec)
                    if (best_record_by_k[K] is None
                            or _score_record(rec) < _score_record(best_record_by_k[K])):
                        best_record_by_k[K] = rec
                        best_run_by_k[K] = _make_run_from_envelope(
                            env, self.spec, self.prep, self.params,
                            self.dataset_name, self.dataset_kind, attempts=[])

                    done += 1; k_progress[K] += 1
                    self.progress.emit(
                        done, total,
                        f"K={K}  ({k_progress[K]}/{k_total})  ·  "
                        f"LL={env['log_likelihood']:.0f}  BIC={bic:.0f}  "
                        f"({env['elapsed_s']:.1f}s)"
                    )

                    if k_progress[K] == k_total and best_run_by_k[K] is not None:
                        best_run_by_k[K].attempts = list(attempts_by_k[K])
                        self.k_done.emit(K, best_run_by_k[K])

            self._executor = None
            by_k: dict[int, dict] = {}
            for K in self.k_values:
                br = best_run_by_k[K]
                if br is not None:
                    br.attempts = list(attempts_by_k[K])
                by_k[K] = {
                    "best_run": br,
                    "attempts": list(attempts_by_k[K]),
                    "bic": br.bic if br else float("inf"),
                    "ll": br.log_likelihood if br else float("-inf"),
                }
            valid = {K: d for K, d in by_k.items() if d["best_run"] is not None}
            best_k = min(valid, key=lambda K: valid[K]["bic"]) if valid else None
            self.finished.emit(BICScanResult(by_k=by_k, best_k=best_k))
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            self._executor = None
