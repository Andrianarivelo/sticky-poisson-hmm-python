"""Top-level task functions for ``ProcessPoolExecutor``.

This module exists because Windows uses spawn semantics: every worker process
re-imports Python from scratch and looks up the initializer / task functions
by their dotted name. Putting them here keeps the imports simple and avoids
circular references.

The pool layout is:

* ``_init_worker`` runs once per child process and stashes the prepared trials
  and method spec in a module global. That avoids re-pickling the data on
  every task.
* ``run_fit_task`` then receives only ``(K, params, seed)`` and returns a
  small dict with the fitted :class:`hmm_spikes` result object and metadata.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

# Re-attach the project paths inside the worker. The parent process sets
# PYTHONPATH so they're already on sys.path here, but this is belt-and-braces.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
for p in (str(_REPO_ROOT), str(_REPO_ROOT / "sticky-poisson-hmm-python")):
    if p not in sys.path:
        sys.path.insert(0, p)

# Keep BLAS to a single thread per worker so N processes don't oversubscribe.
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(var, "1")


def _lower_process_priority() -> None:
    """Drop this worker to background priority so the GUI stays responsive."""
    if sys.platform.startswith("win"):
        try:
            import ctypes
            BELOW_NORMAL = 0x00004000
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(), BELOW_NORMAL)
        except Exception:
            pass
    else:
        try:
            os.nice(10)  # POSIX: raise niceness
        except Exception:
            pass


_STATE: dict = {}


def _init_worker(spec_key: str, trials, n_units: int, bin_size: float,
                 n_symbols, data_kind: str) -> None:
    """Run once per worker process to cache the prepared trials and spec."""
    _lower_process_priority()
    from hmm_studio.models import REGISTRY, Prepared

    _STATE["spec"] = REGISTRY[spec_key]
    _STATE["prep"] = Prepared(trials=list(trials), n_units=int(n_units),
                              bin_size=float(bin_size),
                              n_symbols=None if n_symbols is None else int(n_symbols))
    _STATE["data_kind"] = data_kind


def run_fit_task(n_states: int, params: dict, seed: int) -> dict:
    """Fit one HMM and return a small result envelope."""

    spec = _STATE.get("spec"); prep = _STATE.get("prep")
    if spec is None or prep is None:
        return {"ok": False, "seed": seed, "K": n_states,
                "error": "worker not initialized"}
    t0 = time.perf_counter()
    try:
        result = spec.fit(prep, int(n_states), dict(params), random_state=int(seed))
    except Exception:
        return {"ok": False, "seed": seed, "K": int(n_states),
                "error": traceback.format_exc()}
    elapsed = time.perf_counter() - t0
    return {
        "ok": True,
        "seed": int(seed),
        "K": int(n_states),
        "elapsed_s": float(elapsed),
        "log_likelihood": float(result.log_likelihood),
        "n_iter": int(getattr(result, "n_iter", 0)),
        "converged": bool(getattr(result, "converged", True)),
        "threshold_satisfied": bool(getattr(result, "threshold_satisfied", True)),
        "result": result,
    }
