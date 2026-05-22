from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable

import numpy as np


@dataclass
class RestartRecord:
    n_states: int
    restart: int
    seed: int
    status: str
    converged: bool
    threshold_satisfied: bool
    restored_on_exit: bool
    n_iter: int
    log_likelihood: float
    bic: float
    min_diag: float
    model_path: str
    returncode: int
    message: str


@dataclass
class BICScanResult:
    records: list[RestartRecord]
    best_strict: RestartRecord | None
    best_diagnostic: RestartRecord | None

    def to_dicts(self) -> list[dict]:
        return [asdict(record) for record in self.records]


def poisson_hmm_bic(log_likelihood: float, n_states: int, n_neurons: int, total_bins: int) -> float:
    """BIC used in the original Poisson HMM toolbox."""

    n_params = n_states * (n_states - 1) + n_states * n_neurons
    return float(-2.0 * log_likelihood + n_params * np.log(total_bins))


def save_counts_archive(counts: list[np.ndarray], path: str | Path, *, bin_size: float) -> Path:
    """Save variable-length count trials without pickle."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "n_trials": np.array(len(counts), dtype=np.int64),
        "bin_size": np.array(float(bin_size)),
    }
    for idx, trial in enumerate(counts):
        arrays[f"trial_{idx}"] = np.asarray(trial, dtype=np.float64)
    np.savez_compressed(path, **arrays)
    return path


def load_counts_archive(path: str | Path) -> tuple[list[np.ndarray], float]:
    archive = np.load(path, allow_pickle=False)
    n_trials = int(archive["n_trials"])
    counts = [np.asarray(archive[f"trial_{idx}"], dtype=float) for idx in range(n_trials)]
    return counts, float(archive["bin_size"])


def _thread_limited_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    return env


def _best_record(records: list[RestartRecord], *, require_converged: bool) -> RestartRecord | None:
    valid = []
    for record in records:
        if record.status not in {"strict", "diagnostic"}:
            continue
        if not record.threshold_satisfied:
            continue
        if require_converged and not record.converged:
            continue
        valid.append(record)
    if not valid:
        return None
    return min(valid, key=lambda record: (record.bic, -record.log_likelihood))


def run_sticky_poisson_bic_scan_isolated(
    counts: list[np.ndarray],
    candidate_states: Iterable[int],
    *,
    bin_size: float,
    n_restarts: int = 10,
    threshold: float = 0.8,
    max_iter: int = 1000,
    timeout_s: float | None = None,
    base_seed: int = 0,
    output_dir: str | Path | None = None,
    python_executable: str | None = None,
) -> BICScanResult:
    """Run a crash-isolated serial BIC scan for sticky Poisson HMMs.

    Each K and restart is fit in a fresh Python subprocess. If a low-level
    numerical backend aborts, the parent process records the failed restart and
    continues instead of losing the whole scan.
    """

    counts = [np.asarray(trial, dtype=float) for trial in counts]
    if not counts:
        raise ValueError("counts must contain at least one trial")
    n_neurons = counts[0].shape[0]
    if any(trial.shape[0] != n_neurons for trial in counts):
        raise ValueError("all trials must have the same neuron count")

    output_path = Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    records: list[RestartRecord] = []
    py = python_executable or sys.executable

    with tempfile.TemporaryDirectory(prefix="sticky_hmm_scan_") as tmp:
        tmp_path = Path(tmp)
        counts_path = save_counts_archive(counts, tmp_path / "counts.npz", bin_size=bin_size)
        for n_states in candidate_states:
            for restart in range(n_restarts):
                seed = int(base_seed + 100_003 * int(n_states) + restart)
                run_dir = tmp_path / f"k{int(n_states)}_r{restart}"
                run_dir.mkdir(parents=True, exist_ok=True)
                summary_path = run_dir / "summary.json"
                model_path = run_dir / "model.npz"
                cmd = [
                    py,
                    "-m",
                    "hmm_spikes._fit_worker",
                    "--counts",
                    str(counts_path),
                    "--summary",
                    str(summary_path),
                    "--model",
                    str(model_path),
                    "--states",
                    str(int(n_states)),
                    "--threshold",
                    str(float(threshold)),
                    "--max-iter",
                    str(int(max_iter)),
                    "--seed",
                    str(seed),
                ]
                try:
                    completed = subprocess.run(
                        cmd,
                        cwd=str(Path(__file__).resolve().parents[1]),
                        env=_thread_limited_env(),
                        text=True,
                        capture_output=True,
                        timeout=timeout_s,
                        check=False,
                    )
                except subprocess.TimeoutExpired as exc:
                    records.append(
                        RestartRecord(
                            n_states=int(n_states),
                            restart=restart,
                            seed=seed,
                            status="timeout",
                            converged=False,
                            threshold_satisfied=False,
                            restored_on_exit=False,
                            n_iter=0,
                            log_likelihood=float("-inf"),
                            bic=float("inf"),
                            min_diag=float("nan"),
                            model_path="",
                            returncode=-1,
                            message=f"timeout after {timeout_s}s: {(exc.stderr or '')[-1000:]}",
                        )
                    )
                    continue

                if completed.returncode != 0 or not summary_path.exists():
                    stderr_tail = (completed.stderr or "")[-2000:]
                    stdout_tail = (completed.stdout or "")[-1000:]
                    records.append(
                        RestartRecord(
                            n_states=int(n_states),
                            restart=restart,
                            seed=seed,
                            status="crashed",
                            converged=False,
                            threshold_satisfied=False,
                            restored_on_exit=False,
                            n_iter=0,
                            log_likelihood=float("-inf"),
                            bic=float("inf"),
                            min_diag=float("nan"),
                            model_path="",
                            returncode=completed.returncode,
                            message=(stderr_tail or stdout_tail or "subprocess failed without output"),
                        )
                    )
                    continue

                data = json.loads(summary_path.read_text(encoding="utf-8"))
                final_model_path = ""
                if model_path.exists() and output_path is not None:
                    final_model = output_path / f"k{int(n_states)}_restart{restart}_seed{seed}.npz"
                    shutil.copy2(model_path, final_model)
                    final_model_path = str(final_model)
                elif model_path.exists():
                    final_model_path = str(model_path)

                records.append(
                    RestartRecord(
                        n_states=int(data["n_states"]),
                        restart=restart,
                        seed=seed,
                        status=str(data["status"]),
                        converged=bool(data["converged"]),
                        threshold_satisfied=bool(data["threshold_satisfied"]),
                        restored_on_exit=bool(data["restored_on_exit"]),
                        n_iter=int(data["n_iter"]),
                        log_likelihood=float(data["log_likelihood"]),
                        bic=float(data["bic"]),
                        min_diag=float(data["min_diag"]),
                        model_path=final_model_path,
                        returncode=completed.returncode,
                        message=str(data.get("message", "")),
                    )
                )

    if output_path is not None:
        (output_path / "bic_scan_records.json").write_text(
            json.dumps([asdict(record) for record in records], indent=2),
            encoding="utf-8",
        )

    return BICScanResult(
        records=records,
        best_strict=_best_record(records, require_converged=True),
        best_diagnostic=_best_record(records, require_converged=False),
    )
