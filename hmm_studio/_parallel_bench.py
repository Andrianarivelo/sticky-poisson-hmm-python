"""Benchmark serial vs parallel fits on the user's photometry CSV."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "sticky-poisson-hmm-python"))

import numpy as np
from PySide6.QtCore import QEventLoop, QThread, QTimer
from PySide6.QtWidgets import QApplication

from hmm_studio import dataio
from hmm_studio.csv_import import CsvImportDialog
from hmm_studio.models import REGISTRY
from hmm_studio.workers import (
    BICScanWorker, FitWorker, ParallelBICScanWorker, ParallelFitWorker,
)


def build_dataset():
    app = QApplication.instance() or QApplication([])
    dlg = CsvImportDialog(Path(r"C:\Analysis\trial_0014_AIN01.csv"))
    # Keep only the dFF column for a fair, small benchmark
    from PySide6.QtCore import Qt
    for i in range(dlg.col_list.count()):
        it = dlg.col_list.item(i)
        it.setCheckState(Qt.Checked if it.text() == "dFF" else Qt.Unchecked)
    dlg.mode.setCurrentIndex(0)
    dlg._on_accept()
    ds = dlg.dataset
    assert ds is not None
    print(f"Dataset: {ds.n_channels} ch x {ds.n_bins} bins  dt={ds.bin_size:.4g} s")
    return ds


def run_worker(worker, label: str) -> float:
    app = QApplication.instance() or QApplication([])
    thread = QThread(); worker.moveToThread(thread)
    loop = QEventLoop(); box = {}

    def done(*a): box["a"] = a; loop.quit()
    def fail(msg): box["err"] = msg; loop.quit()
    worker.finished.connect(done); worker.failed.connect(fail)
    thread.started.connect(worker.run)
    t0 = time.perf_counter()
    thread.start()
    QTimer.singleShot(20 * 60 * 1000, loop.quit)  # 20-min cap
    loop.exec()
    elapsed = time.perf_counter() - t0
    thread.quit(); thread.wait(5000)
    if "err" in box:
        print(f"  [{label}] FAILED: {box['err'][:200]}")
        return elapsed
    return elapsed


def main():
    app = QApplication.instance() or QApplication([])
    ds = build_dataset()
    spec = REGISTRY["sticky_gaussian"]
    rng = np.random.default_rng(0)
    prep = spec.prepare(ds.trials, ds.kind, ds.bin_size, rng)
    params = {"threshold": 0.85, "max_iter": 60}
    restarts = 4

    # --- serial ---
    w = FitWorker(spec, prep, n_states=3, params=params, restarts=restarts,
                  base_seed=42, dataset_name=ds.name, dataset_kind=ds.kind)
    t_serial = run_worker(w, "serial")
    print(f"Serial   ({restarts} restarts):  {t_serial:5.1f} s")

    # --- parallel ---
    n_workers = min(restarts, max(1, (os.cpu_count() or 1) - 1))
    w = ParallelFitWorker(spec, prep, n_states=3, params=params, restarts=restarts,
                          base_seed=42, dataset_name=ds.name, dataset_kind=ds.kind,
                          max_workers=n_workers)
    t_par = run_worker(w, f"parallel x{n_workers}")
    print(f"Parallel ({restarts} restarts, {n_workers} workers):  {t_par:5.1f} s")
    if t_par > 0:
        print(f"Speedup: {t_serial / t_par:.2f}x")


if __name__ == "__main__":
    main()
