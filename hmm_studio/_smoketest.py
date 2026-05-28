"""Smoke test: import, fit a small model, decode, build the main window headless."""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
# Make this script's stdout safe for non-ASCII on Windows consoles.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "sticky-poisson-hmm-python"))

failures = []


def step(name, fn):
    print(f"\n=== {name} ===")
    t0 = time.perf_counter()
    try:
        out = fn()
        dt = time.perf_counter() - t0
        print(f"  ok  ({dt*1000:.0f} ms)")
        return out
    except Exception:
        failures.append(name)
        print(f"  FAIL")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
def imports():
    import hmm_spikes  # noqa
    from hmm_studio import dataio, models, theme, workers, views, panels, app  # noqa
    return True


def make_demo_and_fit():
    from hmm_studio import dataio
    from hmm_studio.models import REGISTRY
    from hmm_studio.workers import FitWorker

    ds = dataio.demo_spike_counts(n_neurons=8, n_states=3, n_bins=600, n_trials=2, seed=42)
    print(f"  trials={ds.n_trials} channels={ds.n_channels} bins={ds.n_bins} bin_size={ds.bin_size}")

    spec = REGISTRY["sticky_poisson"]
    import numpy as np
    rng = np.random.default_rng(0)
    prep = spec.prepare(ds.trials, ds.kind, ds.bin_size, rng)
    # Run inline (no QThread) by calling the fit directly.
    params = {"threshold": 0.8, "max_iter": 150}
    result = spec.fit(prep, 3, params, random_state=7)
    print(f"  LL = {result.log_likelihood:.2f}  iter={result.n_iter}  "
          f"converged={result.converged}  threshold_satisfied={result.threshold_satisfied}")

    # Decode trial 0
    post, vit = spec.decode(result, prep.trials[0])
    print(f"  posterior shape={post.shape}  viterbi unique states={len(set(vit.tolist()))}")
    return ds, result, post, vit


def make_continuous_and_fit():
    from hmm_studio import dataio
    from hmm_studio.models import REGISTRY
    import numpy as np

    ds = dataio.demo_continuous(n_channels=3, n_states=3, n_bins=800, n_trials=2)
    spec = REGISTRY["sticky_gaussian"]
    prep = spec.prepare(ds.trials, ds.kind, ds.bin_size, np.random.default_rng(0))
    result = spec.fit(prep, 3, {"threshold": 0.8, "max_iter": 120}, random_state=11)
    print(f"  gaussian LL = {result.log_likelihood:.2f}  iter={result.n_iter}")
    post, vit = spec.decode(result, prep.trials[0])
    print(f"  posterior shape={post.shape}  viterbi T={len(vit)}")
    return ds, result


def make_multinoulli_and_fit():
    from hmm_studio import dataio
    from hmm_studio.models import REGISTRY
    import numpy as np

    ds = dataio.demo_spike_counts(n_neurons=6, n_states=3, n_bins=400, n_trials=2)
    spec = REGISTRY["multinoulli"]
    prep = spec.prepare(ds.trials, ds.kind, ds.bin_size, np.random.default_rng(0))
    result = spec.fit(prep, 3, {"max_iter": 100}, random_state=3)
    print(f"  multinoulli LL = {result.log_likelihood:.2f}  iter={result.n_iter}")
    post, vit = spec.decode(result, prep.trials[0])
    print(f"  posterior shape={post.shape}")
    return ds, result


def headless_window():
    # Build the QApplication + main window with offscreen platform.
    from PySide6.QtWidgets import QApplication
    from hmm_studio import theme
    from hmm_studio.app import StudioWindow

    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(theme.stylesheet())
    theme.configure_pyqtgraph()
    win = StudioWindow()
    win.resize(1400, 900)

    # Load demo and trigger renders
    win._load_demo("counts")
    # Render Data tab
    win.tabs.setCurrentIndex(0)
    win.repaint()

    # Fit one model synchronously (bypassing worker thread for the test)
    from hmm_studio.models import REGISTRY
    from hmm_studio.workers import _make_run, _total_bins
    import numpy as np
    spec = REGISTRY["sticky_poisson"]
    rng = np.random.default_rng(0)
    prep = spec.prepare(win.dataset.trials, win.dataset.kind, win.dataset.bin_size, rng)
    result = spec.fit(prep, 3, {"threshold": 0.8, "max_iter": 100}, random_state=21)
    run = _make_run(spec, prep, 3, {"threshold": 0.8, "max_iter": 100}, 21, result,
                    win.dataset.name, win.dataset.kind, 0.0, attempts=[])
    win._add_run(run)
    win._activate_run(run.run_id)

    # Force-update each tab to catch render errors
    for i in range(win.tabs.count()):
        win.tabs.setCurrentIndex(i)
        win.repaint()
        app.processEvents()
    print(f"  tabs ok, {win.tabs.count()} rendered")
    win.close()
    return True


# ---------------------------------------------------------------------------
def threaded_fit_worker():
    """Run FitWorker via a real QThread under the offscreen platform."""
    import numpy as np
    from PySide6.QtCore import QEventLoop, QThread, QTimer
    from PySide6.QtWidgets import QApplication

    from hmm_studio import dataio
    from hmm_studio.models import REGISTRY
    from hmm_studio.workers import FitWorker

    app = QApplication.instance() or QApplication([])
    ds = dataio.demo_spike_counts(n_neurons=6, n_states=3, n_bins=300, n_trials=2)
    spec = REGISTRY["sticky_poisson"]
    prep = spec.prepare(ds.trials, ds.kind, ds.bin_size, np.random.default_rng(0))
    worker = FitWorker(spec, prep, 3, {"threshold": 0.8, "max_iter": 100},
                       restarts=2, base_seed=42,
                       dataset_name=ds.name, dataset_kind=ds.kind)
    thread = QThread()
    worker.moveToThread(thread)
    loop = QEventLoop()
    result_box = {}

    def on_done(run):
        result_box["run"] = run
        loop.quit()

    def on_fail(msg):
        result_box["error"] = msg
        loop.quit()

    worker.finished.connect(on_done)
    worker.failed.connect(on_fail)
    thread.started.connect(worker.run)
    thread.start()
    QTimer.singleShot(30000, loop.quit)  # safety timeout
    loop.exec()
    thread.quit(); thread.wait(2000)

    if "error" in result_box:
        raise RuntimeError(result_box["error"])
    run = result_box["run"]
    print(f"  worker LL={run.log_likelihood:.2f}  BIC={run.bic:.2f}  iter={run.n_iter}")
    return True


def threaded_bic_scan():
    import numpy as np
    from PySide6.QtCore import QEventLoop, QThread, QTimer
    from PySide6.QtWidgets import QApplication

    from hmm_studio import dataio
    from hmm_studio.models import REGISTRY
    from hmm_studio.workers import BICScanWorker

    app = QApplication.instance() or QApplication([])
    ds = dataio.demo_spike_counts(n_neurons=6, n_states=3, n_bins=250, n_trials=2)
    spec = REGISTRY["sticky_poisson"]
    prep = spec.prepare(ds.trials, ds.kind, ds.bin_size, np.random.default_rng(0))
    worker = BICScanWorker(spec, prep, k_values=[2, 3, 4],
                           params={"threshold": 0.8, "max_iter": 60},
                           restarts=2, base_seed=42,
                           dataset_name=ds.name, dataset_kind=ds.kind)
    thread = QThread(); worker.moveToThread(thread)
    loop = QEventLoop()
    box = {}

    def on_done(res):
        box["res"] = res; loop.quit()

    def on_fail(msg):
        box["error"] = msg; loop.quit()

    worker.finished.connect(on_done); worker.failed.connect(on_fail)
    thread.started.connect(worker.run); thread.start()
    QTimer.singleShot(60000, loop.quit)
    loop.exec()
    thread.quit(); thread.wait(2000)
    if "error" in box:
        raise RuntimeError(box["error"])
    res = box["res"]
    print(f"  best K = {res.best_k}")
    for k, info in res.by_k.items():
        run = info["best_run"]
        if run:
            print(f"    K={k}: BIC={run.bic:.1f}  LL={run.log_likelihood:.1f}  conv={run.converged}")
    return True


step("imports", imports)
step("poisson demo + fit + decode", make_demo_and_fit)
step("gaussian demo + fit + decode", make_continuous_and_fit)
step("multinoulli demo + fit + decode", make_multinoulli_and_fit)
step("threaded FitWorker", threaded_fit_worker)
step("threaded BICScanWorker", threaded_bic_scan)
step("headless main window", headless_window)

print("\n========================================")
if failures:
    print("FAILURES:", failures)
    sys.exit(1)
print("All smoke tests passed.")
