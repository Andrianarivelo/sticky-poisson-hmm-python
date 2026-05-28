"""Drive the GUI under the offscreen platform and save PNG screenshots."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
# Offscreen Qt on this machine can't auto-discover TrueType fonts, so point
# it at the Windows font folder before any Qt module is imported.
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "sticky-poisson-hmm-python"))

import numpy as np
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QFont, QFontDatabase, QGuiApplication
from PySide6.QtWidgets import QApplication

from hmm_studio import dataio, theme
from hmm_studio.app import StudioWindow
from hmm_studio.models import REGISTRY
from hmm_studio.workers import _make_run, BICScanResult


SHOT_DIR = HERE / "_screenshots"
SHOT_DIR.mkdir(exist_ok=True)


def snap(window, name: str):
    window.repaint()
    QApplication.processEvents()
    QApplication.processEvents()
    pix = window.grab()
    out = SHOT_DIR / f"{name}.png"
    pix.save(str(out), "PNG")
    print(f"  saved {out}  ({pix.width()}x{pix.height()})")
    return out


def run_fit(window, method_key: str, K: int, params: dict, seed: int = 21):
    spec = REGISTRY[method_key]
    prep = spec.prepare(window.dataset.trials, window.dataset.kind,
                        window.dataset.bin_size, np.random.default_rng(0))
    result = spec.fit(prep, K, params, random_state=seed)
    run = _make_run(spec, prep, K, params, seed, result,
                    window.dataset.name, window.dataset.kind, 0.0, attempts=[])
    window._add_run(run)
    window._activate_run(run.run_id)
    return run


def fake_bic_scan(window, method_key: str, k_values: list, params: dict):
    spec = REGISTRY[method_key]
    prep = spec.prepare(window.dataset.trials, window.dataset.kind,
                        window.dataset.bin_size, np.random.default_rng(0))
    by_k = {}
    for K in k_values:
        attempts = []
        best = None
        for r in range(2):
            seed = 100 + 31 * K + r
            try:
                result = spec.fit(prep, K, params, random_state=seed)
            except Exception:
                continue
            ll = float(result.log_likelihood)
            from hmm_studio.workers import _total_bins
            n_params = spec.n_params(K, prep.n_units)
            bic = -2.0 * ll + n_params * float(np.log(max(_total_bins(prep), 1)))
            rec = {"restart": r, "seed": seed, "status": "ok",
                   "log_likelihood": ll, "bic": bic,
                   "converged": bool(getattr(result, "converged", True)),
                   "threshold_satisfied": bool(getattr(result, "threshold_satisfied", True)),
                   "n_iter": int(getattr(result, "n_iter", 0)),
                   "elapsed_s": 0.0}
            attempts.append(rec)
            if best is None or bic < best.bic:
                best = _make_run(spec, prep, K, params, seed, result,
                                 window.dataset.name, window.dataset.kind, 0.0, attempts=[])
        by_k[K] = {"best_run": best, "attempts": attempts,
                   "bic": best.bic if best else float("inf"),
                   "ll": best.log_likelihood if best else float("-inf")}
        if best:
            best.name = f"{best.method_label} · K={K}  (BIC scan)"
            window._add_run(best)
    valid = {K: d for K, d in by_k.items() if d["best_run"] is not None}
    best_k = min(valid, key=lambda K: valid[K]["bic"]) if valid else None
    res = BICScanResult(by_k=by_k, best_k=best_k)
    window.bic_result = res
    window.view_bic.update_bic(res)
    return res


def main():
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication.instance() or QApplication([])
    # Force-load a couple of TTFs so the offscreen platform has fonts to use.
    for fn in ("segoeui.ttf", "arial.ttf", "tahoma.ttf"):
        path = Path(r"C:\Windows\Fonts") / fn
        if path.exists():
            QFontDatabase.addApplicationFont(str(path))
    app.setFont(QFont("Segoe UI", 9))
    app.setStyleSheet(theme.stylesheet())
    theme.configure_pyqtgraph()

    win = StudioWindow()
    win.resize(1600, 1000)
    win.show()
    app.processEvents()

    # 1. Initial screen
    snap(win, "01_empty_state")

    # 2. After loading demo counts
    win._load_demo("counts")
    app.processEvents()
    snap(win, "02_data_counts")

    # 3. Fit one sticky Poisson and view all tabs
    run = run_fit(win, "sticky_poisson", K=3,
                  params={"threshold": 0.8, "max_iter": 250})
    for i, name in enumerate(["data", "decoding", "states", "training", "bic", "compare"]):
        win.tabs.setCurrentIndex(i)
        app.processEvents()
        snap(win, f"03_tab_{i}_{name}")

    # 4. Run a (real) BIC scan inline so we get content on the BIC tab
    fake_bic_scan(win, "sticky_poisson",
                  k_values=[2, 3, 4, 5],
                  params={"threshold": 0.8, "max_iter": 150})
    win.tabs.setCurrentWidget(win.view_bic)
    app.processEvents()
    snap(win, "04_bic_filled")

    # 5. Compare tab with multiple runs
    win.view_compare.update_runs(win.runs)
    win.tabs.setCurrentWidget(win.view_compare)
    app.processEvents()
    snap(win, "05_compare_runs")

    # 6. Continuous demo to exercise traces + Gaussian view
    win._load_demo("continuous")
    app.processEvents()
    snap(win, "06_data_continuous")
    run_fit(win, "sticky_gaussian", K=3,
            params={"threshold": 0.8, "max_iter": 250})
    win.tabs.setCurrentWidget(win.view_decoding)
    app.processEvents()
    snap(win, "07_gaussian_decoding")
    win.tabs.setCurrentWidget(win.view_states)
    app.processEvents()
    snap(win, "08_gaussian_states")

    print("\nAll screenshots written to", SHOT_DIR)


if __name__ == "__main__":
    main()
