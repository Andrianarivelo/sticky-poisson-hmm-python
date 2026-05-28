"""Main window: docks, tabs, worker lifecycle, exports."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# Force PySide6 as the Qt binding for pyqtgraph BEFORE importing it.
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("QT_API", "pyside6")

import numpy as np
from PySide6.QtCore import Qt, QThread, QSize
from PySide6.QtGui import QAction, QIcon, QKeySequence, QGuiApplication
from PySide6.QtWidgets import (
    QApplication, QDockWidget, QFileDialog, QLabel, QMainWindow, QMessageBox,
    QStatusBar, QTabWidget, QToolBar, QWidget,
)

from . import dataio, state_store, theme
from .dataio import Dataset
from .models import REGISTRY, methods_for_kind
from .panels import ControlPanel, RunsPanel
from .views import (
    BICView, CompareView, DataView, DecodingView, EmptyState, StatesView,
    TrainingView, decode_trial,
)
from .workers import (
    BICScanResult, BICScanWorker, FitWorker, ParallelBICScanWorker,
    ParallelFitWorker, Run,
)


# ----------------------------------------------------------------------------
class StudioWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HMM Studio · Sticky Poisson / Gaussian / Multinoulli HMMs")
        self.resize(1500, 950)

        # ----- state -----
        self.dataset: Optional[Dataset] = None
        self.runs: list[Run] = []
        self.active_run_id: Optional[str] = None
        self.bic_result: Optional[BICScanResult] = None
        self.project_path: Optional[Path] = None
        self._worker_thread: Optional[QThread] = None
        self._worker = None
        self._session = state_store.load()

        # ----- toolbar -----
        tb = QToolBar(); tb.setMovable(False); tb.setIconSize(QSize(18, 18))
        self.addToolBar(tb)
        self._act_load = QAction("Load data…", self)
        self._act_load.setShortcut(QKeySequence.Open)
        self._act_load.setToolTip("Open a CSV / NPY / NPZ / MAT file  (Ctrl+O)")
        self._act_load_project = QAction("Load project...", self)
        self._act_load_project.setShortcut("Ctrl+Shift+O")
        self._act_load_project.setToolTip("Open a saved HMM Studio project with fitted runs")
        self._act_save_project = QAction("Save project...", self)
        self._act_save_project.setShortcut(QKeySequence.Save)
        self._act_save_project.setToolTip("Save the dataset, fitted runs, BIC scan, and active view")
        self._act_demo_counts = QAction("Demo · counts", self)
        self._act_demo_counts.setToolTip("Generate a synthetic spike-count dataset")
        self._act_demo_cont = QAction("Demo · continuous", self)
        self._act_demo_cont.setToolTip("Generate a synthetic continuous (photometry-like) dataset")
        self._act_clear = QAction("Clear session", self)
        self._act_clear.setToolTip("Discard the current dataset and all runs")
        self._act_snap = QAction("Export tab → PNG", self)
        self._act_snap.setShortcut("Ctrl+Shift+S")
        self._act_snap.setToolTip("Save the current tab as a PNG image  (Ctrl+Shift+S)")
        self._act_about = QAction("About", self)
        self._act_about.setToolTip("About HMM Studio")
        for a in (self._act_load, self._act_load_project, self._act_save_project,
                  self._act_demo_counts, self._act_demo_cont,
                  self._act_clear, self._act_snap, self._act_about):
            tb.addAction(a)
        self._act_load.triggered.connect(self._toolbar_load)
        self._act_load_project.triggered.connect(self._load_project)
        self._act_save_project.triggered.connect(self._save_project)
        self._act_demo_counts.triggered.connect(lambda: self._load_demo("counts"))
        self._act_demo_cont.triggered.connect(lambda: self._load_demo("continuous"))
        self._act_clear.triggered.connect(self._clear_session)
        self._act_snap.triggered.connect(self._snapshot_tab)
        self._act_about.triggered.connect(self._show_about)

        # ----- central tabs -----
        self.tabs = QTabWidget(); self.tabs.setDocumentMode(True)
        self.view_data = DataView()
        self.view_decoding = DecodingView()
        self.view_states = StatesView()
        self.view_training = TrainingView()
        self.view_bic = BICView()
        self.view_compare = CompareView()
        self._empty_data = EmptyState(
            "Welcome to HMM Studio",
            "Load your own time series with  Upload file…  on the left,\n"
            "or click  Demo counts  to explore with a synthetic dataset.\n\n"
            "Supported formats: CSV · TSV · NPY · NPZ · MATLAB .mat",
            glyph="◉")
        self.tabs.addTab(self._empty_data, "Data")
        self.tabs.addTab(EmptyState("No fit yet",
                                    "Fit a model from the left panel to see the\n"
                                    "decoded states overlaid on your data.",
                                    glyph="◐"), "Decoding")
        self.tabs.addTab(EmptyState("No fit yet",
                                    "Per-state emission profiles, the transition\n"
                                    "matrix, and the stationary distribution appear here.",
                                    glyph="▦"), "States")
        self.tabs.addTab(EmptyState("No fit yet",
                                    "Training log-likelihood and self-transition\n"
                                    "trajectories appear here.",
                                    glyph="∿"), "Training")
        self.tabs.addTab(self.view_bic, "Model selection (BIC)")
        self.tabs.addTab(self.view_compare, "Compare")
        self.setCentralWidget(self.tabs)

        # ----- left dock: controls -----
        self.controls = ControlPanel()
        dock_l = QDockWidget("Controls")
        dock_l.setWidget(self.controls)
        dock_l.setFeatures(QDockWidget.NoDockWidgetFeatures)
        dock_l.setAllowedAreas(Qt.LeftDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock_l)

        # ----- right dock: runs -----
        self.runs_panel = RunsPanel()
        dock_r = QDockWidget("Runs")
        dock_r.setWidget(self.runs_panel)
        dock_r.setFeatures(QDockWidget.NoDockWidgetFeatures)
        dock_r.setAllowedAreas(Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, dock_r)

        # ----- status bar -----
        self.setStatusBar(QStatusBar())
        self._status_data = QLabel("No dataset")
        self._status_runs = QLabel("0 runs")
        self.statusBar().addPermanentWidget(self._status_data)
        self.statusBar().addPermanentWidget(self._status_runs)
        self.statusBar().showMessage("Ready.")

        # ----- wire signals -----
        c = self.controls
        c.load_requested.connect(self._on_load_path)
        c.demo_requested.connect(self._load_demo)
        c.preprocess_requested.connect(self._on_preprocess)
        c.trial_changed.connect(self._on_trial_changed)
        c.fit_requested.connect(self._on_fit)
        c.scan_requested.connect(self._on_scan)
        c.cancel_requested.connect(self._cancel_worker)

        r = self.runs_panel
        r.activate_requested.connect(self._activate_run)
        r.remove_requested.connect(self._remove_run)
        r.compare_requested.connect(self._compare_runs)
        r.export_states_requested.connect(self._export_states)
        r.export_model_requested.connect(self._export_model)
        r.export_posteriors_requested.connect(self._export_posteriors)

        self._refresh_status()
        self._restore_session()

    # ------------------------------------------------------------------
    # Persistent session state
    # ------------------------------------------------------------------
    def _restore_session(self):
        """Apply remembered UI defaults and offer to reload the last dataset."""
        s = self._session
        if s.workers is not None:
            try:
                self.controls.workers.setValue(int(s.workers))
            except Exception:
                pass
        if s.n_states:
            try:
                self.controls.n_states.setValue(int(s.n_states))
            except Exception:
                pass
        if s.restarts:
            try:
                self.controls.restarts.setValue(int(s.restarts))
            except Exception:
                pass
        if s.scan_min and s.scan_max:
            try:
                self.controls.scan_min.setValue(int(s.scan_min))
                self.controls.scan_max.setValue(int(s.scan_max))
            except Exception:
                pass
        if s.method_key:
            idx = self.controls.method_combo.findData(s.method_key)
            if idx >= 0:
                self.controls.method_combo.setCurrentIndex(idx)

        # Auto-reload the last dataset path if it still exists.
        if s.last_data_path and Path(s.last_data_path).exists():
            self.statusBar().showMessage(
                f"Reloading last dataset: {Path(s.last_data_path).name}…", 6000)
            try:
                self._on_load_path(
                    Path(s.last_data_path),
                    s.last_data_kind or "",
                    s.last_orientation or "auto",
                    float(s.last_bin_size or 0.05),
                    Path(s.last_data_path).stem,
                )
            except Exception as exc:
                self.statusBar().showMessage(f"Could not auto-reload: {exc}", 8000)

    def _persist_session(self, **overrides):
        try:
            kw = dict(
                workers=int(self.controls.workers.value()),
                n_states=int(self.controls.n_states.value()),
                restarts=int(self.controls.restarts.value()),
                scan_min=int(self.controls.scan_min.value()),
                scan_max=int(self.controls.scan_max.value()),
                method_key=self.controls.method_combo.currentData() or "",
            )
            if self.dataset is not None:
                kw["last_data_path"] = self.dataset.source or ""
                kw["last_data_kind"] = self.dataset.kind
                kw["last_bin_size"] = float(self.dataset.bin_size)
            kw.update(overrides)
            state_store.update(**kw)
        except Exception:
            pass

    def _default_save_dir(self) -> str:
        """Folder Save dialogs should open in: the loaded dataset's folder."""
        if self.dataset and self.dataset.source:
            p = Path(self.dataset.source)
            if p.exists():
                return str(p.parent)
        if self._session.last_export_dir and Path(self._session.last_export_dir).exists():
            return self._session.last_export_dir
        return str(Path.home())

    def _safe_project_stem(self) -> str:
        if self.dataset and self.dataset.source:
            stem = Path(self.dataset.source).stem
        elif self.dataset:
            stem = self.dataset.name
        else:
            stem = "hmm_studio_project"
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)
        return safe.strip("._-") or "hmm_studio_project"

    def _suggest_project_path(self) -> str:
        if self.project_path:
            return str(self.project_path)
        return str(Path(self._default_save_dir()) / f"{self._safe_project_stem()}.hmmstudio")

    # ------------------------------------------------------------------
    # Project files
    # ------------------------------------------------------------------
    def _save_project(self):
        if self.dataset is None:
            QMessageBox.information(self, "No project", "Load data or a demo before saving a project.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save HMM Studio project",
            self._suggest_project_path(),
            "HMM Studio project (*.hmmstudio);;All files (*.*)",
        )
        if not path:
            return
        path_obj = Path(path)
        if not path_obj.suffix:
            path_obj = path_obj.with_suffix(".hmmstudio")
        try:
            trial_index = max(self.controls.trial_select.currentIndex(), 0)
            dataio.save_project(
                path_obj,
                dataset=self.dataset,
                runs=self.runs,
                active_run_id=self.active_run_id,
                bic_result=self.bic_result,
                trial_index=trial_index,
                ui_state=self.controls.current_project_ui_state(),
            )
            self.project_path = path_obj
            self.statusBar().showMessage(f"Project saved to {path_obj}", 7000)
            self._persist_session(last_export_dir=str(path_obj.parent))
        except Exception as exc:
            QMessageBox.critical(self, "Save project error", str(exc))

    def _load_project(self):
        if self._worker_thread is not None:
            QMessageBox.information(self, "Busy", "Wait for the running fit or scan before loading a project.")
            return
        if self.dataset is not None or self.runs:
            reply = QMessageBox.question(
                self,
                "Load project",
                "Replace the current dataset and fitted runs with the project file?",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Yes,
            )
            if reply != QMessageBox.Yes:
                return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load HMM Studio project",
            self._default_save_dir(),
            "HMM Studio project (*.hmmstudio *.hmmsproj);;All files (*.*)",
        )
        if not path:
            return
        try:
            payload = dataio.load_project(path)
            self._apply_project_payload(payload, Path(path))
            self.statusBar().showMessage(
                f"Project loaded: {Path(path).name} with {len(self.runs)} fitted run(s)",
                8000,
            )
            self._persist_session(last_export_dir=str(Path(path).parent))
        except Exception as exc:
            QMessageBox.critical(self, "Load project error", str(exc))

    def _apply_project_payload(self, payload: dict, path: Path | None = None):
        ds = payload["dataset"]
        runs = list(payload.get("runs") or [])
        run_ids = {r.run_id for r in runs}
        active_id = payload.get("active_run_id")
        if active_id not in run_ids:
            active_id = runs[-1].run_id if runs else None

        self.project_path = path
        self.dataset = ds
        self.runs = runs
        self.active_run_id = active_id
        self.bic_result = payload.get("bic_result")

        self.controls.set_dataset(ds)
        self.controls.apply_project_ui_state(payload.get("ui_state") or {})
        trial_idx = self.controls.trial_select.currentIndex()
        if trial_idx < 0:
            trial_idx = int(payload.get("trial_index") or 0)
        trial_idx = max(0, min(trial_idx, ds.n_trials - 1))
        self.controls.trial_select.setCurrentIndex(trial_idx)

        self._replace_tab("Data", self.view_data)
        self.view_data.update_dataset(ds, trial_idx)
        self.runs_panel.set_runs(self.runs)
        self.view_compare.update_runs(self.runs, self.dataset, trial_idx)
        self.view_bic.update_bic(self.bic_result)

        active_run = self._run_by_id(active_id) if active_id else None
        if active_run is not None:
            self._render_run(active_run, trial_idx)
        else:
            self._render_empty_run_tabs()
            self.tabs.setCurrentWidget(self.view_data)
        self._refresh_status()

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------
    def _toolbar_load(self):
        # delegate to controls so kind/orientation/bin-size come from UI
        self.controls.btn_load.click()

    def _on_load_path(self, path: Path, kind: str, orient: str, bin_size: float, name: str):
        # CSV / TSV / TXT / DAT files go through the adaptive import dialog
        # so the user can skip metadata rows and pick columns of interest.
        if path.suffix.lower() in {".csv", ".tsv", ".txt", ".dat"}:
            from .csv_import import run_import

            try:
                ds = run_import(path, parent=self)
            except Exception as exc:
                QMessageBox.critical(self, "Load error", f"{exc}")
                return
            if ds is None:
                return  # user cancelled
            self._set_dataset(ds)
            self.statusBar().showMessage(f"Loaded {ds.name}: {ds.info()}", 6000)
            return

        # Binary / structured formats use the fast path.
        try:
            raw = dataio.read_raw(path)
            chosen_kind = kind if kind else raw.suggested_kind
            ds = dataio.build_dataset(raw, kind=chosen_kind, bin_size=bin_size,
                                      time_axis=orient, name=name,
                                      source=str(path))
            self._set_dataset(ds)
            self.statusBar().showMessage(f"Loaded {ds.name}: {ds.info()}", 6000)
        except Exception as exc:
            QMessageBox.critical(self, "Load error", f"{exc}")

    def _load_demo(self, kind: str):
        if kind == "continuous":
            ds = dataio.demo_continuous()
        else:
            ds = dataio.demo_spike_counts()
        self._set_dataset(ds)
        self.statusBar().showMessage(f"Generated synthetic demo: {ds.info()}", 6000)

    def _on_preprocess(self, threshold_z: float, refractory: float, polarity: str):
        if self.dataset is None or self.dataset.kind != "continuous":
            QMessageBox.information(self, "Preprocess",
                                    "Load a continuous dataset first.")
            return
        try:
            ds = dataio.events_from_continuous(self.dataset, threshold_z=threshold_z,
                                               refractory_s=refractory,
                                               polarity=polarity)
            self._set_dataset(ds)
            self.statusBar().showMessage(f"Detected events → {ds.info()}", 6000)
        except Exception as exc:
            QMessageBox.critical(self, "Preprocess error", str(exc))

    def _set_dataset(self, ds: Dataset):
        self.project_path = None
        self.dataset = ds
        self.controls.set_dataset(ds)
        # Replace data tab with DataView
        self.tabs.removeTab(0)
        self.tabs.insertTab(0, self.view_data, "Data")
        self.view_data.update_dataset(ds, 0)
        # Reset downstream views to empty placeholders if dataset incompatible
        self.tabs.setCurrentIndex(0)
        self._refresh_status()
        self._persist_session()

    def _on_trial_changed(self, idx: int):
        if self.dataset is None: return
        self.view_data.update_dataset(self.dataset, idx)
        self.view_compare.update_runs(self.runs, self.dataset, idx)
        run = self._active_run()
        if run is not None:
            self._render_run(run, idx)

    # ------------------------------------------------------------------
    # Fitting / BIC scan
    # ------------------------------------------------------------------
    def _ensure_compatible(self, method_key: str) -> bool:
        if self.dataset is None:
            QMessageBox.information(self, "No dataset", "Load or generate data first.")
            return False
        spec = REGISTRY[method_key]
        if not spec.compatible(self.dataset.kind):
            QMessageBox.warning(
                self, "Incompatible method",
                f"{spec.label} needs '{spec.requires}' data, but the dataset is "
                f"'{self.dataset.kind}'. If you have continuous data and want a "
                f"Poisson model, use the event-detection preprocessor first."
            )
            return False
        return True

    def _start_worker(self, worker, on_finished, on_failed, on_progress=None):
        if self._worker_thread is not None:
            QMessageBox.information(self, "Busy", "A job is already running.")
            return False
        thread = QThread(self)
        worker.moveToThread(thread)
        if on_progress:
            worker.progress.connect(on_progress)
        worker.finished.connect(on_finished)
        worker.failed.connect(on_failed)
        worker.finished.connect(lambda *_: self._stop_worker())
        worker.failed.connect(lambda *_: self._stop_worker())
        thread.started.connect(worker.run)
        self._worker_thread = thread
        self._worker = worker
        self.controls.set_busy(True, "Starting...")
        thread.start()
        return True

    def _stop_worker(self):
        if self._worker_thread is None: return
        self._worker_thread.quit()
        self._worker_thread.wait(2000)
        self._worker_thread = None
        self._worker = None
        self.controls.set_busy(False, "Ready.")

    def _cancel_worker(self):
        if self._worker is not None and hasattr(self._worker, "cancel"):
            self._worker.cancel()
            self.statusBar().showMessage("Cancelling...", 2000)

    def _on_fit(self, method_key: str, n_states: int, params: dict,
                restarts: int, workers: int = 1):
        if not self._ensure_compatible(method_key): return
        self._persist_session()
        spec = REGISTRY[method_key]
        rng = np.random.default_rng(0)
        prep = spec.prepare(self.dataset.trials, self.dataset.kind,
                            self.dataset.bin_size, rng)
        if workers > 1 and restarts > 1:
            worker = ParallelFitWorker(
                spec, prep, n_states, params, restarts, base_seed=42,
                dataset_name=self.dataset.name, dataset_kind=self.dataset.kind,
                max_workers=workers,
            )
        else:
            worker = FitWorker(spec, prep, n_states, params, restarts,
                               base_seed=42, dataset_name=self.dataset.name,
                               dataset_kind=self.dataset.kind)
        self._start_worker(
            worker,
            on_finished=self._on_fit_done,
            on_failed=self._on_worker_failed,
            on_progress=self._on_progress,
        )

    def _on_scan(self, method_key: str, k_values: list, params: dict,
                 restarts: int, workers: int = 1):
        if not self._ensure_compatible(method_key): return
        # Warn the user when a scan would launch a really heavy workload.
        total_fits = len(k_values) * restarts
        n_bins = self.dataset.n_bins if self.dataset else 0
        if total_fits >= 12 and n_bins >= 20000:
            est = total_fits / max(workers, 1)   # approximate "rounds" needed
            r = QMessageBox.question(
                self, "Heavy BIC scan",
                f"This will run <b>{total_fits} fits</b> on a {n_bins:,}-bin dataset.\n\n"
                f"With <b>{workers} parallel worker(s)</b> that is about "
                f"<b>{est:.0f} sequential rounds</b> of EM. Long signals are slow per fit.\n\n"
                "Tips to make it tractable:\n"
                "  • increase the bin size in the importer (e.g. 0.05–0.2 s)\n"
                "  • reduce the K-range or restarts\n"
                "  • lower workers if your machine struggles\n\n"
                "Start the scan now?",
                QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Yes,
            )
            if r != QMessageBox.Yes:
                return
        self._persist_session()
        spec = REGISTRY[method_key]
        rng = np.random.default_rng(0)
        prep = spec.prepare(self.dataset.trials, self.dataset.kind,
                            self.dataset.bin_size, rng)
        if workers > 1:
            worker = ParallelBICScanWorker(
                spec, prep, k_values, params, restarts, base_seed=42,
                dataset_name=self.dataset.name, dataset_kind=self.dataset.kind,
                max_workers=workers,
            )
        else:
            worker = BICScanWorker(spec, prep, k_values, params, restarts,
                                   base_seed=42, dataset_name=self.dataset.name,
                                   dataset_kind=self.dataset.kind)
        worker.k_done.connect(self._on_scan_k_done)
        self._start_worker(
            worker,
            on_finished=self._on_scan_done,
            on_failed=self._on_worker_failed,
            on_progress=self._on_progress,
        )

    def _on_progress(self, current: int, total: int, msg: str):
        self.controls.set_progress(current, total, msg)

    def _on_worker_failed(self, msg: str):
        QMessageBox.critical(self, "Job failed", msg)

    def _on_fit_done(self, run: Run):
        self._add_run(run)
        self._activate_run(run.run_id)
        self.statusBar().showMessage(
            f"Fit complete: {run.name}  ·  iter={run.n_iter}  ·  {run.elapsed_s*1000:.0f} ms",
            6000,
        )

    def _on_scan_k_done(self, k: int, run: Run):
        # Add each K's best run to the runs list so the user can inspect any K.
        run.name = f"{run.method_label} · K={k}  (BIC scan)"
        self._add_run(run)

    def _on_scan_done(self, bic_result: BICScanResult):
        self.bic_result = bic_result
        self.view_bic.update_bic(bic_result)
        idx = self.tabs.indexOf(self.view_bic)
        if idx < 0:
            idx = self.tabs.addTab(self.view_bic, "Model selection (BIC)")
        self.tabs.setCurrentIndex(idx)
        if bic_result.best_k is not None:
            best_run = bic_result.by_k[bic_result.best_k]["best_run"]
            if best_run is not None:
                self._activate_run(best_run.run_id)
            self.statusBar().showMessage(
                f"BIC scan complete. Best K = {bic_result.best_k}", 8000)
        else:
            self.statusBar().showMessage("BIC scan finished with no valid model.", 6000)

    # ------------------------------------------------------------------
    # Runs management
    # ------------------------------------------------------------------
    def _add_run(self, run: Run):
        self.runs.append(run)
        self.runs_panel.set_runs(self.runs)
        self.view_compare.update_runs(self.runs, self.dataset, self._current_trial_index())
        self._refresh_status()

    def _remove_run(self, run_id: str):
        self.runs = [r for r in self.runs if r.run_id != run_id]
        self.runs_panel.set_runs(self.runs)
        self.view_compare.update_runs(self.runs, self.dataset, self._current_trial_index())
        if self.active_run_id == run_id:
            self.active_run_id = None
            self._render_empty_run_tabs()
        self._refresh_status()

    def _active_run(self) -> Optional[Run]:
        for r in self.runs:
            if r.run_id == self.active_run_id:
                return r
        return None

    def _current_trial_index(self) -> int:
        idx = self.controls.trial_select.currentIndex()
        if self.dataset is None:
            return 0
        return max(0, min(idx if idx >= 0 else 0, self.dataset.n_trials - 1))

    def _activate_run(self, run_id: str):
        run = next((r for r in self.runs if r.run_id == run_id), None)
        if run is None: return
        self.active_run_id = run_id
        ti = self.controls.trial_select.currentIndex()
        if ti < 0: ti = 0
        self._render_run(run, ti)

    def _render_run(self, run: Run, trial_idx: int):
        # Switch tab widgets to the real views
        self._replace_tab("Decoding", self.view_decoding)
        self._replace_tab("States", self.view_states)
        self._replace_tab("Training", self.view_training)
        self.view_decoding.update_run(run, self.dataset, trial_idx)
        self.view_states.update_run(run, self.dataset, trial_idx)
        self.view_training.update_run(run, self.dataset, trial_idx)
        self.tabs.setCurrentWidget(self.view_decoding)

    def _render_empty_run_tabs(self):
        self._replace_tab("Decoding", EmptyState("No active run", "Select a run on the right to view it."))
        self._replace_tab("States", EmptyState("No active run", ""))
        self._replace_tab("Training", EmptyState("No active run", ""))

    def _replace_tab(self, title: str, widget: QWidget):
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == title:
                self.tabs.removeTab(i)
                self.tabs.insertTab(i, widget, title)
                return
        self.tabs.addTab(widget, title)

    def _compare_runs(self, run_ids: list):
        runs = [r for r in self.runs if r.run_id in run_ids]
        self.view_compare.update_runs(runs if runs else self.runs, self.dataset, self._current_trial_index())
        self.tabs.setCurrentWidget(self.view_compare)

    # ------------------------------------------------------------------
    # Exports
    # ------------------------------------------------------------------
    def _run_by_id(self, run_id: str) -> Optional[Run]:
        return next((r for r in self.runs if r.run_id == run_id), None)

    def _suggest_save_path(self, run, suffix: str, ext: str) -> str:
        """Suggested path: <source-folder>/<source-name>_<run_tag>_<suffix>.<ext>."""
        folder = self._default_save_dir()
        if self.dataset and self.dataset.source:
            stem = Path(self.dataset.source).stem
        else:
            stem = run.dataset_name
        return str(Path(folder) / f"{stem}_{run.method_key}_K{run.n_states}_{suffix}.{ext}")

    def _export_states(self, run_id: str):
        run = self._run_by_id(run_id)
        if run is None or self.dataset is None: return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export states CSV",
            self._suggest_save_path(run, "states", "csv"),
            "CSV (*.csv)")
        if not path: return
        try:
            decodes = [decode_trial(run, i) for i in range(self.dataset.n_trials)]
            decodes = [(d.posterior, d.viterbi) for d in decodes]
            dataio.export_states_csv(path, run, decodes, self.dataset)
            self.statusBar().showMessage(f"States written to {path}", 6000)
            self._persist_session(last_export_dir=str(Path(path).parent))
        except Exception as exc:
            QMessageBox.critical(self, "Export error", str(exc))

    def _export_model(self, run_id: str):
        run = self._run_by_id(run_id)
        if run is None: return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export model NPZ",
            self._suggest_save_path(run, "model", "npz"),
            "NumPy archive (*.npz)")
        if not path: return
        try:
            dataio.export_model_npz(path, run)
            self.statusBar().showMessage(f"Model written to {path}", 6000)
            self._persist_session(last_export_dir=str(Path(path).parent))
        except Exception as exc:
            QMessageBox.critical(self, "Export error", str(exc))

    def _export_posteriors(self, run_id: str):
        run = self._run_by_id(run_id)
        if run is None or self.dataset is None: return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export posteriors NPZ",
            self._suggest_save_path(run, "posteriors", "npz"),
            "NumPy archive (*.npz)")
        if not path: return
        try:
            decodes = [decode_trial(run, i) for i in range(self.dataset.n_trials)]
            decodes_t = [(d.posterior, d.viterbi) for d in decodes]
            dataio.export_posteriors_npz(path, decodes_t)
            self.statusBar().showMessage(f"Posteriors written to {path}", 6000)
            self._persist_session(last_export_dir=str(Path(path).parent))
        except Exception as exc:
            QMessageBox.critical(self, "Export error", str(exc))

    def _snapshot_tab(self):
        widget = self.tabs.currentWidget()
        if widget is None: return
        tab_name = self.tabs.tabText(self.tabs.currentIndex()).replace(" ", "_") or "tab"
        default = str(Path(self._default_save_dir()) / f"hmm_studio_{tab_name}.png")
        path, _ = QFileDialog.getSaveFileName(self, "Save tab as PNG", default,
                                              "PNG (*.png)")
        if not path: return
        pix = widget.grab()
        pix.save(path, "PNG")
        self.statusBar().showMessage(f"Saved screenshot to {path}", 4000)
        self._persist_session(last_export_dir=str(Path(path).parent))

    # ------------------------------------------------------------------
    def _clear_session(self):
        if QMessageBox.question(self, "Clear session",
                                "Discard the dataset and all runs?") != QMessageBox.Yes:
            return
        self.dataset = None
        self.runs = []
        self.active_run_id = None
        self.bic_result = None
        self.project_path = None
        self.controls.set_dataset(None)
        self.runs_panel.set_runs([])
        self.view_bic.update_bic(None)
        self.view_compare.update_runs([], None, 0)
        self._render_empty_run_tabs()
        self.tabs.removeTab(0)
        self.tabs.insertTab(0, self._empty_data, "Data")
        self.tabs.setCurrentIndex(0)
        self._refresh_status()

    def _refresh_status(self):
        if self.dataset:
            self._status_data.setText(
                f"  Dataset: {self.dataset.name}  ·  {self.dataset.info()}  ")
        else:
            self._status_data.setText("  No dataset  ")
        self._status_runs.setText(f"  {len(self.runs)} run(s)  ")

    def _show_about(self):
        QMessageBox.about(
            self, "About HMM Studio",
            "<h3>HMM Studio</h3>"
            "<p>Elegant desktop GUI for sticky Poisson / Gaussian / Multinoulli "
            "and graph-informed HMMs.</p>"
            "<p>Method reference: Li & La Camera (2025) <i>PLOS ONE</i> 20(7): e0325979.</p>"
            "<p>Backed by the local <code>hmm_spikes</code> package.</p>"
        )


# ----------------------------------------------------------------------------
def run() -> int:
    """Entry point."""
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(theme.stylesheet())
    theme.configure_pyqtgraph()
    win = StudioWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(run())
