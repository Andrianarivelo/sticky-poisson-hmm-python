"""Left control panel and right runs panel."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame,
    QGroupBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QProgressBar,
    QPushButton, QScrollArea, QSizePolicy, QSpinBox, QVBoxLayout, QWidget,
)

from . import dataio, theme
from .models import REGISTRY, MethodSpec, ParamSpec, methods_for_kind


# ----------------------------------------------------------------------------
# Helper: dynamic input from a ParamSpec
# ----------------------------------------------------------------------------
def _make_input(spec: ParamSpec) -> tuple[QWidget, callable]:
    if spec.kind == "int":
        w = QSpinBox()
        w.setRange(int(spec.minimum), int(spec.maximum))
        w.setSingleStep(int(spec.step) or 1)
        w.setValue(int(spec.default))
        w.setToolTip(spec.help or spec.label)
        return w, w.value
    if spec.kind == "float":
        w = QDoubleSpinBox()
        w.setDecimals(int(spec.decimals))
        w.setRange(float(spec.minimum), float(spec.maximum))
        w.setSingleStep(float(spec.step))
        w.setValue(float(spec.default))
        w.setToolTip(spec.help or spec.label)
        return w, w.value
    if spec.kind == "bool":
        w = QCheckBox(spec.label); w.setChecked(bool(spec.default))
        w.setToolTip(spec.help or spec.label)
        return w, w.isChecked
    raise ValueError(spec.kind)


# ----------------------------------------------------------------------------
# Control panel
# ----------------------------------------------------------------------------
class ControlPanel(QWidget):
    load_requested = Signal(Path, str, str, float, str)  # path, kind, time_axis, bin_size, name
    demo_requested = Signal(str)                          # 'counts' | 'continuous'
    preprocess_requested = Signal(float, float, str)      # threshold_z, refractory_s, polarity
    trial_changed = Signal(int)
    fit_requested = Signal(str, int, dict, int, int)      # method_key, K, params, restarts, workers
    scan_requested = Signal(str, list, dict, int, int)    # method_key, K_list, params, restarts, workers
    cancel_requested = Signal()

    def __init__(self):
        super().__init__()
        self.setMinimumWidth(360)
        outer = QVBoxLayout(self); outer.setContentsMargins(12, 12, 12, 12); outer.setSpacing(10)

        # First-run hint (auto-hidden once a dataset is loaded)
        self.hint_banner = QLabel(
            "<b style='color:%s'>Get started:</b>  click <b>Upload file…</b> or <b>Demo counts</b>."
            % theme.ACCENT
        )
        self.hint_banner.setStyleSheet(
            f"background:{theme.PANEL}; border:1px solid {theme.ACCENT_DK};"
            f" border-radius:8px; padding:8px 12px;"
        )
        self.hint_banner.setWordWrap(True)
        outer.addWidget(self.hint_banner)

        # ----- scroll area -----
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
        inner = QWidget(); v = QVBoxLayout(inner); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(12)
        scroll.setWidget(inner)
        outer.addWidget(scroll, 1)

        # ----- Data group -----
        g_data = QGroupBox("Data")
        f = QFormLayout(g_data); f.setContentsMargins(8, 16, 8, 8); f.setSpacing(8)
        btn_row = QHBoxLayout()
        self.btn_load = QPushButton("Upload file…")
        self.btn_load.setObjectName("primary")
        self.btn_load.setToolTip("Open a CSV, TSV, NPY, NPZ, or MATLAB .mat file (Ctrl+O)")
        self.btn_demo_counts = QPushButton("Demo counts")
        self.btn_demo_counts.setToolTip("Generate a synthetic spike-count dataset to explore the app")
        self.btn_demo_cont = QPushButton("Demo continuous")
        self.btn_demo_cont.setToolTip("Generate a synthetic continuous (photometry-like) dataset")
        self.btn_demo_counts.setObjectName("ghost"); self.btn_demo_cont.setObjectName("ghost")
        btn_row.addWidget(self.btn_load); btn_row.addWidget(self.btn_demo_counts); btn_row.addWidget(self.btn_demo_cont)
        f.addRow(btn_row)
        self.kind_combo = QComboBox(); self.kind_combo.addItems(["counts", "continuous"])
        self.kind_combo.setToolTip("counts: non-negative integers (spike bins).\ncontinuous: real-valued traces (photometry, LFP, etc.)")
        self.bin_size = QDoubleSpinBox(); self.bin_size.setDecimals(4); self.bin_size.setRange(1e-5, 60.0); self.bin_size.setValue(0.05); self.bin_size.setSingleStep(0.005)
        self.bin_size.setSuffix(" s")
        self.bin_size.setToolTip("Sampling period of each time bin. Used to convert means to Hz.")
        self.time_axis = QComboBox(); self.time_axis.addItems(["auto", "columns are time", "rows are time"])
        self.time_axis.setToolTip("How the file's 2D arrays are oriented. Auto picks the longer axis as time.")
        self.trial_select = QComboBox()
        self.trial_select.setToolTip("Which trial to display in the plots.")
        self.info_label = QLabel("No dataset loaded."); self.info_label.setObjectName("hint"); self.info_label.setWordWrap(True)
        f.addRow("Data kind", self.kind_combo)
        f.addRow("Bin size", self.bin_size)
        f.addRow("Orientation", self.time_axis)
        f.addRow("Trial", self.trial_select)
        f.addRow(self.info_label)

        # ----- Preprocessing group -----
        g_prep = QGroupBox("Preprocess continuous → events  (optional)")
        f2 = QFormLayout(g_prep); f2.setContentsMargins(8, 16, 8, 8); f2.setSpacing(8)
        self.prep_thresh = QDoubleSpinBox(); self.prep_thresh.setRange(0.5, 10.0); self.prep_thresh.setValue(2.5); self.prep_thresh.setSingleStep(0.1); self.prep_thresh.setDecimals(2)
        self.prep_refractory = QDoubleSpinBox(); self.prep_refractory.setRange(0.0, 5.0); self.prep_refractory.setValue(0.2); self.prep_refractory.setSingleStep(0.05); self.prep_refractory.setDecimals(3); self.prep_refractory.setSuffix(" s")
        self.prep_polarity = QComboBox(); self.prep_polarity.addItems(["positive", "negative"])
        self.btn_prep = QPushButton("Detect events → new dataset")
        f2.addRow("Threshold (z)", self.prep_thresh)
        f2.addRow("Refractory", self.prep_refractory)
        f2.addRow("Polarity", self.prep_polarity)
        f2.addRow(self.btn_prep)

        # ----- Model group -----
        g_model = QGroupBox("Model")
        self.model_form = QFormLayout(g_model); self.model_form.setContentsMargins(8, 16, 8, 8); self.model_form.setSpacing(8)
        self.method_combo = QComboBox()
        self.method_combo.setToolTip("HMM family. Only methods compatible with the current data kind are listed.")
        self.method_desc = QLabel(""); self.method_desc.setWordWrap(True); self.method_desc.setObjectName("hint")
        self.n_states = QSpinBox(); self.n_states.setRange(2, 32); self.n_states.setValue(3)
        self.n_states.setToolTip("Number of latent states K the model will look for.")
        self.restarts = QSpinBox(); self.restarts.setRange(1, 200); self.restarts.setValue(5)
        self.restarts.setToolTip("Random restarts per fit. The best (by BIC, then LL) is kept.")
        import os as _os
        cpu = _os.cpu_count() or 1
        # Conservative default: keep at least half the cores free so the OS / GUI
        # stay snappy. A BIC scan with 4 parallel workers will run 4 fits at a
        # time while leaving the rest of the machine usable.
        default_workers = max(1, min(4, cpu // 2 or 1))
        self.workers = QSpinBox(); self.workers.setRange(1, max(32, cpu))
        self.workers.setValue(default_workers)
        self.workers.setToolTip(
            f"Parallel worker processes. Each fit runs in its own process at\n"
            f"background priority so the GUI stays responsive.\n\n"
            f"This machine has {cpu} CPU(s). The default ({default_workers}) keeps\n"
            f"the OS and the GUI snappy. Increase only if you don't mind your\n"
            f"computer being busy."
        )
        self._cpu_count = cpu
        self.scan_min = QSpinBox(); self.scan_min.setRange(2, 32); self.scan_min.setValue(2)
        self.scan_max = QSpinBox(); self.scan_max.setRange(2, 32); self.scan_max.setValue(6)
        for w in (self.scan_min, self.scan_max):
            w.setToolTip("BIC scan: sweep K from min to max (inclusive).")
        scan_row = QHBoxLayout(); scan_row.addWidget(self.scan_min); scan_row.addWidget(QLabel("→")); scan_row.addWidget(self.scan_max)
        self.model_form.addRow("Method", self.method_combo)
        self.model_form.addRow(self.method_desc)
        self.model_form.addRow("Number of states K", self.n_states)
        self.model_form.addRow("Restarts per fit", self.restarts)
        self.model_form.addRow("Parallel workers", self.workers)
        self.workers_warn = QLabel(""); self.workers_warn.setObjectName("hint")
        self.workers_warn.setWordWrap(True)
        self.workers_warn.setStyleSheet(f"color:{theme.WARN};")
        self.model_form.addRow(self.workers_warn)
        self.workers.valueChanged.connect(self._update_workers_warning)
        self._update_workers_warning(self.workers.value())
        self.model_form.addRow("BIC K-range", scan_row)
        # Dynamic params live below; tracked here so we can rebuild them.
        self._param_rows: list[tuple[QWidget, QWidget]] = []
        self._param_getters: dict[str, callable] = {}
        self._param_widgets: dict[str, QWidget] = {}

        v.addWidget(g_data); v.addWidget(g_prep); v.addWidget(g_model); v.addStretch()

        # ----- Pinned action footer (always visible, outside the scroll) -----
        footer = QFrame(); footer.setObjectName("controlFooter")
        footer.setStyleSheet(
            f"#controlFooter {{ background:{theme.BG_1}; border-top:1px solid {theme.BORDER};"
            f" border-radius:10px; padding:8px; }}"
        )
        f_lay = QVBoxLayout(footer); f_lay.setContentsMargins(8, 8, 8, 8); f_lay.setSpacing(6)

        actions = QHBoxLayout(); actions.setSpacing(6)
        self.btn_fit = QPushButton("▶  Fit model"); self.btn_fit.setObjectName("primary")
        self.btn_fit.setToolTip("Fit the selected HMM with the chosen K and restarts (Enter)")
        self.btn_fit.setShortcut("Return")
        self.btn_scan = QPushButton("Scan BIC")
        self.btn_scan.setToolTip("Sweep K across the BIC range and pick the best model")
        self.btn_cancel = QPushButton("Cancel"); self.btn_cancel.setEnabled(False)
        self.btn_cancel.setToolTip("Stop the running fit or BIC scan")
        actions.addWidget(self.btn_fit, 2); actions.addWidget(self.btn_scan, 1); actions.addWidget(self.btn_cancel, 1)
        f_lay.addLayout(actions)

        self.progress = QProgressBar(); self.progress.setRange(0, 1); self.progress.setValue(0)
        self.status_lbl = QLabel("Ready."); self.status_lbl.setObjectName("hint")
        f_lay.addWidget(self.progress); f_lay.addWidget(self.status_lbl)
        outer.addWidget(footer)

        # ----- Connections -----
        self.btn_load.clicked.connect(self._on_load_click)
        self.btn_demo_counts.clicked.connect(lambda: self.demo_requested.emit("counts"))
        self.btn_demo_cont.clicked.connect(lambda: self.demo_requested.emit("continuous"))
        self.btn_prep.clicked.connect(self._on_preprocess)
        self.kind_combo.currentTextChanged.connect(self._refresh_methods)
        self.method_combo.currentIndexChanged.connect(self._refresh_params)
        self.trial_select.currentIndexChanged.connect(lambda i: self.trial_changed.emit(max(i, 0)))
        self.btn_fit.clicked.connect(self._on_fit)
        self.btn_scan.clicked.connect(self._on_scan)
        self.btn_cancel.clicked.connect(self.cancel_requested.emit)

        self._refresh_methods(self.kind_combo.currentText())

    # ----- Public API -----
    def set_dataset(self, dataset):
        if dataset is None:
            self.info_label.setText("No dataset loaded.")
            self.trial_select.clear()
            self.btn_prep.setEnabled(False)
            self.hint_banner.show()
            return
        self.kind_combo.setCurrentText(dataset.kind)
        self.bin_size.setValue(float(dataset.bin_size))
        self.info_label.setText(dataset.info())
        self.trial_select.blockSignals(True)
        self.trial_select.clear()
        self.trial_select.addItems([f"Trial {i+1}" for i in range(dataset.n_trials)])
        self.trial_select.setCurrentIndex(0)
        self.trial_select.blockSignals(False)
        self.btn_prep.setEnabled(dataset.kind == "continuous")
        self._refresh_methods(dataset.kind)
        self.hint_banner.hide()

    def set_busy(self, busy: bool, message: str = ""):
        for b in (self.btn_fit, self.btn_scan, self.btn_load, self.btn_demo_counts,
                  self.btn_demo_cont, self.btn_prep):
            b.setEnabled(not busy)
        self.btn_cancel.setEnabled(busy)
        if message:
            self.status_lbl.setText(message)
        if busy:
            self.progress.setRange(0, 0)  # spinner
        else:
            self.progress.setRange(0, 1); self.progress.setValue(0)

    def set_progress(self, current: int, total: int, message: str):
        if total <= 0:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, total); self.progress.setValue(current)
        self.status_lbl.setText(message)

    def current_method_key(self) -> str | None:
        return self.method_combo.currentData()

    def current_project_ui_state(self) -> dict:
        return {
            "kind": self.kind_combo.currentText(),
            "bin_size": float(self.bin_size.value()),
            "time_axis_index": int(self.time_axis.currentIndex()),
            "trial_index": int(max(self.trial_select.currentIndex(), 0)),
            "method_key": self.method_combo.currentData() or "",
            "n_states": int(self.n_states.value()),
            "restarts": int(self.restarts.value()),
            "workers": int(self.workers.value()),
            "scan_min": int(self.scan_min.value()),
            "scan_max": int(self.scan_max.value()),
            "params": self._collect_params(),
        }

    def apply_project_ui_state(self, state: dict):
        if not state:
            return
        method_key = state.get("method_key") or ""
        if method_key:
            idx = self.method_combo.findData(method_key)
            if idx >= 0:
                self.method_combo.setCurrentIndex(idx)
        for attr in ("n_states", "restarts", "workers", "scan_min", "scan_max"):
            value = state.get(attr)
            widget = getattr(self, attr, None)
            if value is not None and hasattr(widget, "setValue"):
                widget.setValue(int(value))
        trial_index = state.get("trial_index")
        if trial_index is not None and self.trial_select.count():
            self.trial_select.setCurrentIndex(max(0, min(int(trial_index), self.trial_select.count() - 1)))
        params = state.get("params") or {}
        for name, value in params.items():
            widget = self._param_widgets.get(name)
            if widget is None:
                continue
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif hasattr(widget, "setValue"):
                widget.setValue(value)

    # ----- Internals -----
    def _refresh_methods(self, kind: str):
        self.method_combo.blockSignals(True)
        self.method_combo.clear()
        for spec in methods_for_kind(kind):
            self.method_combo.addItem(spec.label, spec.key)
        self.method_combo.blockSignals(False)
        self._refresh_params()

    def _clear_param_widgets(self):
        # Remove dynamic rows from the form layout.
        for label_w, input_w in self._param_rows:
            self.model_form.removeRow(input_w)  # removes label too
        self._param_rows.clear()
        self._param_getters.clear()
        self._param_widgets.clear()

    def _update_workers_warning(self, value: int):
        cpu = getattr(self, "_cpu_count", 1)
        if value > max(1, cpu // 2) and cpu > 2:
            self.workers_warn.setText(
                f"Warning: {value} workers on {cpu} CPUs will saturate the machine. "
                f"Your desktop may become sluggish during fitting."
            )
        else:
            self.workers_warn.setText("")

    def _refresh_params(self):
        self._clear_param_widgets()
        key = self.method_combo.currentData()
        if not key:
            self.method_desc.setText(""); return
        spec = REGISTRY[key]
        self.method_desc.setText(spec.description)
        for p in spec.params:
            input_w, getter = _make_input(p)
            self.model_form.addRow(p.label, input_w)
            self._param_rows.append((None, input_w))
            self._param_getters[p.name] = getter
            self._param_widgets[p.name] = input_w

    def _collect_params(self) -> dict:
        return {name: g() for name, g in self._param_getters.items()}

    def _on_load_click(self):
        # Start the file dialog in the folder of the last loaded file, if any.
        start_dir = ""
        try:
            from . import state_store
            last = state_store.load().last_data_path
            if last and Path(last).parent.exists():
                start_dir = str(Path(last).parent)
        except Exception:
            pass
        path, _ = QFileDialog.getOpenFileName(
            self, "Load time series",
            start_dir,
            filter="Time series (*.csv *.tsv *.txt *.dat *.npy *.npz *.mat);;All files (*.*)"
        )
        if not path:
            return
        kind = self.kind_combo.currentText()
        orient = ["auto", "columns", "rows"][self.time_axis.currentIndex()]
        self.load_requested.emit(Path(path), kind, orient, float(self.bin_size.value()), Path(path).stem)

    def _on_preprocess(self):
        self.preprocess_requested.emit(
            float(self.prep_thresh.value()),
            float(self.prep_refractory.value()),
            self.prep_polarity.currentText(),
        )

    def _on_fit(self):
        key = self.method_combo.currentData()
        if not key:
            return
        self.fit_requested.emit(key, int(self.n_states.value()),
                                self._collect_params(),
                                int(self.restarts.value()),
                                int(self.workers.value()))

    def _on_scan(self):
        key = self.method_combo.currentData()
        if not key:
            return
        lo, hi = int(self.scan_min.value()), int(self.scan_max.value())
        if hi < lo: lo, hi = hi, lo
        self.scan_requested.emit(key, list(range(lo, hi + 1)),
                                 self._collect_params(),
                                 int(self.restarts.value()),
                                 int(self.workers.value()))


# ----------------------------------------------------------------------------
# Runs panel (right dock)
# ----------------------------------------------------------------------------
class RunsPanel(QWidget):
    activate_requested = Signal(str)      # run_id
    remove_requested = Signal(str)
    export_states_requested = Signal(str)
    export_model_requested = Signal(str)
    export_posteriors_requested = Signal(str)
    compare_requested = Signal(list)      # list[run_id]

    def __init__(self):
        super().__init__()
        self.setMinimumWidth(300)
        v = QVBoxLayout(self); v.setContentsMargins(12, 12, 12, 12); v.setSpacing(8)
        sub = QLabel("Click to view · Ctrl-click to compare"); sub.setObjectName("hint")
        v.addWidget(sub)

        self.list = QListWidget(); self.list.setSelectionMode(QListWidget.ExtendedSelection)
        self.list.setToolTip("Each fit becomes a row. Single-click to view. Ctrl-click to multi-select for Compare.")
        v.addWidget(self.list, 1)

        # Two columns of buttons so the panel is shorter.
        btns = QVBoxLayout(); btns.setSpacing(6)
        row1 = QHBoxLayout(); row1.setSpacing(6)
        self.btn_view = QPushButton("View"); self.btn_view.setObjectName("primary")
        self.btn_view.setToolTip("Show this run in the Decoding / States / Training tabs")
        self.btn_compare = QPushButton("Compare")
        self.btn_compare.setToolTip("Send the selected runs to the Compare tab")
        row1.addWidget(self.btn_view); row1.addWidget(self.btn_compare)

        row2 = QHBoxLayout(); row2.setSpacing(6)
        self.btn_export_states = QPushButton("States CSV…")
        self.btn_export_states.setToolTip("Per-trial / per-bin posterior + Viterbi state sequence")
        self.btn_export_model = QPushButton("Model NPZ…")
        self.btn_export_model.setToolTip("Save means, transition matrix, history etc.")
        row2.addWidget(self.btn_export_states); row2.addWidget(self.btn_export_model)

        row3 = QHBoxLayout(); row3.setSpacing(6)
        self.btn_export_post = QPushButton("Posteriors NPZ…")
        self.btn_export_post.setToolTip("Per-trial posterior probability + Viterbi sequence arrays")
        self.btn_remove = QPushButton("Remove")
        self.btn_remove.setToolTip("Drop the selected run(s) from the session")
        row3.addWidget(self.btn_export_post); row3.addWidget(self.btn_remove)

        btns.addLayout(row1); btns.addLayout(row2); btns.addLayout(row3)
        v.addLayout(btns)

        self.btn_view.clicked.connect(self._on_view)
        self.btn_compare.clicked.connect(self._on_compare)
        self.btn_remove.clicked.connect(self._on_remove)
        self.btn_export_states.clicked.connect(self._on_export_states)
        self.btn_export_model.clicked.connect(self._on_export_model)
        self.btn_export_post.clicked.connect(self._on_export_post)
        self.list.itemClicked.connect(self._on_item_clicked)
        self.list.itemDoubleClicked.connect(lambda *_: self._on_view())

    def _selected_run_ids(self) -> list[str]:
        return [it.data(Qt.UserRole) for it in self.list.selectedItems()]

    def set_runs(self, runs: list):
        self.list.clear()
        for i, run in enumerate(runs, start=1):
            text = (f"#{i}  ·  {run.method_label}  ·  K={run.n_states}\n"
                    f"BIC {run.bic:,.1f}   LL {run.log_likelihood:,.1f}   "
                    f"{'conv' if run.converged else 'no-conv'} · "
                    f"{'θ ok' if run.threshold_satisfied else 'θ fail'}")
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, run.run_id)
            item.setToolTip(
                f"{run.method_label} · K={run.n_states}\n"
                f"Dataset: {run.dataset_name}   ({run.dataset_kind})\n"
                f"BIC: {run.bic:,.2f}\n"
                f"log-likelihood: {run.log_likelihood:,.2f}\n"
                f"iterations: {run.n_iter}    seed: {run.seed}\n"
                f"params: {run.params}\n"
                f"created: {run.timestamp}"
            )
            self.list.addItem(item)
        if self.list.count():
            self.list.setCurrentRow(self.list.count() - 1)

    # Actions
    def _on_item_clicked(self, item: QListWidgetItem):
        if QApplication.keyboardModifiers() & (Qt.ControlModifier | Qt.ShiftModifier):
            return
        run_id = item.data(Qt.UserRole)
        if run_id:
            self.activate_requested.emit(run_id)

    def _on_view(self):
        ids = self._selected_run_ids()
        if ids:
            self.activate_requested.emit(ids[0])

    def _on_compare(self):
        ids = self._selected_run_ids()
        if len(ids) >= 1:
            self.compare_requested.emit(ids)

    def _on_remove(self):
        for rid in self._selected_run_ids():
            self.remove_requested.emit(rid)

    def _on_export_states(self):
        ids = self._selected_run_ids()
        if ids: self.export_states_requested.emit(ids[0])

    def _on_export_model(self):
        ids = self._selected_run_ids()
        if ids: self.export_model_requested.emit(ids[0])

    def _on_export_post(self):
        ids = self._selected_run_ids()
        if ids: self.export_posteriors_requested.emit(ids[0])
