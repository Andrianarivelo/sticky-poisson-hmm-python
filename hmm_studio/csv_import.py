"""Adaptive CSV / TSV / TXT importer.

Lots of real-world time-series CSVs look nothing like a plain matrix:
fiber photometry files start with metadata lines, only the ``dFF`` column is
the signal of interest, spike-time exports use one column per neuron with
uneven lengths, etc. This dialog lets the user:

  * skip a configurable number of metadata / comment rows,
  * choose the delimiter (or auto-detect it),
  * mark the first data row as a header,
  * pick which columns to import,
  * choose how to interpret those columns:
       - continuous channels (each column is a signal),
       - spike times per column (cells are spike timestamps),
       - counts matrix (each row is a time bin).

It returns a :class:`hmm_studio.dataio.Dataset` ready for the rest of the app.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit, QPushButton,
    QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from . import dataio, theme


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _detect_delimiter(sample: str) -> str:
    counts = {d: sample.count(d) for d in [",", "\t", ";", "|", " "]}
    # Prefer tab over space if both appear a lot.
    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    for delim, count in ordered:
        if count > 0:
            return delim
    return ","


def _read_preview_lines(path: Path, n: int = 60) -> list[str]:
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        return [next(fh, "").rstrip("\n") for _ in range(n)]


# ----------------------------------------------------------------------------
class CsvImportDialog(QDialog):
    """Modal dialog returning a :class:`Dataset` (or ``None`` on cancel)."""

    def __init__(self, path: str | Path, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle(f"Import CSV — {Path(path).name}")
        self.resize(1180, 780)
        self._path = Path(path)
        self._preview_lines = _read_preview_lines(self._path, 80)
        self._df = None                  # last successful parse (pandas DataFrame)
        self._dataset: Optional[dataio.Dataset] = None

        # ----- header --------------------------------------------------
        root = QVBoxLayout(self); root.setContentsMargins(14, 14, 14, 14); root.setSpacing(10)
        title = QLabel(f"<b>{self._path.name}</b>"); title.setObjectName("h1")
        sub = QLabel(f"Use this dialog to skip metadata rows, pick a delimiter, "
                     f"choose columns to import, and decide how to interpret them.")
        sub.setObjectName("hint"); sub.setWordWrap(True)
        root.addWidget(title); root.addWidget(sub)

        # ----- splittable body: raw preview (top) / parse + select (bottom)
        body = QSplitter(Qt.Vertical); root.addWidget(body, 1)

        # Raw preview
        raw_box = QGroupBox("Raw file preview  (first 80 lines)")
        rl = QVBoxLayout(raw_box); rl.setContentsMargins(10, 18, 10, 10)
        self.raw_text = QPlainTextEdit(); self.raw_text.setReadOnly(True)
        self.raw_text.setFont(QFont("Cascadia Code, Consolas, monospace", 10))
        self.raw_text.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.raw_text.setPlainText("\n".join(self._preview_lines))
        rl.addWidget(self.raw_text)
        body.addWidget(raw_box)

        # Parse + interpretation
        bottom = QWidget(); bl = QHBoxLayout(bottom); bl.setContentsMargins(0, 0, 0, 0); bl.setSpacing(10)

        # ---- parse options
        parse_box = QGroupBox("Parsing"); pf = QFormLayout(parse_box); pf.setContentsMargins(10, 18, 10, 10); pf.setSpacing(6)
        self.skip_rows = QSpinBox(); self.skip_rows.setRange(0, 200); self.skip_rows.setValue(self._guess_skip_rows())
        self.skip_rows.setToolTip("Metadata / comment lines to skip before the data starts.")
        self.delim = QComboBox(); self.delim.addItems(["auto", ", (comma)", "\\t (tab)", "; (semicolon)", "| (pipe)", "  (space)"])
        self.delim.setToolTip("Field separator. Auto picks the most common candidate.")
        self.has_header = QCheckBox("First data row is a header")
        self.has_header.setChecked(True)
        self.has_header.setToolTip("If checked, the first non-skipped row provides the column names.")
        self.decimal = QComboBox(); self.decimal.addItems([".", ","])
        self.decimal.setToolTip("Decimal separator used in numeric values.")
        self.na_str = QComboBox(); self.na_str.setEditable(True)
        self.na_str.addItems(["", "NaN", "NA", "null"])
        self.na_str.setToolTip("Text that should be treated as missing.")
        pf.addRow("Skip rows", self.skip_rows)
        pf.addRow("Delimiter", self.delim)
        pf.addRow("Decimal", self.decimal)
        pf.addRow("NA tokens", self.na_str)
        pf.addRow(self.has_header)
        self.btn_reparse = QPushButton("Reparse")
        self.btn_reparse.setToolTip("Re-read the file with the current parsing options.")
        pf.addRow(self.btn_reparse)

        # ---- parsed preview
        parsed_box = QGroupBox("Parsed preview")
        pl = QVBoxLayout(parsed_box); pl.setContentsMargins(10, 18, 10, 10)
        self.parse_msg = QLabel(""); self.parse_msg.setObjectName("hint")
        self.parsed_table = QTableWidget(0, 0)
        self.parsed_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.parsed_table.verticalHeader().setVisible(False)
        pl.addWidget(self.parse_msg); pl.addWidget(self.parsed_table, 1)

        # ---- interpretation
        interp_box = QGroupBox("Interpretation"); il = QVBoxLayout(interp_box); il.setContentsMargins(10, 18, 10, 10); il.setSpacing(6)
        self.mode = QComboBox()
        self.mode.addItems([
            "Continuous channels  (each selected column is a signal)",
            "Spike times          (each selected column is a list of spike times)",
            "Counts matrix        (rows are bins, columns are channels of counts)",
        ])
        self.mode.setToolTip("How to interpret the parsed columns when building the dataset.")
        mode_row = QHBoxLayout(); mode_row.addWidget(QLabel("Mode")); mode_row.addWidget(self.mode, 1)
        il.addLayout(mode_row)

        # Column picker
        col_row = QHBoxLayout()
        col_lbl = QLabel("Columns to import")
        self.col_list = QListWidget(); self.col_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.col_list.setToolTip("Tick the columns to include. Untick metadata or time columns.")
        col_row.addWidget(col_lbl); col_row.addWidget(self.col_list, 1)
        il.addLayout(col_row, 1)
        col_btns = QHBoxLayout()
        self.btn_all = QPushButton("Select all"); self.btn_none = QPushButton("Select none")
        col_btns.addWidget(self.btn_all); col_btns.addWidget(self.btn_none); col_btns.addStretch()
        il.addLayout(col_btns)

        # Mode-specific parameters
        self.mode_form = QFormLayout(); self.mode_form.setSpacing(6)
        self.time_col = QComboBox()
        self.time_col.setToolTip("Optional column holding the time of each row. "
                                 "If picked, the bin size is inferred from successive differences.")
        self.bin_size = QDoubleSpinBox(); self.bin_size.setRange(1e-6, 60.0); self.bin_size.setDecimals(6); self.bin_size.setValue(0.05); self.bin_size.setSuffix(" s")
        self.bin_size.setToolTip("Time bin size (seconds). For spike times this is the bin used to count.")
        self.t_start = QDoubleSpinBox(); self.t_start.setRange(-1e9, 1e9); self.t_start.setDecimals(4); self.t_start.setValue(0.0); self.t_start.setSuffix(" s")
        self.t_start.setToolTip("Start of the time range used for spike-time binning.")
        self.t_end = QDoubleSpinBox(); self.t_end.setRange(-1e9, 1e9); self.t_end.setDecimals(4); self.t_end.setValue(60.0); self.t_end.setSuffix(" s")
        self.t_end.setToolTip("End of the time range used for spike-time binning.")
        self.btn_autorange = QPushButton("Auto from data")
        self.btn_autorange.setToolTip("Set the spike-time range from min/max of selected columns.")
        self.dataset_name = QComboBox(); self.dataset_name.setEditable(True)
        self.dataset_name.addItem(self._path.stem)
        self.dataset_name.setToolTip("Name shown in the runs panel.")

        self.mode_form.addRow("Time column", self.time_col)
        self.mode_form.addRow("Bin size", self.bin_size)
        self.mode_form.addRow("Start time", self.t_start)
        range_row = QHBoxLayout(); range_row.addWidget(self.t_end); range_row.addWidget(self.btn_autorange)
        self.mode_form.addRow("End time", range_row)
        self.mode_form.addRow("Dataset name", self.dataset_name)
        il.addLayout(self.mode_form)

        # arrange three columns in the bottom row
        bl.addWidget(parse_box, 1)
        bl.addWidget(parsed_box, 2)
        bl.addWidget(interp_box, 2)
        body.addWidget(bottom)
        body.setStretchFactor(0, 1); body.setStretchFactor(1, 2)

        # ----- buttons -------------------------------------------------
        self.message = QLabel(""); self.message.setObjectName("hint")
        self.message.setWordWrap(True)
        root.addWidget(self.message)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Import")
        buttons.button(QDialogButtonBox.Ok).setObjectName("primary")
        root.addWidget(buttons)

        # ----- signals -------------------------------------------------
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        self.btn_reparse.clicked.connect(self._reparse)
        self.skip_rows.valueChanged.connect(self._reparse)
        self.delim.currentIndexChanged.connect(self._reparse)
        self.has_header.toggled.connect(self._reparse)
        self.decimal.currentIndexChanged.connect(self._reparse)
        self.na_str.lineEdit().editingFinished.connect(self._reparse)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.btn_all.clicked.connect(lambda: self._set_all_checked(True))
        self.btn_none.clicked.connect(lambda: self._set_all_checked(False))
        self.btn_autorange.clicked.connect(self._auto_time_range)
        self.time_col.currentIndexChanged.connect(self._time_col_changed)

        self._reparse()
        self._mode_changed()

    # ------------------------------------------------------------------
    def _guess_skip_rows(self) -> int:
        """Pick a reasonable default for the metadata-row count."""
        for i, line in enumerate(self._preview_lines):
            stripped = line.strip()
            if not stripped:
                continue
            # Skip obvious comment / metadata lines
            if stripped.startswith("#") or stripped.startswith('"#') or stripped.startswith("%") or stripped.startswith("//"):
                continue
            return i
        return 0

    def _delim_value(self) -> Optional[str]:
        idx = self.delim.currentIndex()
        if idx == 0:
            joined = "\n".join(self._preview_lines[self.skip_rows.value():
                                                   self.skip_rows.value() + 5])
            return _detect_delimiter(joined)
        return {1: ",", 2: "\t", 3: ";", 4: "|", 5: " "}[idx]

    # ------------------------------------------------------------------
    def _reparse(self):
        try:
            import pandas as pd
        except Exception as exc:
            QMessageBox.critical(self, "pandas required",
                                 f"pandas is needed for adaptive CSV import: {exc}")
            return
        try:
            df = pd.read_csv(
                self._path,
                sep=self._delim_value(),
                engine="python",
                skiprows=int(self.skip_rows.value()),
                header=0 if self.has_header.isChecked() else None,
                decimal=self.decimal.currentText(),
                na_values=[self.na_str.currentText()] if self.na_str.currentText() else None,
                on_bad_lines="skip",
                skip_blank_lines=True,
                comment=None,
            )
            if not self.has_header.isChecked():
                df.columns = [f"col_{i+1}" for i in range(df.shape[1])]
        except Exception as exc:
            self.parse_msg.setText(f"<span style='color:{theme.BAD}'>Parse error: {exc}</span>")
            self.parsed_table.setRowCount(0); self.parsed_table.setColumnCount(0)
            self._df = None
            return
        self._df = df
        self._fill_preview_table(df)
        self._fill_column_picker(df)
        self._fill_time_col(df)
        self._mode_changed()
        n_num = sum(1 for c in df.columns if np.issubdtype(df[c].dtype, np.number))
        self.parse_msg.setText(
            f"Parsed {len(df):,} rows × {df.shape[1]} columns  ·  {n_num} numeric"
        )

    def _fill_preview_table(self, df):
        n_show = min(20, len(df))
        self.parsed_table.setRowCount(n_show)
        self.parsed_table.setColumnCount(df.shape[1])
        self.parsed_table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for r in range(n_show):
            for c in range(df.shape[1]):
                val = df.iat[r, c]
                txt = "" if val is None or (isinstance(val, float) and np.isnan(val)) else f"{val}"
                self.parsed_table.setItem(r, c, QTableWidgetItem(txt))

    def _fill_column_picker(self, df):
        prev_checked = {self.col_list.item(i).text(): self.col_list.item(i).checkState() == Qt.Checked
                        for i in range(self.col_list.count())}
        self.col_list.clear()
        for col in df.columns:
            name = str(col)
            it = QListWidgetItem(name)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            # Default-check numeric columns that aren't obviously a time / index column.
            numeric = np.issubdtype(df[col].dtype, np.number)
            looks_like_time = name.lower() in {"time", "t", "timestamp", "index", "idx"}
            default = numeric and not looks_like_time
            checked = prev_checked.get(name, Qt.Checked if default else Qt.Unchecked)
            it.setCheckState(Qt.Checked if checked == Qt.Checked else Qt.Unchecked)
            self.col_list.addItem(it)

    def _fill_time_col(self, df):
        prev = self.time_col.currentText()
        self.time_col.blockSignals(True)
        self.time_col.clear()
        self.time_col.addItem("(none — use bin size below)")
        for col in df.columns:
            self.time_col.addItem(str(col))
        # Auto-pick a column named time / t / timestamp.
        for i, col in enumerate(df.columns, start=1):
            if str(col).lower() in {"time", "t", "timestamp"} and np.issubdtype(df[col].dtype, np.number):
                self.time_col.setCurrentIndex(i)
                break
        if prev:
            ix = self.time_col.findText(prev)
            if ix >= 0:
                self.time_col.setCurrentIndex(ix)
        self.time_col.blockSignals(False)
        self._time_col_changed()

    def _time_col_changed(self):
        if self._df is None: return
        idx = self.time_col.currentIndex()
        if idx <= 0:
            return
        col = self._df.columns[idx - 1]
        t = self._df[col].dropna().to_numpy(dtype=float)
        if t.size >= 2:
            diffs = np.diff(t)
            diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
            if diffs.size:
                self.bin_size.setValue(float(np.median(diffs)))

    def _set_all_checked(self, on: bool):
        for i in range(self.col_list.count()):
            self.col_list.item(i).setCheckState(Qt.Checked if on else Qt.Unchecked)

    def _auto_time_range(self):
        cols = self._selected_columns()
        if self._df is None or not cols:
            return
        all_vals = []
        for c in cols:
            v = self._df[c].dropna().to_numpy(dtype=float, na_value=np.nan)
            v = v[np.isfinite(v)]
            if v.size:
                all_vals.append(v)
        if not all_vals:
            return
        full = np.concatenate(all_vals)
        self.t_start.setValue(float(full.min()))
        self.t_end.setValue(float(full.max()))

    def _mode_changed(self):
        mode = self.mode.currentIndex()
        # 0 continuous, 1 spike times, 2 counts matrix
        is_continuous = mode == 0
        is_spike = mode == 1
        is_counts = mode == 2
        self.time_col.parentWidget().setEnabled(True)
        for w in (self.t_start, self.t_end, self.btn_autorange):
            w.setEnabled(is_spike)
        self.time_col.setEnabled(is_continuous)
        # bin size is meaningful in every mode

    def _selected_columns(self) -> list[str]:
        cols = []
        for i in range(self.col_list.count()):
            it = self.col_list.item(i)
            if it.checkState() == Qt.Checked:
                cols.append(it.text())
        return cols

    # ------------------------------------------------------------------
    def _on_accept(self):
        if self._df is None:
            QMessageBox.warning(self, "Nothing to import", "Parse the file first.")
            return
        cols = self._selected_columns()
        if not cols:
            QMessageBox.warning(self, "No columns",
                                "Tick at least one column to import.")
            return
        df = self._df
        # Validate column types.
        non_numeric = [c for c in cols if not np.issubdtype(df[c].dtype, np.number)]
        if non_numeric:
            QMessageBox.warning(self, "Non-numeric columns",
                                f"These columns aren't numeric and were skipped: {non_numeric}")
            cols = [c for c in cols if c not in non_numeric]
            if not cols: return

        mode = self.mode.currentIndex()
        name = self.dataset_name.currentText().strip() or self._path.stem
        bin_size = float(self.bin_size.value())

        try:
            if mode == 0:        # Continuous channels
                signals = []
                for c in cols:
                    v = df[c].to_numpy(dtype=float)
                    # Forward-fill simple gaps so the model gets a regular array.
                    if np.any(~np.isfinite(v)):
                        ok = np.isfinite(v)
                        if not ok.any():
                            raise ValueError(f"Column '{c}' contains no finite values.")
                        v = v.copy()
                        # Replace NaN/Inf with the previous finite value or zero.
                        last = 0.0
                        for i in range(v.size):
                            if np.isfinite(v[i]):
                                last = v[i]
                            else:
                                v[i] = last
                    signals.append(v)
                arr = np.vstack(signals).astype(float)
                ds = dataio.Dataset(name=name, trials=[arr], kind="continuous",
                                    bin_size=bin_size, channel_labels=list(cols),
                                    source=str(self._path),
                                    sampling_rate=1.0 / bin_size)

            elif mode == 1:      # Spike times per column
                t0 = float(self.t_start.value()); t1 = float(self.t_end.value())
                if t1 <= t0:
                    raise ValueError("End time must be greater than start time.")
                edges = np.arange(t0, t1 + bin_size, bin_size, dtype=float)
                trains = []
                for c in cols:
                    v = df[c].to_numpy(dtype=float)
                    v = v[np.isfinite(v)]
                    trains.append(v)
                counts = np.zeros((len(cols), len(edges) - 1), dtype=float)
                for i, train in enumerate(trains):
                    counts[i] = np.histogram(train, bins=edges)[0]
                ds = dataio.Dataset(name=name, trials=[counts], kind="counts",
                                    bin_size=bin_size, channel_labels=list(cols),
                                    source=str(self._path))

            else:                # Counts matrix
                arr = df[cols].to_numpy(dtype=float)
                # rows = bins, columns = channels  ->  transpose to (channels, bins)
                counts = np.nan_to_num(arr, nan=0.0).T
                counts = np.round(counts).clip(min=0)
                ds = dataio.Dataset(name=name, trials=[counts], kind="counts",
                                    bin_size=bin_size, channel_labels=list(cols),
                                    source=str(self._path))
        except Exception as exc:
            QMessageBox.critical(self, "Import error", str(exc))
            return

        self._dataset = ds
        self.accept()

    @property
    def dataset(self) -> Optional[dataio.Dataset]:
        return self._dataset


# ----------------------------------------------------------------------------
def run_import(path: str | Path, parent: Optional[QWidget] = None
               ) -> Optional[dataio.Dataset]:
    dlg = CsvImportDialog(path, parent=parent)
    if dlg.exec() == QDialog.Accepted:
        return dlg.dataset
    return None
