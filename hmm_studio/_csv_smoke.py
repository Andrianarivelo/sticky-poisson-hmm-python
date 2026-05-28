"""Headless test of the CSV import dialog logic on the user's photometry file."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "sticky-poisson-hmm-python"))

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication

from hmm_studio import theme
from hmm_studio.csv_import import CsvImportDialog


app = QApplication.instance() or QApplication([])
for fn in ("segoeui.ttf", "arial.ttf"):
    p = Path(r"C:\Windows\Fonts") / fn
    if p.exists():
        QFontDatabase.addApplicationFont(str(p))
app.setFont(QFont("Segoe UI", 9))
app.setStyleSheet(theme.stylesheet())
theme.configure_pyqtgraph()

src = Path(r"C:\Analysis\trial_0014_AIN01.csv")
assert src.exists(), src
dlg = CsvImportDialog(src)

# Verify the auto-detection picked the right metadata-skip count.
print(f"auto skip_rows = {dlg.skip_rows.value()}  (expected 3)")
print(f"auto delim     = {dlg._delim_value()!r}")
print(f"header on      = {dlg.has_header.isChecked()}")
print(f"df shape       = {None if dlg._df is None else dlg._df.shape}")
print(f"columns        = {[] if dlg._df is None else list(dlg._df.columns)}")
print(f"col picker     = "
      f"{[(dlg.col_list.item(i).text(), dlg.col_list.item(i).checkState() == Qt.Checked) for i in range(dlg.col_list.count())]}")
print(f"time_col combo = {[dlg.time_col.itemText(i) for i in range(dlg.time_col.count())]}")
print(f"time_col idx   = {dlg.time_col.currentIndex()}  -> "
      f"{dlg.time_col.currentText()!r}")
print(f"inferred dt    = {dlg.bin_size.value():.6g} s")

# Simulate: untick 'raw' and 'isobestic', keep only 'dFF', mode = Continuous,
# and click "Import".
for i in range(dlg.col_list.count()):
    it = dlg.col_list.item(i)
    it.setCheckState(Qt.Checked if it.text() == "dFF" else Qt.Unchecked)

dlg.mode.setCurrentIndex(0)
dlg.dataset_name.setEditText("trial_0014_AIN01_dFF")
dlg._on_accept()
ds = dlg.dataset
assert ds is not None, "Import did not produce a dataset"
print(f"\nDataset built:")
print(f"  name        = {ds.name}")
print(f"  kind        = {ds.kind}")
print(f"  bin_size    = {ds.bin_size:.6g} s   (sampling = {1.0/ds.bin_size:.2f} Hz)")
print(f"  n_channels  = {ds.n_channels}")
print(f"  channel lbl = {ds.channel_labels}")
print(f"  n_bins      = {ds.n_bins}")
print(f"  duration    = {ds.duration_s:.2f} s")
print(f"  trial shape = {ds.trials[0].shape}")
print(f"  first 5     = {ds.trials[0][0, :5]}")

# Now try a fit through the registry to prove the dataset is usable.
from hmm_studio.models import REGISTRY
spec = REGISTRY["sticky_gaussian"]
prep = spec.prepare(ds.trials, ds.kind, ds.bin_size, np.random.default_rng(0))
res = spec.fit(prep, n_states=3, params={"threshold": 0.85, "max_iter": 80}, random_state=2)
print(f"\nsticky-Gaussian fit:")
print(f"  log_likelihood     = {res.log_likelihood:.2f}")
print(f"  n_iter             = {res.n_iter}")
print(f"  converged          = {res.converged}")
print(f"  threshold_ok       = {res.threshold_satisfied}")
print(f"  means.shape        = {res.means.shape}")

print("\nAll checks passed.")
