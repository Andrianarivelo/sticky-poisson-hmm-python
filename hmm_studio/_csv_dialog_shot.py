"""Render the CSV import dialog to PNG (offscreen) for visual review."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "sticky-poisson-hmm-python"))

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

dlg = CsvImportDialog(Path(r"C:\Analysis\trial_0014_AIN01.csv"))
dlg.resize(1240, 820)
dlg.show()
app.processEvents()
app.processEvents()
out = HERE / "_screenshots" / "10_csv_import_continuous.png"
dlg.grab().save(str(out), "PNG")
print("saved", out)

# Switch to spike-time mode to capture that variant too.
dlg.mode.setCurrentIndex(1)
app.processEvents()
out2 = HERE / "_screenshots" / "11_csv_import_spike_times.png"
dlg.grab().save(str(out2), "PNG")
print("saved", out2)
