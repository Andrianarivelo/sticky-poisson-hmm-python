"""Visual theme: an elegant dark palette, Qt stylesheet, and state colors.

The state palette is taken from the sticky Poisson HMM paper's MATLAB demo
(via ``hmm_spikes.plotting.default_state_colors``) so the decoding figures
match the publication's look.
"""

from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------------
# Core palette (deep slate, single warm-cyan accent)
# ----------------------------------------------------------------------------
BG_0 = "#0b0f17"   # window background (deepest)
BG_1 = "#111726"   # panels / docks
BG_2 = "#18213440" # subtle elevated fill
PANEL = "#141b2b"  # cards
BORDER = "#26324a"
GRID = "#1d2740"
TEXT = "#e6ecf5"
TEXT_DIM = "#9fb0c8"
TEXT_FAINT = "#64748b"
ACCENT = "#36d6c3"      # primary accent (teal)
ACCENT_DK = "#1f9b8e"
ACCENT_2 = "#7c5cff"    # secondary accent (violet)
GOOD = "#4ade80"
WARN = "#fbbf24"
BAD = "#f87171"

# Paper state palette, normalized 0-255 RGB, extended with a tasteful set.
_PAPER = np.array(
    [
        [0.9961, 0.6445, 0.0000],
        [0.8500, 0.3250, 0.0980],
        [0.0000, 0.0000, 1.0000],
        [0.1198, 0.3620, 0.2266],
        [0.1172, 0.5625, 0.9961],
        [0.8000, 0.0000, 0.0000],
        [0.7520, 0.8750, 0.0000],
        [0.7194, 0.5208, 0.7194],
    ],
    dtype=float,
)


def state_colors(n_states: int) -> np.ndarray:
    """Return ``(n_states, 3)`` float RGB colors in 0-1, matching the paper."""

    if n_states <= len(_PAPER):
        return _PAPER[:n_states].copy()
    # Extend with a perceptually spaced HSV ramp for large state counts.
    import colorsys

    extra = []
    for i in range(n_states - len(_PAPER)):
        h = (0.58 + 0.61803398875 * (i + 1)) % 1.0
        extra.append(colorsys.hsv_to_rgb(h, 0.62, 0.95))
    return np.vstack([_PAPER, np.array(extra, dtype=float)])


def state_qcolor(index: int, n_states: int, alpha: int = 255):
    from PySide6.QtGui import QColor

    rgb = state_colors(max(n_states, index + 1))[index]
    return QColor(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255), alpha)


def state_hex(index: int, n_states: int) -> str:
    rgb = state_colors(max(n_states, index + 1))[index]
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))


FONT_FAMILY = "Segoe UI, Inter, system-ui, sans-serif"
MONO_FAMILY = "Cascadia Code, Consolas, monospace"


def stylesheet() -> str:
    """Return the global Qt stylesheet."""

    return f"""
    * {{
        font-family: {FONT_FAMILY};
        font-size: 13px;
        color: {TEXT};
        outline: none;
    }}
    QMainWindow, QWidget {{ background: {BG_0}; }}

    QToolBar {{
        background: {BG_1};
        border: none;
        border-bottom: 1px solid {BORDER};
        spacing: 8px;
        padding: 6px 10px;
    }}
    QToolBar QToolButton {{
        background: transparent;
        color: {TEXT_DIM};
        padding: 6px 12px;
        border-radius: 8px;
        font-weight: 600;
    }}
    QToolBar QToolButton:hover {{ background: {PANEL}; color: {TEXT}; }}
    QToolBar QToolButton:pressed {{ background: {ACCENT_DK}; color: white; }}

    QStatusBar {{
        background: {BG_1};
        border-top: 1px solid {BORDER};
        color: {TEXT_DIM};
    }}
    QStatusBar QLabel {{ color: {TEXT_DIM}; }}

    QDockWidget {{
        titlebar-close-icon: none;
        color: {TEXT_DIM};
        font-weight: 700;
    }}
    QDockWidget::title {{
        background: {BG_1};
        padding: 8px 12px;
        border-bottom: 1px solid {BORDER};
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 11px;
    }}

    QScrollArea {{ border: none; background: transparent; }}

    /* Cards / group boxes */
    QGroupBox {{
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 12px;
        margin-top: 18px;
        padding: 14px 12px 12px 12px;
        font-weight: 700;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 14px;
        top: 2px;
        padding: 0 6px;
        color: {ACCENT};
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-size: 11px;
    }}

    QLabel {{ background: transparent; }}
    QLabel#hint {{ color: {TEXT_FAINT}; font-size: 11px; }}
    QLabel#metricValue {{ color: {TEXT}; font-size: 22px; font-weight: 800; }}
    QLabel#metricLabel {{ color: {TEXT_FAINT}; font-size: 10px; text-transform: uppercase; letter-spacing: 1px; }}
    QLabel#h1 {{ color: {TEXT}; font-size: 18px; font-weight: 800; }}

    /* Buttons */
    QPushButton {{
        background: {BG_2};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 8px 14px;
        color: {TEXT};
        font-weight: 600;
    }}
    QPushButton:hover {{ border-color: {ACCENT_DK}; background: #1b2438; }}
    QPushButton:pressed {{ background: {ACCENT_DK}; }}
    QPushButton:disabled {{ color: {TEXT_FAINT}; border-color: {GRID}; background: transparent; }}
    QPushButton#primary {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {ACCENT}, stop:1 {ACCENT_DK});
        color: #04211d; border: none; font-weight: 800;
    }}
    QPushButton#primary:hover {{ background: {ACCENT}; }}
    QPushButton#primary:disabled {{ background: {GRID}; color: {TEXT_FAINT}; }}
    QPushButton#ghost {{ background: transparent; border: 1px solid {BORDER}; }}

    /* Inputs */
    QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
        background: {BG_0};
        border: 1px solid {BORDER};
        border-radius: 7px;
        padding: 6px 8px;
        selection-background-color: {ACCENT_DK};
    }}
    QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {{ border-color: {ACCENT_DK}; }}
    QComboBox::drop-down {{ border: none; width: 22px; }}
    QComboBox::down-arrow {{
        image: none; border-left: 4px solid transparent; border-right: 4px solid transparent;
        border-top: 5px solid {TEXT_DIM}; margin-right: 8px;
    }}
    QComboBox QAbstractItemView {{
        background: {PANEL}; border: 1px solid {BORDER};
        selection-background-color: {ACCENT_DK}; selection-color: white;
        padding: 4px; border-radius: 8px;
    }}
    QSpinBox::up-button, QDoubleSpinBox::up-button,
    QSpinBox::down-button, QDoubleSpinBox::down-button {{ width: 16px; border: none; background: {BG_2}; }}

    QCheckBox {{ spacing: 8px; }}
    QCheckBox::indicator {{
        width: 18px; height: 18px; border-radius: 5px;
        border: 1px solid {BORDER}; background: {BG_0};
    }}
    QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}

    /* Tabs */
    QTabWidget::pane {{ border: none; background: {BG_0}; top: -1px; }}
    QTabBar::tab {{
        background: transparent; color: {TEXT_FAINT};
        padding: 9px 18px; margin-right: 4px;
        border-bottom: 2px solid transparent; font-weight: 700;
    }}
    QTabBar::tab:hover {{ color: {TEXT_DIM}; }}
    QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}

    /* Tables / lists */
    QTableWidget, QListWidget, QTreeWidget {{
        background: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px;
        gridline-color: {GRID}; alternate-background-color: #121a2a;
    }}
    QHeaderView::section {{
        background: {BG_1}; color: {TEXT_DIM}; border: none;
        border-bottom: 1px solid {BORDER}; padding: 7px 8px; font-weight: 700;
    }}
    QTableWidget::item:selected, QListWidget::item:selected {{
        background: {ACCENT_DK}; color: white;
    }}
    QListWidget::item {{ padding: 2px; border-radius: 8px; }}

    /* Progress bar */
    QProgressBar {{
        background: {BG_0}; border: 1px solid {BORDER}; border-radius: 7px;
        text-align: center; color: {TEXT_DIM}; height: 16px;
    }}
    QProgressBar::chunk {{
        border-radius: 6px;
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {ACCENT_DK}, stop:1 {ACCENT});
    }}

    /* Scrollbars */
    QScrollBar:vertical {{ background: transparent; width: 11px; margin: 2px; }}
    QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 5px; min-height: 30px; }}
    QScrollBar::handle:vertical:hover {{ background: {ACCENT_DK}; }}
    QScrollBar:horizontal {{ background: transparent; height: 11px; margin: 2px; }}
    QScrollBar::handle:horizontal {{ background: {BORDER}; border-radius: 5px; min-width: 30px; }}
    QScrollBar::handle:horizontal:hover {{ background: {ACCENT_DK}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

    QToolTip {{
        background: {PANEL}; color: {TEXT}; border: 1px solid {ACCENT_DK};
        border-radius: 6px; padding: 6px;
    }}
    QSplitter::handle {{ background: {BORDER}; }}
    QMenu {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 8px; padding: 6px; }}
    QMenu::item {{ padding: 6px 24px; border-radius: 6px; }}
    QMenu::item:selected {{ background: {ACCENT_DK}; }}
    """


def configure_pyqtgraph() -> None:
    """Apply global pyqtgraph styling consistent with the dark theme."""

    import pyqtgraph as pg

    pg.setConfigOption("background", BG_0)
    pg.setConfigOption("foreground", TEXT_DIM)
    pg.setConfigOptions(antialias=True, useOpenGL=False, imageAxisOrder="row-major")
