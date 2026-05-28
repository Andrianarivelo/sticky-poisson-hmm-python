"""Visualization views: fast pyqtgraph widgets for the central tab area.

Every view is a self-contained :class:`QWidget` exposing a small API:

* ``update_dataset(dataset, trial_idx)``      – refresh on new raw data,
* ``update_run(run, dataset, trial_idx)``     – refresh on a new fit,
* ``update_bic(bic_result)``                  – BIC view only,
* ``update_runs(runs)``                       – comparison view only.

The signature paper figure (raster + colored state spans + posterior traces) is
rendered via a single :class:`pg.ImageItem` for the state-span background. That
stays fast even for very long recordings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout, QHeaderView,
    QLabel, QPushButton, QSizePolicy, QSplitter, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from . import theme
from .models import REGISTRY


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _state_rgba(n_states: int, alpha: int = 110) -> np.ndarray:
    rgb = theme.state_colors(n_states)
    rgba = np.zeros((n_states, 4), dtype=np.uint8)
    rgba[:, :3] = (rgb * 255).astype(np.uint8)
    rgba[:, 3] = alpha
    return rgba


def _state_image_1d(labels: np.ndarray, valid: np.ndarray, n_states: int,
                    alpha: int = 110) -> np.ndarray:
    """Make a (1, T, 4) RGBA strip colored by state, transparent where invalid."""
    rgba = _state_rgba(n_states, alpha=alpha)
    img = rgba[np.clip(labels, 0, n_states - 1)]
    img[~valid] = (0, 0, 0, 0)
    return img[None, :, :]


def _state_image_2d(label_mat: np.ndarray, valid_mat: np.ndarray, n_states: int,
                    alpha: int = 220) -> np.ndarray:
    rgba = _state_rgba(n_states, alpha=alpha)
    img = rgba[np.clip(label_mat, 0, n_states - 1)]
    img[~valid_mat] = (0, 0, 0, 0)
    return img


def _metric_card(label: str, value: str, big: bool = True) -> QFrame:
    card = QFrame()
    card.setStyleSheet(f"background:{theme.PANEL}; border:1px solid {theme.BORDER}; "
                       f"border-radius:10px;")
    lay = QVBoxLayout(card); lay.setContentsMargins(14, 10, 14, 10); lay.setSpacing(2)
    l1 = QLabel(value); l1.setObjectName("metricValue" if big else "h1")
    l2 = QLabel(label.upper()); l2.setObjectName("metricLabel")
    lay.addWidget(l1); lay.addWidget(l2)
    return card


def _plot(layout: pg.GraphicsLayoutWidget, row: int, col: int, title: str = "",
          y_label: str = "", x_label: str = "", colspan: int = 1) -> pg.PlotItem:
    p = layout.addPlot(row=row, col=col, colspan=colspan)
    p.showAxis("right", False); p.showAxis("top", False)
    p.setMenuEnabled(False)
    p.showGrid(x=True, y=True, alpha=0.15)
    p.getViewBox().setBackgroundColor(theme.BG_0)
    p.getAxis("left").setPen(theme.BORDER); p.getAxis("bottom").setPen(theme.BORDER)
    p.getAxis("left").setTextPen(theme.TEXT_DIM); p.getAxis("bottom").setTextPen(theme.TEXT_DIM)
    if title:
        p.setTitle(title, color=theme.TEXT, size="11pt")
    if y_label:
        p.setLabel("left", y_label, color=theme.TEXT_DIM)
    if x_label:
        p.setLabel("bottom", x_label, color=theme.TEXT_DIM)
    return p


def _legend(plot: pg.PlotItem, items: list[tuple[str, QColor]],
            corner: str = "topright"):
    """Compact, translucent legend anchored to a chosen corner of the view box.

    corner: one of 'topright', 'topleft', 'bottomright', 'bottomleft'.
    """
    brush = pg.mkBrush(15, 22, 38, 200)
    pen = pg.mkPen(theme.BORDER)
    leg = pg.LegendItem(brush=brush, pen=pen,
                        labelTextSize="8pt", labelTextColor=theme.TEXT_DIM)
    leg.setParentItem(plot.vb)
    anchors = {
        "topright":    ((1, 0), (1, 0), (-8, 8)),
        "topleft":     ((0, 0), (0, 0), (8, 8)),
        "bottomright": ((1, 1), (1, 1), (-8, -8)),
        "bottomleft":  ((0, 1), (0, 1), (8, -8)),
    }
    item_a, parent_a, offset = anchors.get(corner, anchors["topright"])
    leg.anchor(itemPos=item_a, parentPos=parent_a, offset=offset)
    for label, color in items:
        dummy = pg.PlotDataItem(pen=pg.mkPen(color, width=2))
        leg.addItem(dummy, label)
    return leg


# ----------------------------------------------------------------------------
# Decoded cache
# ----------------------------------------------------------------------------
@dataclass
class TrialDecode:
    posterior: np.ndarray
    viterbi: np.ndarray
    labels: np.ndarray
    maxprob: np.ndarray


def decode_trial(run, trial_idx: int) -> TrialDecode:
    spec = REGISTRY[run.method_key]
    x = run.prepared_trials[trial_idx]
    post, vit = spec.decode(run.result, x)
    labels = post.argmax(axis=0)
    maxprob = post.max(axis=0)
    return TrialDecode(post, vit, labels, maxprob)


# ----------------------------------------------------------------------------
# Empty-state widget
# ----------------------------------------------------------------------------
class EmptyState(QWidget):
    def __init__(self, title: str, hint: str, glyph: str = "◌"):
        super().__init__()
        v = QVBoxLayout(self); v.setAlignment(Qt.AlignCenter); v.setSpacing(14)
        g = QLabel(glyph); g.setAlignment(Qt.AlignCenter)
        g.setStyleSheet(f"color:{theme.ACCENT_DK}; font-size:72px; font-weight:300;")
        t = QLabel(title); t.setObjectName("h1"); t.setAlignment(Qt.AlignCenter)
        h = QLabel(hint); h.setObjectName("hint"); h.setAlignment(Qt.AlignCenter)
        h.setWordWrap(True); h.setMaximumWidth(560)
        v.addStretch(); v.addWidget(g); v.addWidget(t); v.addWidget(h); v.addStretch()


# ----------------------------------------------------------------------------
# Data view: raw data (raster or traces) + summary metrics
# ----------------------------------------------------------------------------
class DataView(QWidget):
    def __init__(self):
        super().__init__()
        root = QVBoxLayout(self); root.setContentsMargins(12, 12, 12, 12); root.setSpacing(10)
        # Header row of metric cards
        self._metrics_row = QHBoxLayout(); self._metrics_row.setSpacing(10)
        root.addLayout(self._metrics_row)
        # Plot
        self.gw = pg.GraphicsLayoutWidget(); self.gw.setBackground(theme.BG_0)
        root.addWidget(self.gw, 1)
        self._dataset = None

    def _clear_metrics(self):
        while self._metrics_row.count():
            w = self._metrics_row.takeAt(0).widget()
            if w: w.deleteLater()

    def update_dataset(self, dataset, trial_idx: int = 0):
        self._dataset = dataset
        self._clear_metrics()
        if dataset is None:
            self.gw.clear()
            return
        # Metric cards
        cards = [
            ("Trials", str(dataset.n_trials)),
            ("Channels", str(dataset.n_channels)),
            ("Time bins", str(dataset.n_bins)),
            ("Δt", f"{dataset.bin_size*1000:.1f} ms"),
            ("Duration", f"{dataset.duration_s:.1f} s"),
            ("Kind", dataset.kind.title()),
        ]
        for label, value in cards:
            self._metrics_row.addWidget(_metric_card(label, value))
        self._metrics_row.addStretch()

        # Render the selected trial
        self.gw.clear()
        trial = dataset.trials[trial_idx]
        t = dataset.time_axis(trial_idx)
        title = f"{dataset.name}   ·   Trial {trial_idx+1} / {dataset.n_trials}"
        if dataset.kind == "counts":
            self._render_raster(trial, t, title)
        else:
            self._render_traces(trial, t, title, dataset.channel_labels)

    def _render_raster(self, counts: np.ndarray, t: np.ndarray, title: str):
        n_units, n_bins = counts.shape
        p = _plot(self.gw, 0, 0, title=title, y_label="Channel index", x_label="time (s)")
        p.setLimits(yMin=-0.5, yMax=n_units + 0.5)
        # Build sparse spike points: one point per (channel, bin) with count > 0,
        # repeated for high-count bins to keep visual weight.
        rows, cols = np.nonzero(counts > 0)
        if rows.size == 0:
            return
        weights = counts[rows, cols].astype(int)
        x = np.repeat(t[cols], np.maximum(weights, 1)) + (np.random.default_rng(0).uniform(
            -0.45, 0.45, size=int(weights.sum())) * (t[1] - t[0] if len(t) > 1 else 0.0))
        y = np.repeat(rows + 1, np.maximum(weights, 1))
        scatter = pg.ScatterPlotItem(
            x=x, y=y, size=3.5, pen=None,
            brush=pg.mkBrush(QColor(theme.TEXT))
        )
        p.addItem(scatter)
        p.setXRange(float(t[0]), float(t[-1]) if t.size > 1 else 1.0, padding=0)

    def _render_traces(self, signals: np.ndarray, t: np.ndarray, title: str,
                       labels: list[str]):
        n_chan, n_bins = signals.shape
        p = _plot(self.gw, 0, 0, title=title, y_label="signal (offset)", x_label="time (s)")
        spread = np.percentile(np.abs(signals), 99) * 2.2 or 1.0
        for ch in range(n_chan):
            color = QColor(theme.ACCENT) if ch == 0 else QColor(theme.TEXT_DIM)
            color.setAlpha(220 if ch < 4 else 160)
            offset = ch * spread
            line = pg.PlotDataItem(t, signals[ch] + offset,
                                   pen=pg.mkPen(color, width=1.2))
            p.addItem(line)
            txt = pg.TextItem(labels[ch] if ch < len(labels) else f"Ch {ch+1}",
                              color=theme.TEXT_DIM, anchor=(1, 0.5))
            txt.setPos(float(t[0]), offset)
            p.addItem(txt)
        p.setXRange(float(t[0]), float(t[-1]) if t.size > 1 else 1.0, padding=0)


# ----------------------------------------------------------------------------
# Decoding view: the signature figure
# ----------------------------------------------------------------------------
class DecodingView(QWidget):
    request_export = Signal()

    def __init__(self):
        super().__init__()
        root = QVBoxLayout(self); root.setContentsMargins(12, 12, 12, 12); root.setSpacing(8)
        # toolbar row
        bar = QHBoxLayout()
        self.mode = QComboBox()
        self.mode.addItems(["Posterior (max a posteriori)", "Viterbi"])
        self.mode.setToolTip(
            "Posterior: colour a bin only if max posterior probability ≥ confidence.\n"
            "Viterbi: colour every bin by the most likely full state sequence."
        )
        self.threshold = pg.SpinBox(value=0.8, bounds=(0.0, 1.0), step=0.05, decimals=2)
        self.threshold.setMaximumWidth(110)
        self.threshold.setToolTip("Minimum posterior probability required to colour a bin.")
        lbl = QLabel("Confidence ≥"); lbl.setObjectName("hint")
        # Percentage opacity for friendliness (internally maps to 0-255).
        self.alpha = pg.SpinBox(value=43, bounds=(0, 100), step=5, decimals=0, suffix=" %")
        self.alpha.setMaximumWidth(90)
        self.alpha.setToolTip("Transparency of the state-coloured background behind the data.")
        bar.addWidget(QLabel("Segmentation:")); bar.addWidget(self.mode)
        bar.addSpacing(20); bar.addWidget(lbl); bar.addWidget(self.threshold)
        bar.addSpacing(20); bar.addWidget(QLabel("Span opacity:")); bar.addWidget(self.alpha)
        bar.addStretch()
        root.addLayout(bar)

        self.gw = pg.GraphicsLayoutWidget(); self.gw.setBackground(theme.BG_0)
        root.addWidget(self.gw, 1)

        self.mode.currentIndexChanged.connect(self._redraw)
        self.threshold.sigValueChanged.connect(self._redraw)
        self.alpha.sigValueChanged.connect(self._redraw)

        self._dataset = None
        self._run = None
        self._trial_idx = 0
        self._decode_cache: TrialDecode | None = None

    def update_dataset(self, dataset, trial_idx: int = 0):
        self._dataset = dataset
        self._trial_idx = trial_idx
        self.gw.clear()

    def update_run(self, run, dataset, trial_idx: int = 0):
        self._dataset = dataset
        self._run = run
        self._trial_idx = trial_idx
        self._decode_cache = decode_trial(run, trial_idx)
        self._redraw()

    def _redraw(self):
        if self._dataset is None or self._run is None or self._decode_cache is None:
            return
        self.gw.clear()
        ds, run, ti = self._dataset, self._run, self._trial_idx
        td = self._decode_cache
        n_states = run.n_states
        if self.mode.currentIndex() == 0:
            labels = td.labels
            valid = td.maxprob >= float(self.threshold.value())
        else:
            labels = td.viterbi
            valid = np.ones_like(labels, dtype=bool)
        alpha = int(round(float(self.alpha.value()) * 255 / 100.0))

        t = ds.time_axis(ti)
        x0, x1 = float(t[0]), float(t[-1]) if t.size > 1 else 1.0

        # Top: raster or traces, with state-span background
        if ds.kind == "counts":
            top = _plot(self.gw, 0, 0,
                        title=f"{run.name}   ·   Trial {ti+1}",
                        y_label="Channel index")
        else:
            top = _plot(self.gw, 0, 0,
                        title=f"{run.name}   ·   Trial {ti+1}",
                        y_label="signal")

        top.getAxis("bottom").setStyle(showValues=False)
        top.setMouseEnabled(x=True, y=False)
        n_chan = ds.trials[ti].shape[0]
        if ds.kind == "counts":
            y_min = -0.5
            y_max = n_chan + 0.5
        else:
            signals = np.asarray(ds.trials[ti], dtype=float)
            spread = float(np.nanpercentile(np.abs(signals), 99)) * 2.2 or 1.0
            offsets = np.arange(n_chan, dtype=float)[:, None] * spread
            stacked = signals + offsets
            finite = stacked[np.isfinite(stacked)]
            if finite.size:
                y_min = float(finite.min())
                y_max = float(finite.max())
            else:
                y_min, y_max = -spread, spread
            headroom = max((y_max - y_min) * 0.04, spread * 0.08, 1e-9)
            y_min -= headroom
            y_max += headroom

        # State-span background image. It is stretched over the actual visible
        # y-range, which keeps continuous traces with large baselines readable.
        img = _state_image_1d(labels, valid, n_states, alpha=alpha)
        bg = pg.ImageItem(img)
        # ImageItem coordinates: rect (x, y, w, h)
        from PySide6.QtCore import QRectF
        bg.setRect(QRectF(x0, y_min, (x1 - x0) if x1 > x0 else 1.0, max(y_max - y_min, 1.0)))
        bg.setZValue(-10)
        top.addItem(bg)

        # Data overlay
        if ds.kind == "counts":
            counts = ds.trials[ti]
            rows, cols = np.nonzero(counts > 0)
            if rows.size:
                weights = counts[rows, cols].astype(int)
                rng = np.random.default_rng(0)
                xs = np.repeat(t[cols], np.maximum(weights, 1))
                xs = xs + rng.uniform(-0.4, 0.4, size=xs.size) * (t[1] - t[0] if t.size > 1 else 0.0)
                ys = np.repeat(rows + 1, np.maximum(weights, 1))
                top.addItem(pg.ScatterPlotItem(x=xs, y=ys, size=3.5, pen=None,
                                               brush=pg.mkBrush(QColor(theme.TEXT))))
            top.setLimits(yMin=y_min, yMax=y_max)
            top.setYRange(y_min, y_max, padding=0)
        else:
            signals = ds.trials[ti]
            for ch in range(n_chan):
                color = QColor(theme.TEXT); color.setAlpha(220)
                top.addItem(pg.PlotDataItem(t, signals[ch] + ch * spread,
                                            pen=pg.mkPen(color, width=1.1)))
            top.setLimits(yMin=y_min, yMax=y_max)
            top.setYRange(y_min, y_max, padding=0)
        top.setXRange(x0, x1, padding=0)

        # Bottom: posterior probability traces
        bot = _plot(self.gw, 1, 0, y_label="state probability", x_label="time (s)")
        bot.setMouseEnabled(x=True, y=False)
        bot.setYRange(0.0, 1.02, padding=0)
        for k in range(n_states):
            color = theme.state_qcolor(k, n_states)
            bot.addItem(pg.PlotDataItem(t, td.posterior[k],
                                        pen=pg.mkPen(color, width=1.6),
                                        name=f"State {k+1}"))
        # threshold line
        bot.addItem(pg.InfiniteLine(pos=float(self.threshold.value()), angle=0,
                                    pen=pg.mkPen(theme.TEXT_FAINT, style=Qt.DashLine)))
        bot.setXRange(x0, x1, padding=0)
        bot.setXLink(top)

        # Layout proportions: top ~2/3, bottom ~1/3
        layout = self.gw.ci.layout
        layout.setRowStretchFactor(0, 3)
        layout.setRowStretchFactor(1, 2)

        # Legend
        _legend(bot, [(f"State {k+1}", theme.state_qcolor(k, n_states))
                      for k in range(n_states)], corner="topright")


# ----------------------------------------------------------------------------
# States view: emission profile + transition matrix + stationary distribution
# ----------------------------------------------------------------------------
class StatesView(QWidget):
    def __init__(self):
        super().__init__()
        root = QVBoxLayout(self); root.setContentsMargins(12, 12, 12, 12); root.setSpacing(8)
        self.gw = pg.GraphicsLayoutWidget(); self.gw.setBackground(theme.BG_0)
        root.addWidget(self.gw, 1)

    def update_run(self, run, dataset, trial_idx: int = 0):
        self.gw.clear()
        spec = REGISTRY[run.method_key]
        profile, y_label = spec.profile(run.result, run.bin_size)
        n_states = run.n_states
        n_units = profile.shape[0]

        # Top row: per-state emission profile as a heatmap
        heat = _plot(self.gw, 0, 0, title="State emission profile",
                     x_label="State", y_label=y_label)
        img = pg.ImageItem(profile.T[:, ::-1])  # rows=states, cols=units flipped for display
        # color map
        cm = pg.colormap.get("magma")
        bar = pg.ColorBarItem(values=(float(profile.min()), float(profile.max())),
                              colorMap=cm, label=y_label, width=15)
        bar.setImageItem(img, insert_in=heat)
        from PySide6.QtCore import QRectF
        img.setRect(QRectF(0.5, -0.5, n_states, n_units))
        heat.addItem(img)
        heat.getAxis("bottom").setTicks([[(i + 1, f"S{i+1}") for i in range(n_states)]])
        heat.setLimits(xMin=0.5, xMax=n_states + 0.5, yMin=-0.5, yMax=n_units - 0.5)
        heat.invertY(True)

        # Second column: per-state bar charts
        bars_plot = _plot(self.gw, 0, 1, title="Per-state profile",
                          x_label="Channel index", y_label=y_label)
        width = 0.8 / max(n_states, 1)
        x_idx = np.arange(n_units) + 1
        for k in range(n_states):
            color = theme.state_qcolor(k, n_states, alpha=220)
            bg = pg.BarGraphItem(x=x_idx - 0.4 + (k + 0.5) * width,
                                 height=profile[:, k], width=width * 0.95,
                                 brush=pg.mkBrush(color), pen=pg.mkPen(color))
            bars_plot.addItem(bg)
        # Add headroom so the legend doesn't sit over the highest bars.
        max_val = float(profile.max()) if profile.size else 1.0
        bars_plot.setYRange(0, max_val * 1.22, padding=0)
        _legend(bars_plot, [(f"State {k+1}", theme.state_qcolor(k, n_states))
                            for k in range(n_states)], corner="topright")

        # Bottom row: transition matrix + stationary distribution
        trans_plot = _plot(self.gw, 1, 0, title="Transition matrix Γ",
                           x_label="to state", y_label="from state")
        gamma = np.asarray(run.result.gamma)
        gimg = pg.ImageItem(gamma)
        cm2 = pg.colormap.get("viridis")
        cbar = pg.ColorBarItem(values=(0.0, 1.0), colorMap=cm2, width=12)
        cbar.setImageItem(gimg, insert_in=trans_plot)
        from PySide6.QtCore import QRectF
        gimg.setRect(QRectF(0.5, 0.5, n_states, n_states))
        trans_plot.addItem(gimg)
        trans_plot.getAxis("bottom").setTicks([[(i + 1, f"S{i+1}") for i in range(n_states)]])
        trans_plot.getAxis("left").setTicks([[(i + 1, f"S{i+1}") for i in range(n_states)]])
        trans_plot.setLimits(xMin=0.5, xMax=n_states + 0.5, yMin=0.5, yMax=n_states + 0.5)
        trans_plot.invertY(True)
        # Annotate cells
        for i in range(n_states):
            for j in range(n_states):
                v = float(gamma[i, j])
                txt = pg.TextItem(f"{v:.2f}", anchor=(0.5, 0.5),
                                  color=theme.TEXT if v < 0.6 else "black")
                txt.setPos(j + 1, i + 1)
                trans_plot.addItem(txt)

        stat_plot = _plot(self.gw, 1, 1, title="Stationary distribution",
                          x_label="state", y_label="probability")
        try:
            import hmm_spikes as H
            stationary = H.stationary_distribution(gamma)
        except Exception:
            stationary = np.diag(gamma) / np.trace(gamma) if np.trace(gamma) > 0 else np.full(n_states, 1/n_states)
        colors = [theme.state_qcolor(k, n_states) for k in range(n_states)]
        for k in range(n_states):
            bar = pg.BarGraphItem(x=[k + 1], height=[float(stationary[k])], width=0.7,
                                  brush=pg.mkBrush(colors[k]), pen=pg.mkPen(colors[k]))
            stat_plot.addItem(bar)
        stat_plot.getAxis("bottom").setTicks([[(i + 1, f"S{i+1}") for i in range(n_states)]])
        stat_plot.setLimits(xMin=0.5, xMax=n_states + 0.5, yMin=0, yMax=1.02)


# ----------------------------------------------------------------------------
# Training view: log-likelihood + self-transition over EM iterations
# ----------------------------------------------------------------------------
class TrainingView(QWidget):
    def __init__(self):
        super().__init__()
        root = QVBoxLayout(self); root.setContentsMargins(12, 12, 12, 12); root.setSpacing(8)
        self.gw = pg.GraphicsLayoutWidget(); self.gw.setBackground(theme.BG_0)
        root.addWidget(self.gw, 1)

    def update_run(self, run, dataset=None, trial_idx: int = 0):
        self.gw.clear()
        history = getattr(run.result, "history", None)
        n_states = run.n_states

        # LL curve
        ll_plot = _plot(self.gw, 0, 0, title="Training log-likelihood",
                        x_label="EM iteration", y_label="log-likelihood")
        if history is not None and history.log_likelihood.size:
            x = np.arange(1, history.log_likelihood.size + 1)
            ll_plot.addItem(pg.PlotDataItem(x, history.log_likelihood,
                                            pen=pg.mkPen(theme.ACCENT, width=2.2)))
        ll_plot.addItem(pg.InfiniteLine(pos=run.log_likelihood, angle=0,
                                        pen=pg.mkPen(theme.TEXT_FAINT, style=Qt.DashLine)))

        # Self-transition trajectories
        st_plot = _plot(self.gw, 1, 0, title="Self-transition probability γ_kk",
                        x_label="EM iteration", y_label="γ_kk")
        st_plot.setYRange(0, 1.02, padding=0)
        if history is not None and history.gamma_diag.size:
            x = np.arange(history.gamma_diag.shape[1])
            for k in range(history.gamma_diag.shape[0]):
                color = theme.state_qcolor(k, n_states)
                st_plot.addItem(pg.PlotDataItem(x, history.gamma_diag[k],
                                                pen=pg.mkPen(color, width=2.0),
                                                name=f"State {k+1}"))
        if hasattr(run.result, "threshold") and run.result.threshold > 0:
            st_plot.addItem(pg.InfiniteLine(pos=float(run.result.threshold), angle=0,
                                            pen=pg.mkPen(theme.WARN, style=Qt.DashLine)))
        _legend(st_plot, [(f"State {k+1}", theme.state_qcolor(k, n_states))
                          for k in range(n_states)], corner="bottomright")

        layout = self.gw.ci.layout
        layout.setRowStretchFactor(0, 1)
        layout.setRowStretchFactor(1, 1)


# ----------------------------------------------------------------------------
# BIC view: BIC vs K curve + per-restart scatter + best-K markers
# ----------------------------------------------------------------------------
class BICView(QWidget):
    def __init__(self):
        super().__init__()
        root = QVBoxLayout(self); root.setContentsMargins(12, 12, 12, 12); root.setSpacing(8)
        info = QHBoxLayout()
        self.title = QLabel("Run a BIC scan to compare candidate state numbers.")
        self.title.setObjectName("h1")
        info.addWidget(self.title); info.addStretch()
        root.addLayout(info)

        self.gw = pg.GraphicsLayoutWidget(); self.gw.setBackground(theme.BG_0)
        root.addWidget(self.gw, 1)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels([
            "K", "BIC", "log-likelihood", "Converged", "θ-OK", "Iter", "Seed"
        ])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeToContents)
        hh.setStretchLastSection(False)
        hh.setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        root.addWidget(self.table)

    def update_bic(self, bic_result):
        self.gw.clear()
        self.table.setRowCount(0)
        if bic_result is None or not bic_result.by_k:
            self.title.setText("Run a BIC scan to compare candidate state numbers.")
            return
        ks = sorted(bic_result.by_k.keys())
        best_bics = np.array([bic_result.by_k[k]["bic"] for k in ks], dtype=float)
        best_lls = np.array([bic_result.by_k[k]["ll"] for k in ks], dtype=float)

        # BIC plot
        bic_plot = _plot(self.gw, 0, 0, title="BIC vs number of states",
                         x_label="K", y_label="BIC (lower is better)")
        bic_plot.addItem(pg.PlotDataItem(ks, best_bics,
                                         pen=pg.mkPen(theme.ACCENT, width=2.5),
                                         symbol="o", symbolSize=10,
                                         symbolBrush=pg.mkBrush(theme.ACCENT),
                                         symbolPen=pg.mkPen("white")))
        # All attempts scatter (faded)
        xs, ys = [], []
        for k in ks:
            for rec in bic_result.by_k[k]["attempts"]:
                if np.isfinite(rec.get("bic", float("inf"))):
                    xs.append(k); ys.append(rec["bic"])
        if xs:
            bic_plot.addItem(pg.ScatterPlotItem(x=xs, y=ys, size=6,
                                                pen=None,
                                                brush=pg.mkBrush(QColor(180, 200, 230, 90))))
        # Highlight best K
        if bic_result.best_k is not None:
            bic_plot.addItem(pg.InfiniteLine(pos=bic_result.best_k, angle=90,
                                             pen=pg.mkPen(theme.ACCENT_2, width=2,
                                                          style=Qt.DashLine),
                                             label=f"Best K = {bic_result.best_k}",
                                             labelOpts={"color": theme.ACCENT_2,
                                                        "position": 0.9}))

        # LL plot
        ll_plot = _plot(self.gw, 0, 1, title="Log-likelihood vs K",
                        x_label="K", y_label="log-likelihood")
        ll_plot.addItem(pg.PlotDataItem(ks, best_lls,
                                        pen=pg.mkPen(theme.GOOD, width=2.5),
                                        symbol="o", symbolSize=10,
                                        symbolBrush=pg.mkBrush(theme.GOOD),
                                        symbolPen=pg.mkPen("white")))

        # Table of best per K
        rows = []
        for k in ks:
            entry = bic_result.by_k[k]
            run = entry["best_run"]
            if run is None:
                continue
            rows.append((k, run.bic, run.log_likelihood, run.converged,
                         run.threshold_satisfied, run.n_iter, run.seed))
        self.table.setRowCount(len(rows))
        for r, (k, bic, ll, conv, thr, ni, sd) in enumerate(rows):
            cells = [
                str(k), f"{bic:,.2f}", f"{ll:,.2f}",
                "✓" if conv else "-", "✓" if thr else "-", str(ni), str(sd),
            ]
            for c, val in enumerate(cells):
                it = QTableWidgetItem(val)
                if k == bic_result.best_k:
                    it.setForeground(QColor(theme.ACCENT))
                self.table.setItem(r, c, it)

        if bic_result.best_k is not None:
            self.title.setText(
                f"Best model: K = {bic_result.best_k}   ·   "
                f"BIC = {bic_result.by_k[bic_result.best_k]['bic']:,.2f}"
            )
        else:
            self.title.setText("No valid restart in the BIC scan. Try more iterations or restarts.")


# ----------------------------------------------------------------------------
# Compare view: table of runs + bar comparisons
# ----------------------------------------------------------------------------
class CompareView(QWidget):
    def __init__(self):
        super().__init__()
        root = QVBoxLayout(self); root.setContentsMargins(12, 12, 12, 12); root.setSpacing(8)

        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels([
            "#", "Method", "Family", "K", "BIC", "log-likelihood",
            "Conv.", "θ-OK", "Iter"
        ])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        root.addWidget(self.table, 1)

        self.gw = pg.GraphicsLayoutWidget(); self.gw.setBackground(theme.BG_0)
        self.gw.setMinimumHeight(460)
        root.addWidget(self.gw, 1)

    def update_runs(self, runs: list, dataset=None, trial_idx: int = 0):
        self.table.setRowCount(0)
        self.gw.clear()
        if not runs:
            return
        self.table.setRowCount(len(runs))
        for r, run in enumerate(runs):
            cells = [
                f"#{r+1}", run.method_label, run.family, str(run.n_states),
                f"{run.bic:,.2f}", f"{run.log_likelihood:,.2f}",
                "✓" if run.converged else "-",
                "✓" if run.threshold_satisfied else "-",
                str(run.n_iter),
            ]
            for c, val in enumerate(cells):
                it = QTableWidgetItem(val)
                it.setToolTip(run.name)
                self.table.setItem(r, c, it)

        # Bar charts: BIC, LL, K
        xs = np.arange(len(runs)) + 1
        labels = [f"#{i+1}" for i in range(len(runs))]
        colors_qc = [theme.state_qcolor(i, max(len(runs), 3)) for i in range(len(runs))]

        bic_plot = _plot(self.gw, 0, 0, title="BIC", x_label="run", y_label="BIC")
        for i, run in enumerate(runs):
            bic_plot.addItem(pg.BarGraphItem(x=[xs[i]], height=[run.bic], width=0.7,
                                             brush=pg.mkBrush(colors_qc[i]),
                                             pen=pg.mkPen(colors_qc[i])))
        bic_plot.getAxis("bottom").setTicks([[(i + 1, labels[i]) for i in range(len(runs))]])

        ll_plot = _plot(self.gw, 0, 1, title="log-likelihood", x_label="run", y_label="LL")
        for i, run in enumerate(runs):
            ll_plot.addItem(pg.BarGraphItem(x=[xs[i]], height=[run.log_likelihood], width=0.7,
                                            brush=pg.mkBrush(colors_qc[i]),
                                            pen=pg.mkPen(colors_qc[i])))
        ll_plot.getAxis("bottom").setTicks([[(i + 1, labels[i]) for i in range(len(runs))]])

        # Switches per minute, averaged over trials, computed from the Viterbi sequence.
        sw_plot = _plot(self.gw, 0, 2, title="switches / min  (Viterbi)",
                        x_label="run", y_label="switches / min")
        for i, run in enumerate(runs):
            try:
                from .models import REGISTRY
                spec = REGISTRY[run.method_key]
                rates = []
                for x in run.prepared_trials:
                    _post, vit = spec.decode(run.result, x)
                    n = vit.size
                    if n < 2 or run.bin_size <= 0:
                        continue
                    switches = int(np.sum(vit[1:] != vit[:-1]))
                    rates.append(switches / (n * run.bin_size / 60.0))
                rate = float(np.mean(rates)) if rates else 0.0
            except Exception:
                rate = 0.0
            sw_plot.addItem(pg.BarGraphItem(x=[xs[i]], height=[rate], width=0.7,
                                            brush=pg.mkBrush(colors_qc[i]),
                                            pen=pg.mkPen(colors_qc[i])))
        sw_plot.getAxis("bottom").setTicks([[(i + 1, labels[i]) for i in range(len(runs))]])
        self._render_temporal_comparison(runs, dataset, trial_idx)
        layout = self.gw.ci.layout
        layout.setRowStretchFactor(0, 1)
        layout.setRowStretchFactor(1, 2)
        layout.setRowStretchFactor(2, 2)

    def _signal_for_comparison(self, dataset, trial_idx: int):
        if dataset is None or dataset.n_trials == 0:
            return None
        ti = max(0, min(int(trial_idx), dataset.n_trials - 1))
        trial = np.asarray(dataset.trials[ti], dtype=float)
        if trial.ndim == 1:
            trial = trial[None, :]
        t = dataset.time_axis(ti)
        n_bins = min(trial.shape[1], t.size)
        trial = trial[:, :n_bins]
        t = t[:n_bins]
        if n_bins == 0:
            return None
        if dataset.kind == "counts":
            y = np.nansum(trial, axis=0)
            label = "total count"
        elif trial.shape[0] == 1:
            y = trial[0]
            label = dataset.channel_labels[0] if dataset.channel_labels else "signal"
        else:
            rows = []
            for row in trial:
                finite = row[np.isfinite(row)]
                if finite.size == 0:
                    rows.append(np.zeros_like(row))
                    continue
                center = float(np.nanmean(finite))
                scale = float(np.nanstd(finite)) or 1.0
                rows.append((row - center) / scale)
            y = np.nanmean(np.vstack(rows), axis=0)
            label = "mean z signal"
        return t, np.asarray(y, dtype=float), label, ti

    def _decode_rows_for_comparison(self, runs: list, trial_idx: int, max_rows: int = 12):
        rows = []
        for run in runs[:max_rows]:
            if not getattr(run, "prepared_trials", None):
                continue
            ti = max(0, min(int(trial_idx), len(run.prepared_trials) - 1))
            try:
                td = decode_trial(run, ti)
            except Exception:
                continue
            rows.append((run, td))
        return rows

    def _state_rows_image(self, rows: list, n_bins: int, alpha: int = 220) -> np.ndarray:
        img = np.zeros((len(rows), n_bins, 4), dtype=np.uint8)
        for r, (run, td) in enumerate(rows):
            rgba = _state_rgba(run.n_states, alpha=alpha)
            labels = np.clip(td.viterbi[:n_bins], 0, run.n_states - 1)
            img[r, :, :] = rgba[labels]
        return img

    def _render_temporal_comparison(self, runs: list, dataset, trial_idx: int):
        signal = self._signal_for_comparison(dataset, trial_idx)
        rows = self._decode_rows_for_comparison(runs, trial_idx)
        if signal is None or not rows:
            return
        t, y, y_label, ti = signal
        n_bins = min([t.size, y.size] + [row[1].viterbi.size for row in rows])
        if n_bins < 2:
            return
        t = t[:n_bins]
        y = y[:n_bins]
        x0 = float(t[0])
        x1 = float(t[-1])
        if x1 <= x0:
            x1 = x0 + float(rows[0][0].bin_size) * max(n_bins, 1)

        finite_y = y[np.isfinite(y)]
        if finite_y.size:
            y_min = float(finite_y.min())
            y_max = float(finite_y.max())
        else:
            y_min, y_max = -1.0, 1.0
        if y_max <= y_min:
            y_min -= 0.5
            y_max += 0.5
        headroom = max((y_max - y_min) * 0.08, 1e-9)
        y_min -= headroom
        y_max += headroom

        first_run, first_td = rows[0]
        signal_plot = _plot(
            self.gw,
            1,
            0,
            title=f"Trial {ti + 1} signal with #{1} state overlay",
            x_label="time (s)",
            y_label=y_label,
            colspan=3,
        )
        signal_plot.setMouseEnabled(x=True, y=False)
        from PySide6.QtCore import QRectF

        overlay = _state_image_1d(
            first_td.viterbi[:n_bins],
            np.ones(n_bins, dtype=bool),
            first_run.n_states,
            alpha=55,
        )
        bg = pg.ImageItem(overlay)
        bg.setRect(QRectF(x0, y_min, x1 - x0, y_max - y_min))
        bg.setZValue(-10)
        signal_plot.addItem(bg)
        signal_plot.addItem(pg.PlotDataItem(t, y, pen=pg.mkPen(QColor(theme.TEXT), width=1.2)))
        signal_plot.setXRange(x0, x1, padding=0)
        signal_plot.setYRange(y_min, y_max, padding=0)

        gantt_title = "Viterbi state Gantt by run"
        if len(runs) > len(rows):
            gantt_title += f"  (showing {len(rows)} of {len(runs)})"
        gantt = _plot(
            self.gw,
            2,
            0,
            title=gantt_title,
            x_label="time (s)",
            y_label="run",
            colspan=3,
        )
        gantt.setMouseEnabled(x=True, y=False)
        img = self._state_rows_image(rows, n_bins)
        item = pg.ImageItem(img)
        item.setRect(QRectF(x0, -0.5, x1 - x0, len(rows)))
        gantt.addItem(item)
        ticks = [
            (i, f"#{i + 1} K={run.n_states}")
            for i, (run, _td) in enumerate(rows)
        ]
        gantt.getAxis("left").setTicks([ticks])
        gantt.setYRange(-0.5, len(rows) - 0.5, padding=0)
        gantt.setXRange(x0, x1, padding=0)
        gantt.setXLink(signal_plot)
        gantt.invertY(True)
