"""spectro-live Qt/PySide6 UI — instrument-style panel.

Layout (oscilloscope-like):
    [ menu bar: File | Instrument | Help ]
    [ big-button toolbar: START/PAUSE · SINGLE · ZERO · FREEZE B · SAVE ]
    [ settings column | spectral plot | measurements column ]
    [ status bar ]
"""
from __future__ import annotations

import json
import math
import os
import queue
import re
import tempfile
import textwrap
import time
from datetime import datetime

import numpy as np

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.patches import PathPatch

from PySide6.QtCore import (QPointF, QRect,
                            QRectF, Qt, QTimer, Signal)
from PySide6.QtGui import (QAction, QActionGroup, QBrush, QColor, QIcon,
                           QKeySequence, QPainter, QPainterPath,
                           QPen, QPixmap, QPolygonF)
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
    QFileDialog, QFrame, QGridLayout, QHBoxLayout, QGroupBox, QLabel,
    QListWidget, QMainWindow, QMenu, QMenuBar, QMessageBox, QPushButton,
    QRadioButton, QScrollArea, QSizePolicy, QSpinBox, QStatusBar, QTextEdit,
    QStyledItemDelegate, QToolBar, QVBoxLayout, QWidget,
)

from . import sources
from .processing import (
    Spectrum, average, cct_from_xyz, chromaticity_image, dominant_wavelength,
    find_peaks, integrate, linear_channels, pchip_smooth,
    peak_info, photopic_efficacy, photometric_absolute, rainbow_palette,
    radiometric_absolute, radiometric_from_photometric,
    ratio_pct, saturation_fraction, smooth, subtract, srgb_preview,
    xyz_from_values, unreliable_band_lo, gamut_report, cri_from_spectrum,
    chrom_space_name,
    xy_from_xyz, xy_to_chrom_space,
)

APP_NAME = "spectro-live"
PLOT_X_MIN, PLOT_X_MAX = 380.0, 730.0

# ---- GNOME-flat dark palette -------------------------------------------
BG      = "#1b1b1b"     # window
CARD    = "#262626"     # cards / panels
FIELD   = "#1e1e1e"     # entry fields
FG      = "#f2f2f2"
MUTED   = "#9a9a9a"
BORDER  = "#3a3a3a"
ACCENT  = "#3584e4"

# One size, one place — the FONT block. Every QSS font-size literal
# comes from here (the colour palette gets a type palette beside it),
# ordered small -> large. The matplotlib chart sizes (CHART_* below)
# are a SEPARATE family — matplotlib "points" are not Qt points, so
# they get their own named constants rather than dishonestly
# sharing these.
FONT_HINT   = "9pt"    # ordered small -> large
FONT_VALUE  = "9.5pt"
FONT_HEAD   = "10pt"
FONT_BASE   = "11pt"   # feeds the app-wide QFont too (tooltips)
FONT_LARGE  = "14pt"

# The chart (matplotlib) family — matplotlib "points" are a different
# unit from Qt points, so they share names with the QSS block but not
# values' units: named once, used everywhere the chart paints text.
CHART_INSET_LEG = 6      # inset legend             (ordered small -> large)
CHART_INSET_TICK = 7     # inset tick labels
CHART_INSET     = 8      # chromaticity inset axes titles
CHART_ANNO      = 10     # annotation letters on the diagram
CHART_AXIS      = 11     # main-plot axis titles/ticks
ACCENT_FG = "#ffffff"
OK      = "#2ec27e"
HOT     = "#e01b24"
WARN    = "#f5c211"
# the LAMPS' "demand" colour: amber, not the yellow the saturation bar
# uses — a warmer, more orange tone for the annunciator family
LAMP_AMBER = "#f5a623"
REF     = "#ff7b52"   # trace B colour
# dark, desaturated tints of OK/ACCENT: the fill for lamps whose
# border says the state but whose inside is not lit (selected /
# calibrated-but-idle)
OK_TINT   = "#24382e"
ACCENT_TINT = "#26364a"

WHITE   = "#ffffff"       # pure white (hover text, chart marks; FG is #f2f2f2)
BLACK   = "#000000"       # pure black (text on amber lamps, D65 dot in true-sRGB)
# -- hover / press states: one family, one place ------------------------
HOVER_BG = "#4a4a4a"      # dropdown popup item under the mouse
BTN_HOVER = "#333333"     # plain buttons
BTN_PRESSED = "#404040"   # plain buttons, pressed
BTN_ACCENT_HOVER = "#4a90e8"   # RUN / SINGLE while active
LAMP_HOVER = "#2c313a"     # annunciator lamps, hover fill (unlit)
LAMP_HOVER_BORDER = "#7fb4ff"  # unlit lamp hover border: a light-blue hint
# tooltips share MUTED's light gray (#9a9a9a) as their background, so
# the text must flip to dark (BLACK) to stay readable — a LIGHT tip
# lifts off the dark widgets it points at even harder than the old
# #303030 did, and never shares a background with the widget beneath
# halfway between MUTED (#9a9a9a) and BORDER (#3a3a3a): a mid-gray
# tip that still lifts clearly off every dark widget it points at
TOOLTIP_BG = "#6a6a6a"
# one shared padding for the whole tooltip family — tight on purpose
TOOLTIP_PAD = "2px 5px"


def tooltip_css() -> str:
    """THE one tooltip recipe. Used twice: in the app stylesheet, and
    appended to every WIDGET stylesheet we set — a widget stylesheet
    BEATS the app stylesheet for everything under that widget, QToolTip
    included (Qt attaches the shared tip to the widget that triggered
    it), so any styled widget that shows tooltips must carry the rule
    itself or its tooltips fall back to the black Fusion default.

    NOTE: no border-radius here, deliberately — QTipLabel paints an
    opaque SQUARE tip panel (palette ToolTipBase) UNDER the QSS, so
    any rounding left the background bleeding past the rounded border
    through the corners. Square border, square panel: same shape,
    no outshoot."""
    return (f"QToolTip {{ background: {TOOLTIP_BG}; color: {FG};"
            f" border: 1px solid {BORDER};"
            f" padding: {TOOLTIP_PAD}; }}")


TOOLTIP_QSS = tooltip_css()

# stylesheet slots painted at the panel level (BODY = the side panels'
# backing, matching the toolbar / status bar): named so the whole
# family is tweakable in one place
PANEL_BODY = CARD

QSS = f"""
/* Annunciator lamps: the per-state stylesheets only repaint colours
   and borders, so the horizontal padding lives here once — the lamp
   text must never touch its border, same rule as the buttons. */
LampLabel {{ padding: 2px 10px; }}
/* the saturation bar paints only its track; the gutter above it (where
   the 100 % label lives, right under the annunciator strip) must not
   form an opaque block colliding with the CALIBRATION / mode lamps */
SatBar {{ background: transparent; }}
QMainWindow, QWidget {{ background: {BG}; color: {FG}; font-size: {FONT_BASE}; }}
QMenuBar {{ background: {CARD}; }}
QMenuBar::item:selected {{ background: {CARD}; }}
QMenu {{ background: {CARD}; color: {FG}; border: 1px solid {BORDER}; }}
QMenu::item {{ padding: 6px 24px; }}
QMenu::item:selected {{ background: {ACCENT}; }}
QToolBar {{ background: {CARD}; border: none; padding: 6px; spacing: 8px; }}
QGroupBox {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px;
    margin-top: 14px; padding: 6px 6px 4px 6px; font-weight: bold;
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 10px; padding: 0 4px;
    color: {FG}; background: {CARD};
}}
QLabel {{ background: transparent; }}
QLabel[valueHead] {{ color: {MUTED}; font-weight: normal; font-size: {FONT_HEAD}; }}
QLabel[bigValue] {{
    background: {FIELD}; border: 1px solid {BORDER}; border-radius: 6px;
    padding: 5px 8px; font-size: {FONT_VALUE}; font-weight: normal;
    font-family: "DejaVu Sans Mono", "Liberation Mono", monospace;
}}
QListWidget {{
    background: {FIELD}; color: {FG}; border: 1px solid {BORDER};
    border-radius: 6px; font-size: {FONT_VALUE};
    font-family: "DejaVu Sans Mono", "Liberation Mono", monospace;
}}
QLabel[nameValue] {{
    color: {MUTED}; font-size: {FONT_LARGE}; font-weight: bold; background: transparent;
    border: none; padding: 0;
}}
QLabel[hint] {{ color: {MUTED}; font-size: {FONT_HINT}; background: transparent; }}
QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {FIELD}; color: {FG}; border: 1px solid {BORDER};
    border-radius: 6px; padding: 4px 6px; min-height: 18px;
}}
QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{ color: {MUTED}; }}
QComboBox QAbstractItemView {{
    background: {FIELD}; color: {FG}; border: 1px solid {BORDER};
    selection-background-color: {ACCENT}; selection-color: {ACCENT_FG};
    font-size: {FONT_BASE};
}}
/* Dropdown / spin arrows, painted from generated chevron image files
   (see _arrow_images / styled_qss): once a stylesheet rules a combo
   or spin box, Qt paints the arrow ONLY from an image: rule — a bare
   color: paints nothing, and the style fallback differs per platform
   (invisible on macOS). The image files make the arrows identical
   everywhere, and the drop-down / spin button subcontrols get the
   border colour so the arrow buttons are delineated like fields. */
QComboBox::drop-down {{
    border: none; width: 26px;
    background: {BORDER};
    border-top-right-radius: 4px; border-bottom-right-radius: 4px;
    margin: 0; padding: 0;
}}
QComboBox::down-arrow {{
    image: url(@ARROW_DOWN@);
    width: 14px; height: 14px;
}}
QSpinBox::up-button, QDoubleSpinBox::up-button {{
    width: 22px; background: {BORDER};
    border: none; border-top-right-radius: 4px;
    margin: 0; padding: 0;
    subcontrol-origin: border; subcontrol-position: top right;
}}
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    width: 22px; background: {BORDER};
    border: none; border-bottom-right-radius: 4px;
    margin: 0; padding: 0;
    subcontrol-origin: border; subcontrol-position: bottom right;
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: url(@ARROW_UP@);
    width: 12px; height: 12px;
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: url(@ARROW_DOWN@);
    width: 12px; height: 12px;
}}
QPushButton {{
    background: {CARD}; color: {FG}; border: 1px solid {BORDER};
    border-radius: 8px; padding: 8px 14px; font-weight: bold;
}}
QPushButton:hover {{ background: {BTN_HOVER}; }}
QPushButton:pressed {{ background: {BTN_PRESSED}; }}
QPushButton:focus {{ border: 1px solid {ACCENT}; }}
QPushButton#btnStart {{
    background: {ACCENT}; color: {ACCENT_FG}; border: none;
    font-size: {FONT_LARGE}; min-height: 30px; min-width: 75px;
}}
QPushButton#btnStart:hover {{ background: {BTN_ACCENT_HOVER}; }}
QPushButton#btnRunning {{
    background: {HOT}; color: {WHITE}; border: none;
    font-size: {FONT_LARGE}; min-height: 30px; min-width: 75px;
}}
QPushButton#btnRunning:hover {{ background: #ff4a52; }}
QPushButton#btnSingle {{
    background: {ACCENT}; color: {ACCENT_FG}; border: none;
    font-size: {FONT_LARGE}; min-height: 30px; min-width: 75px;
}}
QPushButton#btnSingle:hover {{ background: {BTN_ACCENT_HOVER}; }}
/* the three instrument buttons are borderless by design — their
   body IS the colour. Disabled they must not melt into the toolbar:
   give them the shared BORDER outline (reported: SINGLE during
   '… READING' looked like it vanished — CARD on BG with borderless
   body is indistinguishable at a glance) */
QPushButton#btnStart:disabled, QPushButton#btnRunning:disabled,
QPushButton#btnSingle:disabled {{
    background: {CARD}; color: {MUTED}; border: 1px solid {BORDER};
}}
QPushButton[tool] {{ min-height: 20px; padding: 10px 14px; }}
QCheckBox, QRadioButton {{
    background: transparent; spacing: 8px;
}}
/* Checkbox / radio INDICATORS: Fusion draws the unchecked one as a
   bevel in palette colours that vanish on this dark theme (reported:
   unticked boxes were invisible). Ruling the indicator in QSS takes
   over the drawing completely — the marks come from generated
   check/dot PNGs (see _indicator_images), the same trick as the
   dropdown-arrow chevrons. */
QCheckBox::indicator, QRadioButton::indicator {{
    width: 16px; height: 16px;
    border: 1px solid {BORDER}; background: {FIELD};
}}
QCheckBox::indicator {{ border-radius: 3px; }}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
    border-color: {ACCENT};
}}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{
    border-color: {BORDER}; background: {BG};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT}; border-color: {ACCENT};
    image: url(@CHECK@);
}}
QRadioButton::indicator:checked {{
    background: {ACCENT}; border-color: {ACCENT};
    border-radius: 8px;
    image: url(@DOT@);
}}
/* disabled, checked included: Averaging off disables Rolling /
   Cumulative — the text grays to MUTED and the ACCENT fill drops to
   BORDER so the SELECTED radio goes visually inert too (it used to
   stay bright blue while its sibling dimmed) */
QCheckBox:disabled, QRadioButton:disabled {{ color: {MUTED}; }}
QCheckBox::indicator:checked:disabled,
QRadioButton::indicator:checked:disabled {{
    background: {BORDER}; border-color: {BORDER};
}}
QListWidget::item:selected {{ background: {ACCENT}; }}
QTextEdit {{ background: {FIELD}; color: {FG}; border: 1px solid {BORDER}; }}
QStatusBar {{ background: {CARD}; color: {MUTED}; }}
/* the size grip QMainWindow plants in the status bar's bottom-right
   corner: functional (corner-drag resizing) but must not paint its
   own darker rectangle — it shows the bar's CARD colour like the
   rest of the statusbar surface */
QSizeGrip {{ background: transparent; }}
QScrollArea {{ border: none; }}
QScrollBar:vertical {{ background: {BG}; width: 10px; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 5px; }}
QScrollBar::handle:vertical:hover {{ background: {ACCENT}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
/* One place for the whole tooltip family (tooltip_css): the OS
   default hands out a mixed bag (near-black with black text, or black
   on yellow). The rule sits LAST so nothing above can shadow it in
   this stylesheet's own order. */
{TOOLTIP_QSS}

"""



# =======================================================================
# arrow chevron images + popup view styling
# =======================================================================
_ARROW_IMAGES: dict | None = None
_INDICATOR_IMAGES: dict | None = None


def _arrow_images() -> dict:
    """Generate the light chevron PNGs the QSS `image:` rules above
    reference, once per process, into the temp dir. The @2x twins sit
    beside the 1x files so QIcon picks them up on Retina displays.
    Returns {'ARROW_DOWN': path, 'ARROW_UP': path}."""
    global _ARROW_IMAGES
    if _ARROW_IMAGES is not None:
        return _ARROW_IMAGES
    base_dir = os.path.join(tempfile.gettempdir(), "spectrov-arrows")
    os.makedirs(base_dir, exist_ok=True)
    half_w, half_h = 0.32, 0.62          # arrow shape, fractions of cell
    triangles = {
        "ARROW_DOWN": [(0.5 - half_w, 0.5 - half_h / 2),
                       (0.5 + half_w, 0.5 - half_h / 2),
                       (0.5, 0.5 + half_h / 2)],
        "ARROW_UP":   [(0.5 - half_w, 0.5 + half_h / 2),
                       (0.5 + half_w, 0.5 + half_h / 2),
                       (0.5, 0.5 - half_h / 2)],
    }
    out = {}
    for name, tri in triangles.items():
        for scale, px in ((1, 14), (2, 28)):
            pm = QPixmap(px, px)
            pm.fill(Qt.GlobalColor.transparent)
            p = QPainter(pm)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(FG)))
            p.drawPolygon(QPolygonF([QPointF(px * x, px * y)
                                     for x, y in tri]))
            p.end()
            suffix = "" if scale == 1 else "@2x"
            path = os.path.join(base_dir, name.lower() + suffix + ".png")
            pm.save(path, "PNG")
            if scale == 1:
                out[name] = os.path.abspath(path).replace("\\", "/")
    _ARROW_IMAGES = out
    return out


def _indicator_images() -> dict:
    """Generate the check-mark and dot PNGs the QSS ::indicator rules
    reference, once per process, into the temp dir beside the arrow
    chevrons (see _arrow_images for the @2x Retina rationale).
    Returns {'CHECK': path, 'DOT': path}."""
    global _INDICATOR_IMAGES
    if _INDICATOR_IMAGES is not None:
        return _INDICATOR_IMAGES
    base_dir = os.path.join(tempfile.gettempdir(), "spectrov-arrows")
    os.makedirs(base_dir, exist_ok=True)
    out = {}
    for name in ("CHECK", "DOT"):
        for scale, px in ((1, 16), (2, 32)):
            pm = QPixmap(px, px)
            pm.fill(Qt.GlobalColor.transparent)
            p = QPainter(pm)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(FG)))
            if name == "CHECK":
                # a 3-segment polyline check, drawn thick enough to
                # read at 16 px: (x, y) in cell fractions. NB: build a
                # FRESH solid pen — the painter's pen is NoPen here
                # (the dot branch wants no outline), and drawing the
                # polyline with it painted nothing (the shipped
                # checked-checkbox-was-all-blue bug)
                pen = QPen(QColor(FG))
                pen.setWidthF(px / 8.0)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                p.setPen(pen)
                pts = [(0.22, 0.48), (0.42, 0.70), (0.80, 0.24)]
                p.drawPolyline(QPolygonF([QPointF(px * x, px * y)
                                          for x, y in pts]))
            else:                       # DOT: solid inner disc
                d = 0.50
                p.drawEllipse(QRectF(px * (0.5 - d / 2), px * (0.5 - d / 2),
                                      px * d, px * d))
            p.end()
            suffix = "" if scale == 1 else "@2x"
            path = os.path.join(base_dir, name.lower() + suffix + ".png")
            pm.save(path, "PNG")
            if scale == 1:
                out[name] = os.path.abspath(path).replace("\\", "/")
    _INDICATOR_IMAGES = out
    return out


def styled_qss() -> str:
    """QSS with the @ARROW_*@ / @CHECK@ / @DOT@ image tokens replaced
    by the generated file paths (paints nothing meaningful until then
    — the image: rule cannot carry a literal empty url)."""
    qss = QSS
    for token, path in _arrow_images().items():
        qss = qss.replace(f"@{token}@", path)
    for token, path in _indicator_images().items():
        qss = qss.replace(f"@{token}@", path)
    return qss


def _dark_popup(combo: QComboBox) -> None:
    """Pin every combo popup to the dark theme, immune to what its
    ancestors do. Two separate problems live here:

    1. CASCADE BREAK: a stylesheet anywhere on an ancestor (several
       panels use 'background: transparent' to escape the global
       QWidget rule) propagates into the popup and leaves it
       UNPAINTED — on the Mac the 'Colour system' dropdown came out
       white with white text. A rule set directly on the view wins
       the cascade.

    2. WHITE FRAME LINES: the popup's container
       (QComboBoxPrivateContainer, a QFrame) draws its StyledPanel
       frame from the PALETTE — outside the app stylesheet's reach —
       which is light-theme white by default (1px white lines top and
       bottom of every popup). The view sitting 4px inside can never
       cover them; the container itself gets a stylesheet with border
       none, opaque field background.

    The view also gets a plain QStyledItemDelegate: the popup's
    built-in delegate does not route item painting through the
    stylesheet, so ::item rules (hover highlight!) never rendered —
    that is why nothing happened on mouse-over before."""
    view = combo.view()
    view.setStyleSheet(
        "QAbstractItemView {"
        f" background: {FIELD}; color: {FG};"
        f" border: 1px solid {BORDER}; outline: none;"
        f" selection-background-color: {ACCENT};"
        f" selection-color: {ACCENT_FG};"
        " }"
        "QAbstractItemView::item {"
        " background: transparent; color: "
        f"{FG}; padding: 2px 6px; }}"
        "QAbstractItemView::item:hover {"
        f" background: {HOVER_BG}; color: {WHITE}; }}"
        "QAbstractItemView::item:selected {"
        f" background: {ACCENT}; color: {WHITE}; }}")
    view.setItemDelegate(QStyledItemDelegate(view))
    container = view.parentWidget()
    if container is not None and container is not combo:
        container.setObjectName("comboPopupContainer")
        container.setStyleSheet(
            "#comboPopupContainer {"
            f" border: none; background: {FIELD}; }}")


# =======================================================================
# value widget: big number + descriptive name
# =======================================================================
def _paint_only(w: QWidget, name: str, css: str) -> None:
    """Paint exactly ONE widget, never its descendants. The rules in a
    widget stylesheet WITHOUT a selector propagate to every child —
    all the way into the QToolTip Qt attaches to whichever widget
    triggered it (tooltips came out panel-coloured instead of
    TOOLTIP_BG — the 'still black in the measurements' bug) and into
    tooltip-visible widgets into the combo popup views. A selector on a
    unique object name keeps every colour exactly where it was
    intended. The tooltip rule rides along: a widget stylesheet beats
    the app stylesheet for the QToolTip shown from this widget, so the
    recipe must be carried here too (see tooltip_css)."""
    w.setObjectName(name)
    w.setStyleSheet(f"#{name} {{ {css} }}\n{TOOLTIP_QSS}")


def value_box(parent: QGridLayout, row: int, name: str, tooltip: str = ""):
    """A scientific readout: small grey title above a left-aligned
    monospace value field."""
    head = QLabel(name.upper())
    head.setProperty("valueHead", True)
    head.setToolTip(wrap_tip(tooltip))
    val = QLabel("—")
    val.setProperty("bigValue", True)
    val.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
    # let the user select the numbers with the mouse and Ctrl+C them
    val.setTextInteractionFlags(
        Qt.TextInteractionFlag.TextSelectableByMouse)
    val.setToolTip(wrap_tip(tooltip))
    wrap = QWidget()
    # plain QWidgets get painted BG-dark by the global QSS rule —
    # the head VALUE strip must show the group background instead.
    # _paint_only SCOPES the rule: a bare one propagates into every
    # descendant, the QToolTip attached to the strip included (the
    # 'measurements tooltips still black' bug)
    _paint_only(wrap, "valuebox", "background: transparent;")
    lay = QVBoxLayout(wrap)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(1)
    lay.addWidget(head)
    lay.addWidget(val)
    wrap.setToolTip(wrap_tip(tooltip))
    val._head = head          # the title may be renamed at runtime
    parent.addWidget(wrap, row, 0, 1, 2)
    return val


def wrap_tip(text: str, width: int = 72) -> str:
    """Re-wrap a tooltip so no line dwarfs the screen: Qt NEVER wraps
    rich text, so every tooltip must carry its own <br> breaks. Manual
    <br>s are kept (they are deliberate structure); anything longer
    than `width` is wrapped on spaces, and because HTML tags contain
    no spaces the units stay whole — a <b>bold</b> word is never cut
    in half. Every tooltip in the app goes through this, on the same
    principle as the QToolTip style rule: one family, one look."""
    if not text:
        return text
    out = []
    for para in text.replace("\n", " ").split("<br>"):
        para = " ".join(para.split())          # collapse runs of spaces
        if not para:
            out.append("")
            continue
        if len(para) <= width:
            out.append(para)
        else:
            out.extend(textwrap.wrap(
                para, width=width, break_long_words=False,
                break_on_hyphens=False))
    return "<br>".join(out)


def btn(text, tooltip, callback, minh=None):
    b = QPushButton(text)
    b.setToolTip(wrap_tip(tooltip))
    b.clicked.connect(callback)
    b.setProperty("tool", True)
    # the same pointing-hand affordance the annunciator lamps have —
    # every one of these is clickable
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    return b


def mono_icon(kind: str, px: int = 22):
    """A simple monochrome WHITE icon drawn by hand — alpha/line art
    that sits cleanly on the dark theme (the QStyle standard icons
    are multi-coloured and clash with it). Transparent background,
    2 px strokes, antialiased."""
    pm = QPixmap(px, px)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(FG), 2.0)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    brush = QBrush(QColor(FG))
    s = px
    m = 3                       # margin

    if kind == "play":          # solid triangle
        p.setBrush(brush)
        path = QPainterPath()
        path.moveTo(m + 1, m)
        path.lineTo(s - m, s / 2)
        path.lineTo(m + 1, s - m)
        path.closeSubpath()
        p.drawPath(path)
    elif kind == "stop":        # solid square
        p.setBrush(brush)
        p.drawRect(QRectF(m + 1, m + 1, s - 2 * m - 2, s - 2 * m - 2))
    elif kind == "single":      # solid circle: one deliberate shot
        p.setBrush(brush)
        p.drawEllipse(QRectF(m + 1, m + 1, s - 2 * m - 2, s - 2 * m - 2))
    elif kind == "zero":        # slashed circle: background captured
        p.drawEllipse(QRectF(m, m, s - 2 * m, s - 2 * m))
        p.drawLine(QPointF(m + 1, s - m - 1), QPointF(s - m - 1, m + 1))
    elif kind == "clear-zero":  # circle with X: the stored zero removed
        p.drawEllipse(QRectF(m, m, s - 2 * m, s - 2 * m))
        p.drawLine(QPointF(m + 3, m + 3), QPointF(s - m - 3, s - m - 3))
        p.drawLine(QPointF(s - m - 3, m + 3), QPointF(m + 3, s - m - 3))
    elif kind == "freeze":      # arrow dropping onto a base line
        p.drawLine(QPointF(s / 2, m), QPointF(s / 2, s - m - 5))
        p.drawLine(QPointF(s / 2 - 4.5, s - m - 9),
                   QPointF(s / 2, s - m - 5))
        p.drawLine(QPointF(s / 2 + 4.5, s - m - 9),
                   QPointF(s / 2, s - m - 5))
        p.drawLine(QPointF(m + 2, s - m), QPointF(s - m - 2, s - m))
    elif kind == "clear-b":    # X: reference removed
        p.drawLine(QPointF(m + 2, m + 2), QPointF(s - m - 2, s - m - 2))
        p.drawLine(QPointF(s - m - 2, m + 2), QPointF(m + 2, s - m - 2))
    elif kind == "load":        # arrow rising from a tray
        p.drawLine(QPointF(s / 2, s - m - 1), QPointF(s / 2, m + 6))
        p.drawLine(QPointF(s / 2 - 4.5, m + 10.5),
                   QPointF(s / 2, m + 6))
        p.drawLine(QPointF(s / 2 + 4.5, m + 10.5),
                   QPointF(s / 2, m + 6))
        p.drawLine(QPointF(m + 2, s - m + 1 - 1), QPointF(s - m - 2, s - m))
    elif kind == "save":       # arrow dropping into a tray
        p.drawLine(QPointF(s / 2, m + 1), QPointF(s / 2, s - m - 6))
        p.drawLine(QPointF(s / 2 - 4.5, s - m - 10.5),
                   QPointF(s / 2, s - m - 6))
        p.drawLine(QPointF(s / 2 + 4.5, s - m - 10.5),
                   QPointF(s / 2, s - m - 6))
        p.drawLine(QPointF(m + 2, s - m), QPointF(s - m - 2, s - m))
    p.end()
    return QIcon(pm)


class LampLabel(QLabel):
    """An annunciator lamp that is also a button: clicking emits
    `clicked`. Only the CALIBRATION lamp uses that — a click demands a
    fresh calibration of every reading mode."""
    clicked = Signal()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(e)


class SatBar(QWidget):
    """Narrow vertical annunciator bar to the right of the chart: how
    close the busiest sensor channel is to its estimated clip level.
    Log scale (100 % = estimated clip at the top, 10 % at half height,
    0 at the bottom — the bottom sliver is the whole sub-1 % trace
    region; values above clip pinned at the top); dark when unknown;
    steady green below 70 %, amber to 99 %,
    flashing red at ~clip. The estimate comes from Argyll's documented
    integration-time ceilings, not a calibration of the user's unit."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frac = None          # None = no honest number exists
        self._lit = False          # flash phase (shared blink timer)
        # the flash (definitely at/over the estimated clip) starts well
        # past the estimate: the ceiling is an ESTIMATE from Argyll's
        # integration-time ceilings, so a reading a few % above it is a
        # solid-red warning, not a saturation alarm
        self._flash_from = 1.2
        self.setMinimumWidth(88)
        self.setMinimumHeight(160)
        # vertical insets of the track: default 6 px top and bottom;
        # the window replaces them with the chart's axes-box edges so
        # the bar's top and bottom align with the plot itself
        self._al_top = 6.0
        self._al_bottom = 6.0
        # click-through: the bar FLOATS over the centre panel (the
        # fix for the clipped "100 %" label), so on small windows its
        # top can overlap the lamp-button row below it. It has no
        # mouse interaction of its own, so let every click fall
        # straight through to whatever it covers.
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents,
                          True)

    def align_to(self, top_px: float, bottom_px: float):
        """Align the track's top and bottom with the main chart's axes
        box: the bar sits in the same layout row as the canvas, so its
        own top edge IS the canvas's top edge — `top_px`/`bottom_px`
        (canvas pixel coords of the axes box top / bottom, which
        double as the bar's insets) make the bar end exactly where the
        plot ends instead of floating inside its widget."""
        top_px = max(float(top_px), 0.0)
        bottom_px = max(float(bottom_px), 0.0)
        if (top_px, bottom_px) != (self._al_top, self._al_bottom):
            self._al_top, self._al_bottom = top_px, bottom_px
            self.update()

    def set_fraction(self, frac):
        changed = frac != self._frac
        self._frac = frac
        if changed:
            self.update()

    def tick(self):
        """Called from the shared 0.25 s blink timer for the flashing
        phase at clip level."""
        if self._frac is not None and self._frac >= self._flash_from:
            self._lit = not self._lit
            self.update()

    def _fill_colour(self):
        f = self._frac
        if f is None:
            return MUTED
        if f < 0.7:
            return OK
        if f < 0.995:
            return WARN
        if f >= self._flash_from and self._lit:
            return HOT
        # over the estimate but not (yet) clearly at clip: steady red
        return HOT if f < self._flash_from else "#4a1013"

    def paintEvent(self, _e):
        p = QPainter(self)
        try:
            self._paint(p)
        finally:
            p.end()

    def _paint(self, p):
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        # the bar occupies the left part; a 46 px label gutter at the
        # right carries the scale ticks (100 % = estimated clip at the
        # top, 10 % at half height, 0 at the bottom) — wide enough that
        # "100%" is legible, never clipped to a bare "1". Vertical
        # inset: the track is pinned to the axes box of the main chart
        # (align_to), defaulting to the old 6 px breathing room
        track = QRectF(4, self._al_top, w - 52,
                       max(h - self._al_top - self._al_bottom, 12.0))
        p.setPen(QColor(BORDER))
        p.setBrush(QColor(FIELD))
        p.drawRoundedRect(track, 5, 5)
        f = self._frac
        if f is not None:
            # log scale: 0 (bottom) .. 100 % (the estimated clip
            # level) at the VERY TOP — 10 % lands at exactly half
            # height; anything over clip simply pins at the top,
            # coloured by _fill_colour (steady red / flashing red above
            # the estimate)
            lo, hi = -2.0, 0.0        # log10(0.01) .. log10(1.0)
            v = math.log10(max(float(f), 1e-2))
            frac = (v - lo) / (hi - lo)
            frac = min(max(frac, 0.0), 1.0)
            fh = frac * track.height()
            if f > 0.0 and fh < 3.0:
                # a real but sub-1 % reading: show a visible sliver
                # instead of an (identical-looking) empty bar
                fh = 3.0
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(self._fill_colour()))
            r = track.height() - fh
            p.drawRoundedRect(QRectF(track.left(), track.top() + r,
                                    track.width(), fh), 5, 5)
            p.setPen(QColor(MUTED))
            p.setFont(self.font())
            lx = track.right() + 4
            lw = max(w - lx - 2, 20)
            # decade-and-doubling scale marks: 100 % (estimated clip) at
            # the top, then 50/10/5/2/1 % following the log scale (10 %
            # lands at exactly half height), and 0 at the very bottom —
            # the bottom sliver covers the whole 0–1 % trace region
            # (a log scale has no true zero)
            labels = (("100%", 1.0), ("50%", 0.849), ("25%", 0.699),
                      ("10%", 0.5), ("5%", 0.350), ("2.5%", 0.199),
                      ("0", 0.0))
            for label, vf in labels:
                y = track.bottom() - vf * track.height()
                p.drawLine(int(track.right()) + 1, int(y),
                           int(track.right()) + 3, int(y))
                # room check: the chart's top margin is deliberately
                # near-zero (tight chart), so the track top — the 100 %
                # tick — sits a couple of pixels from the widget's top
                # edge. A label CENTRED on that tick would be half
                # outside the widget and Qt clips painting there; no
                # background colour can cure it. When there is no room
                # above the tick, hang the text DOWNWARD from the
                # widget's top edge instead of centring it on the tick.
                if y - 9 < 0.0:
                    p.drawText(QRectF(lx, 0.0, lw, 18.0),
                               Qt.AlignmentFlag.AlignLeft
                               | Qt.AlignmentFlag.AlignTop,
                               label)
                else:
                    p.drawText(QRectF(lx, y - 8, lw, 16),
                               Qt.AlignmentFlag.AlignLeft
                               | Qt.AlignmentFlag.AlignVCenter,
                               label)
        else:
            p.setPen(QColor(MUTED))
            p.drawText(track, Qt.AlignmentFlag.AlignCenter
                       | Qt.TextFlag.TextWordWrap, "n/a")


# =======================================================================
# main window
# =======================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1280, 820)

        # ---- state ------------------------------------------------------
        self.events: queue.Queue = queue.Queue()
        self.worker = None
        self.running = False
        self.source_name = "ColorMunki (spotread)"
        self.mode_name = "Emissive"
        self.high_res = True
        self.clip_unreliable = False  # drop the ColorMunki's cloned bins
        self.fast = False
        self.force_cal = False   # demand a fresh calibration at every start
        self.csv_log = False
        # explicit ArgyllCMS spotread binary ("" = whatever is on PATH)
        self.argyll_path = ""
        # full spotread transcript to ~/spectro-live-logs/instrument-log-*.txt
        self.inst_log = False
        self.unit_name = "Raw (a.u.)"
        self.avg_on = False
        self.avg_rolling = True
        self.avg_n = 4
        self.smooth_on = False
        self.smooth_n = 5
        self.view_name = "Overlay (A + B)"
        self.curve_style = "lines"    # lines | smooth | steps
        self.x_mode = "wl"            # wl | freq | ev (x axis)
        self.chrom_space = "1931"      # 1931 | 1976 | 1960 diagram
        self.true_srgb = True     # chromaticity: true-colour sRGB island
        # how fresh is the calibration of the live session (green
        # border on the CALIBRATION lamp) — monotonic deadline, fades
        # after Argyll's dark-cal validity (1 h)
        self._cal_ok_until = 0.0
        # {"e"|"a"|"r": wall-clock stamp of the last fresh dance};
        # filled by _restore_settings, defaults empty before that
        self.cal_ts = {}
        self._sat_frac = None         # busiest sensor channel vs clip
        self.zero_ref: Spectrum | None = None
        self.btn_zero = None            # toolbar Zero button (highlight)
        self.reference: Spectrum | None = None
        self._rad_prefix = "m"   # radiometric chart unit prefix (kW..nW)
        self.frames: list[Spectrum] = []
        self.trace_a: Spectrum | None = None
        self.last_raw: Spectrum | None = None
        self.marker1, self.marker2 = 546.1, 435.8
        self.x_lo, self.x_hi = PLOT_X_MIN, PLOT_X_MAX   # adapts per instrument
        self.band_lo, self.band_hi = self.x_lo, self.x_hi
        self.band_from_markers = False
        self.frame_count = 0
        self.read_times: list[float] = []
        self._t_frame = time.monotonic()
        self.log_lines: list[str] = []
        self.diag_path = None
        self.csv_path = None
        self.pending_cal = False
        self._pos_confirmed = False  # wheel slot confirmed by a reading?
        # a live instrument session exists (until no-device/reconnecting);
        # True at launch because the session auto-starts
        self._device_online = True
        self._t_frame = time.monotonic()
        # hardware-button frame tracking (paused single-captures vs. a
        # held/closed switch that keeps re-triggering spotread)
        self._last_hw_frame = 0.0
        self._hw_streak = 0
        self._hw_suppressing = False
        self._stop_since = 0.0         # time of the last STOP click
        self._cal_progress = None      # lamp-as-progress-bar fill fraction
        self._cal_stage = None         # remembered pass stage (1/3…)
        # vertical-unit selector: created here (the View → Units menu in
        # _menubar needs it), filled & wired in the settings panel
        from PySide6.QtWidgets import QComboBox as _QCB
        self.cmb_units = _QCB()

        self.inst_type = ""            # from spotread's "Instrument Type:"
        self.device_port = None        # None = spotread's default device
        self._restore_settings()      # load mode / hi-res / fast choices
        self._menubar()
        self._toolbar()
        central = QWidget()
        self.setCentralWidget(central)
        lay = QHBoxLayout(central)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(8)
        lay.addWidget(self._settings_panel(), 0)
        lay.addWidget(self._center_panel(), 1)
        lay.addWidget(self._readout_panel(), 0)
        self.ax.set_xlabel(self._axis_label(), color=FG, fontsize=CHART_AXIS)
        self._statusbar()
        self.status("Ready. Press START to connect and "
                    "auto-calibrate the instrument.")

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.timer.start(120)
        # level-apply the Averaging enable/disable ONCE at startup too
        # (without a toggle the radios' state never caught up: the app
        # opened with the checkbox off but the radios still lit)
        self._refresh_settings()

        # auto-connect on launch: start the instrument session right away
        # AND begin continuous recording immediately after calibration
        # (equivalent to pressing START at launch).
        self.worker = self._make_worker()
        self.worker.start()
        self._start_logs()
        self.running = True
        self._begin_recording()
        # every dropdown's popup view gets its background pinned
        # directly on the view itself: ancestors with stylesheets
        # (the panels use 'background: transparent') break the QSS
        # cascade into the popup (_dark_popup) — without this the
        # popup renders unpainted (white with white text on macOS,
        # e.g. the Colour system dropdown)
        for _c in self.findChildren(QComboBox):
            _dark_popup(_c)
        # honest startup wording: with Argyll's host cache still valid
        # spotread starts on it and NO wheel dance happens (the session
        # state will say "Calibration still valid"); the wheel is only
        # needed if spotread actually demands a calibration (stale
        # cache, forced calibration), which the CALIBRATION lamp shows.
        if (sources.argyll_cal_remaining(
                self.mode_name, bool(getattr(self, "fast", False)))
                or 0.0) > 0.0:
            self.status("Startup: reusing Argyll's saved calibration "
                        "while valid (dark 1 h, white 24 h) — turn the "
                        "wheel to position 2 ONLY if the CALIBRATION "
                        "lamp lights up; continuous readings begin "
                        "automatically.")
        else:
            self.status("Startup: calibrating (wheel to position 2 when "
                        "the CALIBRATION lamp lights up), then "
                        "continuous readings begin automatically.")

    # ---- persistent settings ------------------------------------------
    _CFG_PATH = os.path.expanduser("~/.config/spectro-live/settings.json")

    def _load_settings(self) -> dict:
        try:
            with open(self._CFG_PATH) as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_settings(self):
        """Persist the mode / hi-res / fast choices across restarts."""
        try:
            os.makedirs(os.path.dirname(self._CFG_PATH), exist_ok=True)
            with open(self._CFG_PATH, "w") as f:
                json.dump({"mode": self.mode_name, "high_res": self.high_res,
                           "fast": self.fast, "force_cal": self.force_cal,
                           "inst_log": getattr(self, "inst_log", False),
                           "argyll_path": getattr(self, "argyll_path", ""),
                           "curve_style": self.curve_style,
                           "x_mode": self.x_mode,
                           "clip_unreliable": self.clip_unreliable,
                           "chrom_space": getattr(self, "chrom_space", "1931"),
                           "cal_ts": getattr(self, "cal_ts", {})},
                          f, indent=2)
        except Exception:
            pass   # persistence is a convenience, never fatal

    def _restore_settings(self):
        cfg = self._load_settings()
        if cfg.get("mode") in ("Emissive", "Ambient", "Reflective"):
            self.mode_name = cfg["mode"]
        self.high_res = bool(cfg.get("high_res", self.high_res))
        self.fast = bool(cfg.get("fast", self.fast))
        self.force_cal = bool(cfg.get("force_cal", self.force_cal))
        self.inst_log = bool(cfg.get("inst_log", self.inst_log))
        # explicit ArgyllCMS binary — publish to the sources module
        # BEFORE the worker starts, so the first session uses it
        self.argyll_path = str(cfg.get("argyll_path", "") or "")
        if self.argyll_path:
            sources.set_argyll_path(self.argyll_path)
        self.clip_unreliable = bool(cfg.get("clip_unreliable",
                                            self.clip_unreliable))
        if cfg.get("curve_style") in ("lines", "smooth", "steps"):
            self.curve_style = cfg["curve_style"]
        if cfg.get("x_mode") in ("wl", "freq", "ev"):
            self.x_mode = cfg["x_mode"]
        if cfg.get("chrom_space") in ("1931", "1976", "1960"):
            self.chrom_space = cfg["chrom_space"]
        # true sRGB colours are no longer a menu option: always on
        # (a stale settings file carrying "true_srgb": false is
        # simply ignored)
        self.true_srgb = True
        # wall-clock stamps of the last FRESH calibration dance per
        # reading mode — the countdown after a cache-reuse restart is
        # reconstructed from these instead of restarting the window
        self.cal_ts = dict(cfg.get("cal_ts", {}))
        if not isinstance(self.cal_ts, dict):
            self.cal_ts = {}

    # ===================================================================
    # menubar: File | Instrument | Help
    # ===================================================================
    def _menubar(self):
        mb = self.menuBar()

        m_file = mb.addMenu("&File")
        for label, fn in (
                ("Load spectrum as trace A…", lambda: self.on_load_a()),
                ("Load reference trace B…", lambda: self.on_load_ref()),
                (None, None),
                ("Save trace A as CSV…", lambda: self.on_save_a()),
                ("Save reference B as CSV…", lambda: self.on_save_b()),
                ("Export plot as PNG…", lambda: self.on_export_png()),
                (None, None),
                ("Quit", self.close)):
            if label is None:
                m_file.addSeparator()
            else:
                m_file.addAction(label, fn)

        m_inst = mb.addMenu("&Instrument")
        self.act_source = QAction(f"Source: {self.source_name}", self)
        self.act_source.triggered.connect(self._menu_cycle_source)
        m_inst.addAction(self.act_source)

        self.act_mode = QAction(f"Reading mode: {self.mode_name}", self)
        self.act_mode.triggered.connect(self._menu_cycle_mode)
        m_inst.addAction(self.act_mode)

        self.menu_dev = m_inst.addMenu("Device…")
        self.menu_dev.aboutToShow.connect(self._fill_device_menu)
        self.menu_dev.setToolTip(
            wrap_tip("Connected ArgyllCMS-capable instruments (USB devices with a "
            "HID interface). The number is spotread's own instrument "
            "port; if enumeration disagrees, spotread is the final judge."))

        m_inst.addSeparator()
        self.act_hires = QAction("High spectral resolution (-H)", self)
        self.act_hires.setCheckable(True)
        self.act_hires.setChecked(self.high_res)
        self.act_hires.setToolTip(wrap_tip("More spectral samples per scan "
                                  "(recommended; slightly slower)."))
        self.act_hires.toggled.connect(self._on_option_changed)
        m_inst.addAction(self.act_hires)

        self.act_fast = QAction("Fast readings (non-adaptive integration)", self)
        self.act_fast.setCheckable(True)
        self.act_fast.setToolTip(wrap_tip("Skip the adaptive long exposure for dim "
                                 "targets: ~2-3× faster but noisy on dark "
                                 "screens. Takes effect on the next start."))
        self.act_fast.toggled.connect(self._on_option_changed)
        m_inst.addAction(self.act_fast)

        m_inst.addSeparator()
        self.act_clipbl = QAction("Ignore unreliable band (<400 nm)", self)
        self.act_clipbl.setCheckable(True)
        # checked BEFORE the connect: _refresh() during __init__ would
        # hit widgets that do not exist yet (setting it here never
        # emits toggled)
        self.act_clipbl.setChecked(self.clip_unreliable)
        self.act_clipbl.setToolTip(
            wrap_tip("The ColorMunki's calibration cannot vouch for the first "
            "bins of its spectral range: Argyll's driver DUPLICATES "
            "the 400 nm bin below 400 nm (420 nm for reflective "
            "readings) — those values are clones, not measurements.\n"
            "ON: that region is excluded from the chart, every "
            "measurement, the colour data and exports.\n"
            "OFF: the raw instrument values are shown (highlighted in "
            "bordeaux) — nothing else changes."))
        self.act_clipbl.toggled.connect(self._on_unreliable_clip_changed)
        m_inst.addAction(self.act_clipbl)

        m_inst.addSeparator()
        m_inst.addSeparator()
        self.act_forcecal = QAction("Force calibration (ignore saved state)", self)
        self.act_forcecal.setCheckable(True)
        self.act_forcecal.setChecked(self.force_cal)
        self.act_forcecal.setToolTip(
            wrap_tip("ArgyllCMS caches each calibration on this computer "
            "(~/.cache/ArgyllCMS/) and normally reuses it while it is "
            "valid: dark calibration for 1 h, reflective white for 24 h. "
            "With this OFF (spotread -N), USB blips and mode switches "
            "resume measuring WITHOUT the wheel dance. Tick it to demand "
            "a full calibration at every session start instead. spotread "
            "still checks before every reading and asks for calibration "
            "whenever the cache is genuinely stale."))
        self.act_forcecal.toggled.connect(self._on_force_cal_changed)
        m_inst.addAction(self.act_forcecal)

        m_inst.addSeparator()
        self.act_instlog = QAction("Write instrument log", self)
        self.act_instlog.setCheckable(True)
        self.act_instlog.setChecked(self.inst_log)
        self.act_instlog.setToolTip(wrap_tip(
            "Record the full spotread transcript (every line the "
            "instrument prints: readings, spectra, calibration "
            "verdicts, warnings) to "
            "~/spectro-live-logs/instrument-log-*.txt. OFF by default "
            "— nothing is written to disk behind your back. Turning it "
            "ON mid-session starts a file from that moment; OFF stops "
            "further writes and keeps what was already on disk."))
        self.act_instlog.toggled.connect(self._on_option_changed)
        m_inst.addAction(self.act_instlog)
        self.act_csv = QAction("Auto-log every reading to CSV", self)
        self.act_csv.setCheckable(True)
        self.act_csv.setToolTip(wrap_tip("Append each captured spectrum to "
                                "~/spectro-live-logs/session-*.csv"))
        self.act_csv.toggled.connect(self._on_option_changed)
        m_inst.addAction(self.act_csv)

        m_inst.addSeparator()
        act_argyll = QAction("ArgyllCMS spotread path…", self)
        act_argyll.setToolTip(wrap_tip(
            "By default spotread is found on your PATH. This sets an "
            "explicit binary — e.g. a different ArgyllCMS build — "
            "with its absolute path; the running session restarts on "
            "the new binary (Argyll's host calibration cache is "
            "reused if the build reads it)."))
        act_argyll.triggered.connect(self._set_argyll_path)
        m_inst.addAction(act_argyll)
        self.act_argyll_default = QAction(
            "Use default spotread (from PATH)", self)
        self.act_argyll_default.setToolTip(wrap_tip(
            "Forget the explicit path — spotread is again whatever "
            "shutil.which finds (usually /usr/bin/spotread)."))
        self.act_argyll_default.triggered.connect(
            self._set_argyll_path_default)
        m_inst.addAction(self.act_argyll_default)
        self._argyll_menu_update()

        m_view = mb.addMenu("&View")
        grp = QActionGroup(self)
        self._curve_acts = {}
        for key, label, tip in (
                ("lines", "Curve: point-to-point",
                 "Straight segments between the measured samples "
                 "(default)."),
                ("smooth", "Curve: smooth",
                 "Monotone cubic interpolation through the measured "
                 "samples. Overshoot-free by construction: the smooth "
                 "curve never rises above (or below) a reading that "
                 "isn't there. Applies to the current trace instantly."),
                ("steps", "Curve: steps (sample-and-hold)",
                 "Each sample holds its value horizontally until the "
                 "next one — the honest representation of a discrete "
                 "sensor between samples.")):
            a = QAction(label, self)
            a.setCheckable(True)
            a.setChecked(self.curve_style == key)
            a.setToolTip(wrap_tip(tip))
            grp.addAction(a)
            a.triggered.connect(lambda _c, k=key: self._set_curve_style(k))
            m_view.addAction(a)
            self._curve_acts[key] = a

        # vertical-axis units (moved here from the Trace A panel)
        m_units = m_view.addMenu("Units")
        self._unit_acts = {}
        ugrp = QActionGroup(self)
        unit_tips = {
            "Raw (a.u.)":
                "Instrument counts — proportional to the energy the "
                "sensor receives (the radiometric view, uncalibrated "
                "scale).",
            "V(λ)-weighted /nm":
                "Counts weighted with the CIE V(λ) eye-sensitivity "
                "curve — the perceptual weighting that turns radiometry "
                "into photometry. A pure shape view on the instrument's "
                "own arbitrary scale (a.u.), not cd/m²·nm⁻¹; use "
                "Photometric abs. for the calibrated photometric "
                "spectral quantity.",
            "Photometric abs.":
                "The V(λ)-weighted curve scaled per reading so its "
                "integral equals the instrument's own calibrated Y — "
                "true cd/m²·nm⁻¹ (emissive) or lx·nm⁻¹ (ambient); "
                "reflective falls back to the plain (unweighted) "
                "reflectance factor — a reflectance is dimensionless.",
            "Radiometric abs.":
                "The measured spectrum rescaled so its total equals "
                "the radiometric counterpart of the mode's photometric "
                "quantity — W/(sr·m²)·nm⁻¹ radiance (emissive, anchored "
                "to the calibrated Y), W/m²·nm⁻¹ irradiance (ambient, "
                "anchored to lx = π·Y); the chart picks a comfortable "
                "engineering prefix (kW..nW) from the trace's peak, "
                "like the RADIOMETRIC readout. "
                "V(λ) fully undone so deep "
                "blue/red photons carry their real energy. Reflective "
                "falls back to plain reflectance (dimensionless).",
            "Refl. %":
                "×100 scale — natural for reflective readings, plain "
                "percentage elsewhere.",
        }
        for key in ("Raw (a.u.)", "V(λ)-weighted /nm", "Photometric abs.",
                    "Radiometric abs.", "Refl. %"):
            a = QAction(key, self)
            a.setCheckable(True)
            a.setChecked(self.unit_name == key or
                         {"V(λ)-weighted /nm": "Photopic (a.u.)",
                          "Photometric abs.": "Photometric absolute",
                          "Radiometric abs.": "Radiometric absolute",
                          "Refl. %": "Reflectance %"}.get(key) ==
                         self.unit_name)
            a.setToolTip(wrap_tip(unit_tips[key]))
            ugrp.addAction(a)
            a.triggered.connect(lambda _c, k=key:
                                self.cmb_units.setCurrentText(k))
            m_units.addAction(a)
            self._unit_acts[key] = a
        # keep the menu checkmarks in step when the selection changes by
        # any other path (tests drive the combo directly)
        self.cmb_units.currentTextChanged.connect(
            lambda t: [a.setChecked(a.text() == t)
                       for a in self._unit_acts.values()])

        m_view.addSeparator()
        # horizontal axis: wavelength or optical frequency
        grp_x = QActionGroup(self)
        self._x_acts = {}
        for key, label, tip in (
                ("wl", "Axis: wavelength",
                 "Horizontal axis in nanometres (the instrument's own "
                 "scale)."),
                ("freq", "Axis: frequency (THz)",
                 "Horizontal axis as optical frequency f = c/λ in "
                 "terahertz — presentation only: markers, band and all "
                 "analyses stay in nanometres and are converted on "
                 "display."),
                ("ev", "Axis: energy (eV)",
                 "Horizontal axis as photon energy E = hc/λ in "
                 "electronvolts (λ[nm] · E[eV] = 1239.84) — the "
                 "physicist's view of the same spectrum, red on the "
                 "left, violet on the right. Presentation only: "
                 "markers, band and all analyses stay in nanometres "
                 "and are converted on display.")):
            a = QAction(label, self)
            a.setCheckable(True)
            a.setChecked(self.x_mode == key)
            a.setToolTip(wrap_tip(tip))
            grp_x.addAction(a)
            a.triggered.connect(lambda _c, k=key: self._set_x_mode(k))
            m_view.addAction(a)
            self._x_acts[key] = a

        m_view.addSeparator()
        # chromaticity diagram: space selection + true-sRGB mode
        m_chrom = m_view.addMenu("Chromaticity")
        self._space_acts = {}
        sgrp = QActionGroup(self)
        for key, label, tip in (
                ("1931", "Space: CIE 1931 (x, y)",
                 "The classic 2° observer xy chromaticity diagram."),
                ("1976", "Space: CIE 1976 (u′, v′)",
                 "The CIE 1976 UCS u′v′ diagram — a more perceptually "
                 "uniform projection; distances on the plot correspond "
                 "better to perceived colour differences, which is why "
                 "Duv/CCT tolerances are usually quoted in u′v′."),
                ("1960", "Space: CIE 1960 (u, v)",
                 "The CIE 1960 UCS uv diagram — the space ANSI/NIST "
                 "correlated-colour-temperature math is defined on "
                 "(CCT isotherms are straight lines only here).")):
            a = QAction(label, self)
            a.setCheckable(True)
            a.setChecked(self.chrom_space == key)
            a.setToolTip(wrap_tip(tip))
            sgrp.addAction(a)
            a.triggered.connect(lambda _c, k=key:
                                self._set_chrom_space(k))
            m_chrom.addAction(a)
            self._space_acts[key] = a

        m_help = mb.addMenu("&Help")
        m_help.addAction("Instrument log…", self.on_show_log)
        m_help.addAction("Marker help", self.on_help)

    def _menu_cycle_source(self):
        self.source_name = "Simulator" if self.source_name.startswith("Color") \
            else "ColorMunki (spotread)"
        self.act_source.setText(f"Source: {self.source_name}"
                                + ("  (CLI testing only)" if "Simulator" in
                                   self.source_name else ""))
        self.status(f"Source set to {self.source_name} — applies on next start.")

    def _menu_cycle_mode(self):
        opts = list(getattr(self, "allowed_modes",
                            ("Emissive", "Ambient", "Reflective")))
        cur = opts.index(self.mode_name) if self.mode_name in opts else 0
        nxt = opts[(cur + 1) % len(opts)]
        self._select_mode(nxt)           # fires the full mode switch

    # c = 299 792.458 nm·THz (λ[nm] · f[THz] = 299792.458)
    _C_NM_THZ = 299792.458
    # hc/e = 1239.842 nm·eV (λ[nm] · E[eV])
    _HC_NM_EV = 1239.841984

    def _x_vals(self, wl):
        """Displayed x coordinates for an array of wavelengths: nm in
        wavelength mode, THz (f = c/λ) in frequency mode or eV
        (E = hc/λ) in energy mode — both reciprocals, ascending with
        photon energy."""
        if self.x_mode == "freq":
            return self._C_NM_THZ / np.asarray(wl, dtype=float)
        if self.x_mode == "ev":
            return self._HC_NM_EV / np.asarray(wl, dtype=float)
        return np.asarray(wl, dtype=float)

    def _x_back(self, x: float) -> float:
        """Inverse of _x_vals for a scalar: displayed x -> nm."""
        if self.x_mode == "freq":
            return self._C_NM_THZ / float(x)
        if self.x_mode == "ev":
            return self._HC_NM_EV / float(x)
        return float(x)

    def _axis_label(self) -> str:
        """The x-axis title that belongs to the current axis mode."""
        return {"wl": "Wavelength (nm)", "freq": "Frequency (THz)",
                "ev": "Energy (eV)"}[self.x_mode]

    def _set_chrom_space(self, space: str):
        """View → Chromaticity → space: re-project the diagram (and
        every open window) into the selected chromaticity space."""
        if space not in ("1931", "1976", "1960") or \
                space == self.chrom_space:
            return
        self.chrom_space = space
        self._redraw_chrom_all()
        self.status(f"Chromaticity diagram: {chrom_space_name(space)}")

    def _set_x_mode(self, key: str):
        """View → Axis: wavelength / frequency / energy — pure
        presentation (the same data plotted against f = c/λ or
        E = hc/λ; every analysis stays in nanometres)."""
        if key not in ("wl", "freq", "ev") or key == self.x_mode:
            return
        self.x_mode = key
        act = self._x_acts.get(key)
        if act is not None:
            act.setChecked(True)
        self._save_settings()
        xlabel = self._axis_label()
        self.ax.set_xlabel(xlabel, color=FG, fontsize=CHART_AXIS)
        self._refresh()
        self.status(f"Horizontal axis: {xlabel} — presentation only; "
                    "markers and band stay in nanometres.")

    def _set_curve_style(self, key):
        """View → Curve style: pure presentation, applied to the chart
        immediately — the underlying data never changes (and it is NOT
        the smoothing filter, which averages readings)."""
        if key not in ("lines", "smooth", "steps") or key == self.curve_style:
            return
        self.curve_style = key
        act = self._curve_acts.get(key)
        if act is not None:
            act.setChecked(True)     # the QActionGroup clears the others
        self._save_settings()
        self._refresh()
        self.status(f"Curve style: {key} — applied to the current trace.")

    def _mode_tips(self) -> str:
        if sources.is_colormunki(getattr(self, "inst_type", "")):
            return ("Emissive: screens and lamps (slot 3); Ambient: room "
                    "light via the top diffuser (outer slot past OFF); "
                    "Reflective: hold the aperture flat against an opaque "
                    "surface at slot 3 — the instrument flashes its white "
                    "LED and reports the surface colour.")
        return ("Emissive: screens and lamps; Ambient: room light "
                "through the instrument's ambient adapter, if it has "
                "one; Reflective: place the sensor flat against the "
                "surface — the instrument illuminates it itself.")

    def _on_mode_changed(self, text: str):
        """Backwards-compatible alias: the mode combo is gone (the
        reading-mode lamps are the selector now), tests and any stale
        callers funnel through here."""
        self._select_mode(text)

    # the physical light path a mode uses: emissive and reflective share
    # the front aperture; only ambient reads through the diffuser. This
    # is what voids position confirmation — NOT the lamp key (each mode
    # has a lamp of its own now, but two lamps share one aperture).
    @staticmethod
    def _mode_slot(mode_name: str) -> str:
        return "diffuser" if mode_name == "Ambient" else "aperture"

    def _select_mode(self, text: str):
        """Switch the reading mode (the SURFACE / REFLECTIVE / AMBIENT
        lamps are clickable mode selectors). Same physical-slot rules as
        before: emissive and reflective read through the same aperture,
        so switching between them keeps the position confirmation; any
        switch involving the ambient diffuser does not."""
        if text == self.mode_name:
            return
        if text not in getattr(self, "allowed_modes",
                               ("Emissive", "Ambient", "Reflective")):
            self.status(f"Mode {text} is not supported by this "
                        "instrument — selection unchanged.")
            return
        old_slot = self._mode_slot(self.mode_name)
        new_slot = self._mode_slot(text)
        self.mode_name = text
        self.act_mode.setText(f"Reading mode: {self.mode_name}")
        self._save_settings()
        # Only a PHYSICAL slot change voids the confirmation. Emissive
        # and Reflective read through the same aperture (wheel at the
        # measuring position for both), so switching between them keeps
        # the confirmation; any switch involving the ambient diffuser
        # does not. (A USB blip during the switch can't turn the wheel
        # either — see _on_state("restart").)
        if old_slot != new_slot:
            self._pos_confirmed = False
        self._update_position_lamp()
        self._apply_lamps()           # move the blue selected border
        tips = self._mode_tips()
        if self.worker and self.worker.is_alive():
            # spotread needs a different command line per mode: restart
            # the session (one automatic re-calibration) and keep the
            # running/paused state the user had.
            was_running = self.running
            self.worker.stop()
            self.worker = self._make_worker()
            self.worker.start()
            self._start_logs()
            if was_running:
                self.worker.start_continuous()
                self.status(f"Mode: {text} — restarting session "
                            "(auto-calibrating; wheel to 2 briefly). " + tips)
            else:
                self.status(f"Mode: {text} — new session starts with the "
                            "next reading. " + tips)
        else:
            self.status(f"Mode set to {text} — applies on next start. " + tips)

    def _on_unreliable_clip_changed(self, checked):
        """Ignore-unreliable-band toggle: presentation- AND data-level —
        the clipped region leaves the chart and every displayed or
        exported computation (integral, peaks, XYZ/colour) at once."""
        self.clip_unreliable = checked
        self._save_settings()
        lo = unreliable_band_lo(getattr(self, "inst_type", ""),
                                self.mode_name)
        self.status(("Ignoring" if checked else "Showing")
                    + (f" the unreliable <{lo:.0f} nm region"
                       if lo is not None
                       else " data (no unreliable region known for this "
                            "instrument — nothing to clip)")
                    + (", colours recomputed from the trusted band."
                       if (checked and lo is not None) else ""))
        self._refresh()

    def _on_option_changed(self, *_):
        prev_hires, prev_fast = self.high_res, self.fast
        self.high_res = self.act_hires.isChecked()
        self.fast = self.act_fast.isChecked()
        self.csv_log = self.act_csv.isChecked()
        # instrument transcript logging: start/stop the FILE without
        # touching the in-app log panel (which always stays live)
        self.inst_log = self.act_instlog.isChecked()
        if self.inst_log and self.diag_path is None:
            self._start_logs()     # opens the file from this moment
        elif not self.inst_log and self.diag_path is not None:
            self._diag_write(f"=== instrument log disabled by the user "
                             f"{datetime.now():%Y-%m-%d %H:%M:%S} ===\n")
            self.diag_path = None
        self._save_settings()
        # hi-res / fast change the spotread invocation itself: end the
        # running session so the respawn loop restarts it with the new
        # flag (the calibration cache keeps the restart quick)
        if prev_hires != self.high_res or prev_fast != self.fast:
            w = self.worker
            if w is not None and getattr(w, "is_alive", lambda: False)():
                # the respawn loop builds the spotread command from the
                # WORKER's copy of the flags — publish before restarting
                # (otherwise the toggle silently does nothing)
                w.high_res = self.high_res
                w.fast = self.fast
                fn = getattr(w, "restart_session", None)
                if fn is not None and self.running:
                    fn()
                    self.status("Instrument option changed — restarting "
                                "session (calibration cache reused where "
                                "valid)…")

    def _on_force_cal_changed(self, checked):
        """Force-calibration toggle: takes effect immediately by ending
        the current spotread run — the respawn loop restarts it with the
        new setting (forcing the calibration dance when ON)."""
        self.force_cal = checked
        self._save_settings()
        w = self.worker
        if isinstance(w, sources.SpotreadWorker):
            w.force_cal = checked
            if checked and w.is_alive():
                w.restart_session()
                self.status("Forcing calibration — restart the instrument "
                            "session; turn the wheel to position 2 when the "
                            "CALIBRATION lamp lights up.")
                return
        self.status("Force calibration "
                    + ("ON" if checked else "OFF — reusing Argyll's saved "
                       "calibration where valid (dark 1 h, white 24 h).")
                    + (" Applies at the next session start."
                       if not checked else ""))

    # ===================================================================
    # toolbar: big instrument-style buttons
    # ===================================================================
    def _std_icon(self, kind: str):
        """A hand-drawn monochrome white icon (line art with alpha) —
        guaranteed to render on every system and matching the dark
        theme, unlike both font glyphs (▶/⏺/❄/💾, font-dependent) and
        the QStyle standard icons (multi-coloured, clash badly)."""
        return mono_icon(kind)

    def _toolbar(self):
        tb = QToolBar("Measure")
        tb.setMovable(False)
        tb.setIconSize(tb.iconSize())
        self.addToolBar(tb)

        std_icon = self._std_icon
        self.btn_start = QPushButton("RUN")
        self.btn_start.setIcon(std_icon("play"))
        self.btn_start.setObjectName("btnStart")
        self.btn_start.setToolTip(wrap_tip("<b>RUN / STOP</b><br>"
                                  "Begin continuous readings. The "
                                  "instrument session and its calibration "
                                  "start automatically on launch, so RUN "
                                  "only begins the readings themselves.<br>"
                                  "Wheel slot for readings depends on the "
                                  "mode: 3 = front aperture (emissive and "
                                  "reflective), the outer detent beyond "
                                  "OFF = ambient diffuser.<br>"
                                  "Pause keeps the instrument session and "
                                  "its calibration alive."))
        self.btn_start.clicked.connect(self.on_start_stop)
        tb.addWidget(self.btn_start)

        self.btn_single = QPushButton("SINGLE")
        self.btn_single.setIcon(std_icon("single"))
        self.btn_single.setObjectName("btnSingle")
        self.btn_single.setToolTip(wrap_tip("<b>SINGLE READING</b><br>"
                                   "Take exactly one measurement with the "
                                   "current instrument session (no "
                                   "auto-repeat)."))
        self.btn_single.clicked.connect(self.on_single)
        for b in (self.btn_start, self.btn_single):
            b.setProperty("tool", True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
        tb.addWidget(self.btn_single)

        # center the data-group buttons (Zero … Save A) on the WHOLE
        # WINDOW. The toolbar places ONE container widget after
        # RUN + SINGLE; inside it a plain QHBoxLayout holds a fixed-
        # width PAD, the buttons, and a trailing stretch. The pad is
        # the correction knob: it must grow by exactly half the width
        # of the fixed RUN + SINGLE buttons, which pulls the group
        # left onto the window's centre line. _center_tool_group()
        # recomputes the pad width on every resize as a pure DELTA off
        # the live layout — one measurement, one exact correction, no
        # feedback to wedge. (The earlier mechanism pinned QToolBar
        # spacer minimums from measured geometry instead; QToolBar's
        # space distribution is opaque and stateful — in the starved
        # regime the pins wedged the group against one edge of the
        # window and it STAYED there after growing.)
        group_wrap = QWidget()
        _paint_only(group_wrap, "tbGroup", "background: transparent;")
        group_wrap.setSizePolicy(QSizePolicy.Policy.Expanding,
                                 QSizePolicy.Policy.Preferred)
        gh = QHBoxLayout(group_wrap)
        gh.setContentsMargins(0, 0, 0, 0)
        pad = QWidget()
        _paint_only(pad, "tbGroupPad", "background: transparent;")
        pad.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents,
                         True)
        gh.addWidget(pad)
        self.tb_group_pad = pad

        for text, tip, fn, pix in (
                ("Zero", "<b>Capture zero (dark)</b><br>Store the current "
                 "reading as the new zero/background and subtract it from "
                 "all following readings. Pressing it again re-zeros; "
                 "only Clear zero removes it.",
                 self.on_capture_zero, "zero"),
                ("Clear zero", "Remove the stored zero/background.",
                 self.on_clear_zero, "clear-zero"),
                ("Freeze B", "<b>Freeze reference (B)</b><br>Copy the "
                 "current trace to B for overlay / difference / ratio "
                 "comparison.", self.on_freeze_ref, "freeze"),
                ("Clear B", "Remove the frozen reference trace B.",
                 self.on_clear_ref, "clear-b"),
                ("Load B…", "Load a reference trace B from a CSV or "
                 "Argyll (.sp/.ti3) file.", self.on_load_ref,
                 "load"),
                ("Save A…", "Save the current trace A to a CSV file.",
                 self.on_save_a, "save")):
            b = btn(text, tip, fn)
            b.setIcon(std_icon(pix))
            b.setProperty("tool", True)
            gh.addWidget(b)
            if "zero" in text.lower() and "clear" not in text.lower():
                self.btn_zero = b
            elif "clear zero" in text.lower():
                self.btn_zero_clear = b
            last = b
        # the group's two ends, for tests / geometry probes
        self.tb_group_ends = (self.btn_zero, last)
        gh.addStretch(1)
        tb.addWidget(group_wrap)

    def _center_tool_group(self):
        """Keep the Zero … Save A group centered on the WINDOW: the
        group lives inside a toolbar container with a fixed-width PAD
        to its left, so the pad must carry exactly half the width of
        the fixed RUN + SINGLE buttons, pulling the group onto the
        window's centre line. The correction is computed as a pure
        DELTA off the live layout: one measurement, one exact width —
        a QHBoxLayout shifts the group 1:1 with the pad, so there is
        no feedback loop to wedge. Recomputed on every resize (and
        the slow blink tick as a safety net): a squished window just
        clamps the pad to 0 (group sits compact after RUN + SINGLE —
        nothing fits anyway), and the first resize afterwards
        re-centres it, because nothing is carried over from the
        starved state. (An earlier version pinned QToolBar spacer
        minimums instead — QToolBarLayout's space distribution is
        opaque and stateful, and the pins wedged the group against
        the left edge for good after a squish.)"""
        pad = getattr(self, "tb_group_pad", None)
        ends = getattr(self, "tb_group_ends", None)
        if pad is None or not ends:
            return
        a, b = ends
        wrap = pad.parentWidget()
        if wrap is None:
            return
        tbw = self.width()               # the toolbar spans the window
        # the group's left edge and total width in toolbar
        # coordinates — the toolbar may not have been re-laid out
        # yet inside resizeEvent, but the group's position relative to
        # the toolbar does not depend on the toolbar's width (its
        # container starts right after the fixed buttons)
        ga = a.mapTo(tb := wrap.parentWidget(), a.rect().topLeft()).x()
        gb = b.mapTo(tb, b.rect().topRight()).x()
        group = max(0, gb - ga)
        target = tbw // 2 - group // 2   # wanted left edge, centred
        delta = target - ga              # pixels to shift the group
        # clamp: the pad cannot shrink below 0, and it must leave room
        # for the group itself plus the layout's spacing inside the
        # container (group_wrap starts after RUN + SINGLE)
        # the base is the pad's FIXED width (what the next layout pass
        # will apply), not pad.width() — the applied geometry can lag
        # a pending layout inside resizeEvent
        w = pad.minimumWidth() + delta
        room = wrap.width() - group - 24 if wrap.width() > group + 24 \
            else 0
        w = max(0, min(w, room))
        if w != pad.minimumWidth():
            pad.setFixedWidth(w)
        # force the container's layout at its FRESH width: a setFixedWidth
        # only posts a LayoutRequest, so after a big shrink the old pad
        # width would keep the group clipped past the window edge until
        # some unrelated layout pass fixes it
        gh = wrap.layout()
        if gh is not None:
            xw = wrap.mapTo(tb, wrap.rect().topLeft()).x()
            gh.setGeometry(QRect(0, 0, tbw - xw, wrap.height()))

    # ===================================================================
    # left: settings column (scrollable)
    # ===================================================================
    def _settings_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setMinimumWidth(312)
        scroll.setMaximumWidth(345)
        body = QWidget()
        # the panel body matches the toolbar / status-bar colour,
        # unlike the value-title strips which stay transparent
        # (_paint_only: a bare rule here repaints every tooltip shown
        # from this panel with the panel body colour too)
        _paint_only(body, "settingsBody", f"background: {PANEL_BODY};")
        body.setSizePolicy(QSizePolicy.Policy.Preferred,
                           QSizePolicy.Policy.Preferred)
        v = QVBoxLayout(body)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(10)

        # --- measurement mode -------------------------------------------
        # (the mode selector is the SURFACE / REFLECTIVE / AMBIENT lamp
        # strip at the top of the chart — see _center_panel)

        # --- live trace processing -------------------------------------
        g = QGroupBox("Trace A")
        gl = QGridLayout(g)
        gl.setVerticalSpacing(6)
        gl.setHorizontalSpacing(8)
        gl.setColumnStretch(1, 1)
        r = 0
        # vertical-axis unit selection moved to View → Units (menu);
        # the combo stays as the shared backing store the menu and the
        # tests drive, but no longer occupies panel space
        # (constructed in __init__, before the menubar)
        self.cmb_units.addItems(["Raw (a.u.)", "V(λ)-weighted /nm",
                                 "Photometric abs.", "Radiometric abs.",
                                 "Refl. %"])
        self.cmb_units.setToolTip(
            wrap_tip("Vertical-axis unit of trace A — every option is always "
            "available, and the photometric units follow the reading "
            "mode:<br>"
            "<b>Raw</b>: instrument counts — proportional to the energy "
            "the sensor receives (the radiometric view, uncalibrated "
            "scale).<br>"
            "<b>V(λ)-weighted</b>: counts weighted with the CIE V(λ) "
            "eye-sensitivity curve — the perceptual weighting that turns "
            "radiometry into photometry (deep blue and red are "
            "discounted). A pure <i>shape</i> view on the instrument's "
            "own arbitrary scale (a.u.), NOT cd/m²·nm⁻¹: the raw counts "
            "carry no absolute calibration; use <b>Photometric abs.</b> "
            "for the calibrated photometric spectral quantity.<br>"
            "<b>Photometric abs.</b>: the V(λ)-weighted curve scaled "
            "per reading so its integral equals the instrument's own "
            "calibrated Y — true <i>cd/m²·nm⁻¹</i> (emissive) or "
            "<i>lx·nm⁻¹</i> (ambient); reflective falls back to the "
            "plain reflectance factor (a reflectance is dimensionless — "
            "no V(λ) weighting applies to a surface). The band integral "
            "then reads real luminance/illuminance in the band.<br>"
            "<b>Radiometric abs.</b>: the measured spectrum rescaled "
            "so its total equals the radiometric counterpart of the "
            "mode's photometric quantity — radiance "
            "<i>W/(sr·m²)·nm⁻¹</i> (emissive), irradiance "
            "<i>W/m²·nm⁻¹</i> (ambient; the chart then picks a "
            "comfortable engineering prefix, kW..nW, from the peak — "
            "display-range light lands in mW), undoing "
            "the V(λ) weighting so "
            "every photon carries its real energy; reflective falls "
            "back to plain reflectance (a surface has no radiometric "
            "quantity — reflectance is dimensionless).<br>"
            "<b>Refl. %</b>: ×100 scale — natural for reflective "
            "readings, plain percentage elsewhere."))
        self.cmb_units.currentTextChanged.connect(
            lambda t: (setattr(self, "unit_name",
                               {"V(λ)-weighted /nm": "Photopic (a.u.)",
                                "Photometric abs.":
                                    "Photometric absolute",
                                "Radiometric abs.":
                                    "Radiometric absolute",
                                "Refl. %": "Reflectance %"}
                               .get(t, t)),
                       self._refresh()))
        # NOTE: cmb_units is not added to any layout — the View → Units
        # menu drives it (it stays the single source of truth)
        self.chk_avg = QCheckBox("Averaging")
        self.chk_avg.setToolTip(wrap_tip("Average the last N readings to reduce "
                                "measurement noise."))
        self.chk_avg.toggled.connect(self._refresh_settings)
        gl.addWidget(self.chk_avg, r, 0)
        self.spin_avg = QSpinBox()
        self.spin_avg.setRange(1, 200)
        self.spin_avg.setValue(self.avg_n)
        self.spin_avg.setToolTip(wrap_tip("Number of readings averaged."))
        self.spin_avg.valueChanged.connect(self._refresh_settings)
        self.spin_avg.valueChanged.connect(lambda *_: self._refresh())
        gl.addWidget(self.spin_avg, r, 1)
        r += 1
        self.rb_roll = QRadioButton("Rolling (last N)")
        self.rb_cum = QRadioButton("Cumulative (all)")
        self.rb_roll.setChecked(True)
        for rb, tip in ((self.rb_roll,
                         "Average only the newest N readings (moving window)."),
                        (self.rb_cum, "Average every reading taken so far.")):
            rb.setToolTip(wrap_tip(tip))
            rb.toggled.connect(self._refresh_settings)
        gl.addWidget(self.rb_roll, r, 0, 1, 2)
        gl.addWidget(self.rb_cum, r + 1, 0, 1, 2)
        r += 2
        self.chk_smooth = QCheckBox("Smoothing")
        self.chk_smooth.setToolTip(wrap_tip("Move the trace through a moving-average "
                                   "window: hides sensor noise, may "
                                   "flatten narrow peaks."))
        self.chk_smooth.toggled.connect(self._refresh)
        gl.addWidget(self.chk_smooth, r, 0)
        self.spin_smooth = QSpinBox()
        self.spin_smooth.setRange(3, 99)
        self.spin_smooth.setSingleStep(2)
        self.spin_smooth.setValue(self.smooth_n)
        self.spin_smooth.setToolTip(wrap_tip("Window width in samples (odd number). "
                                    "106 samples cover 380–730 nm."))
        self.spin_smooth.valueChanged.connect(lambda *_: self._refresh())
        gl.addWidget(self.spin_smooth, r, 1)
        v.addWidget(g)

        # --- zero / reference ------------------------------------------
        g = QGroupBox("Zero & ref. B")
        gl = QGridLayout(g)
        gl.setVerticalSpacing(6)
        gl.setColumnStretch(1, 1)
        gl.addWidget(QLabel("Compare"), 0, 0)
        self.cmb_view = QComboBox()
        self.cmb_view.addItems(["Overlay (A + B)", "Difference (A−B)",
                                "Ratio A/B %"])
        self.cmb_view.setToolTip(wrap_tip("<b>Overlay</b>: draw A and frozen B "
                                 "together.<br><b>Difference</b>: A−B.<br>"
                                 "<b>Ratio</b>: A/B × 100%.<br>"
                                 "Without a frozen B, trace A is shown."))
        self.cmb_view.currentTextChanged.connect(
            lambda t: (setattr(self, "view_name", t), self._refresh()))
        gl.addWidget(self.cmb_view, 0, 1)
        lbl = QLabel("B is captured with FREEZE B on the toolbar, or loaded "
                     "from a file. Difference shows A−B; without B the live "
                     "trace A stays visible.")
        lbl.setProperty("hint", True)
        lbl.setWordWrap(True)
        lbl.setSizePolicy(QSizePolicy.Policy.Ignored,
                          QSizePolicy.Policy.Minimum)
        gl.addWidget(lbl, 1, 0, 1, 2)
        v.addWidget(g)

        # --- band / markers ---------------------------------------------
        g = QGroupBox("Band & markers")
        gl = QGridLayout(g)
        gl.setVerticalSpacing(6)
        gl.setColumnStretch(1, 1)
        gl.addWidget(QLabel("from (nm)"), 0, 0)
        self.spin_lo = QDoubleSpinBox()
        self.spin_lo.setRange(self.x_lo, self.x_hi)
        self.spin_lo.setDecimals(1)
        self.spin_lo.setValue(self.band_lo)
        self.spin_lo.setToolTip(wrap_tip("Lower wavelength of the integration band "
                                "(shaded area)."))
        self.spin_lo.valueChanged.connect(self._refresh)
        gl.addWidget(self.spin_lo, 0, 1)
        gl.addWidget(QLabel("to (nm)"), 1, 0)
        self.spin_hi = QDoubleSpinBox()
        self.spin_hi.setRange(self.x_lo, self.x_hi)
        self.spin_hi.setDecimals(1)
        self.spin_hi.setValue(self.band_hi)
        self.spin_hi.setToolTip(wrap_tip("Upper wavelength of the integration band."))
        self.spin_hi.valueChanged.connect(self._refresh)
        gl.addWidget(self.spin_hi, 1, 1)
        self.chk_mband = QCheckBox("Band = markers")
        self.chk_mband.setToolTip(wrap_tip("Instead of the numeric band, integrate "
                                  "between the two click markers on the "
                                  "spectrum plot."))
        self.chk_mband.toggled.connect(self._refresh)
        gl.addWidget(self.chk_mband, 2, 0, 1, 2)
        v.addWidget(g)

        # CIE 1931 chromaticity diagram (true colours) — kept in the left
        # panel so the right side stays a pure numeric readout
        g = QGroupBox("Chromaticity")
        v3 = QVBoxLayout(g)
        v3.setContentsMargins(4, 2, 4, 4)
        v3.setSpacing(4)
        self.chrom_fig = Figure(figsize=(3.4, 2.9), dpi=100)
        self.chrom_handles = self._draw_chromaticity(self.chrom_fig)
        self.chrom_canvas = FigureCanvasQTAgg(self.chrom_fig)
        self.chrom_canvas.setToolTip(
            wrap_tip("CIE 1931 (2°) xy chromaticity diagram in true colours. The "
            "red dot is the CURRENT measurement; the outlined triangles "
            "are the sRGB (blue), Adobe RGB (green), Display P3 "
            "(orange) and Rec. 2020 (yellow) gamuts around the D65 "
            "white point (white +)."))
        self.chrom_canvas.setMinimumHeight(240)
        v3.addWidget(self.chrom_canvas, 1)
        self._chrom_windows = []      # detached (enlarged) copies
        b = btn("⧉  Open in a window", "Show the chromaticity diagram in "
                "a resizable window, so the triangles are easier to read.",
                self._open_chrom_window)
        v3.addWidget(b)
        v.addWidget(g)

        hint = QLabel(
            "Tip: left-click on the spectrum plot sets marker M1, right-click "
            "sets M2. The optional instrument log (Instrument → Write "
            "instrument log) and CSV auto-log also go to "
            "~/spectro-live-logs/ — both off by default.")
        hint.setProperty("hint", True)
        hint.setWordWrap(True)
        hint.setSizePolicy(QSizePolicy.Policy.Ignored,
                           QSizePolicy.Policy.Minimum)
        v.addWidget(hint)
        v.addStretch(1)
        scroll.setWidget(body)
        return scroll

    def _refresh_settings(self, *_):
        self.avg_on = self.chk_avg.isChecked()
        self.avg_n = self.spin_avg.value()
        self.avg_rolling = self.rb_roll.isChecked()
        # level-triggered on the checkbox state (the old edge-trigger
        # missed paths that landed unchecked without a transition,
        # leaving the radios interactive while averaging was off)
        for b in (self.rb_roll, self.rb_cum, self.spin_avg):
            b.setEnabled(self.avg_on)
        self._refresh()

    # ===================================================================
    # centre: annunciator strip + spectral plot
    # ===================================================================
    def _fit_chart_margins(self, init_only: bool = False):
        """Chart margins fitted to the ACTUAL rendered axis text.

        The root cause of the clipped labels was guessed constants:
        any fixed fraction or fixed pixel count is wrong for some
        window size / unit label. So instead of guessing, this
        measures the real bounding boxes of the tick labels and axis
        labels after a render and sizes the left/bottom gutters to
        exactly what they need (plus a small pad). Called on every
        canvas resize and after every redraw (the y label changes
        with the unit selection).
        """
        w = max(int(self.canvas.width()) if not init_only else 900, 1)
        h = max(int(self.canvas.height()) if not init_only else 600, 1)
        self.fig.subplots_adjust(right=0.998, top=0.998)
        left_px, bottom_px = 45.0, 55.0            # sane fallbacks
        try:
            self.fig.canvas.draw()                 # refresh text bboxes
            r = self.fig.canvas.get_renderer()
            yw = max((t.get_window_extent(r).width
                      for t in self.ax.get_yticklabels()
                      if t.get_text()), default=0.0)
            if self.ax.get_ylabel():
                yw += self.ax.yaxis.label.get_window_extent(r).width + 6.0
            left_px = yw + 12.0
            xh = max((t.get_window_extent(r).height
                      for t in self.ax.get_xticklabels()
                      if t.get_text()), default=0.0)
            if self.ax.get_xlabel():
                xh += self.ax.xaxis.label.get_window_extent(r).height + 4.0
            bottom_px = xh + 10.0
        except Exception:
            pass
        sp = self.fig.subplotpars
        left = min(max(left_px / w, 0.06), 0.30)
        bottom = min(max(bottom_px / h, 0.08), 0.35)
        if abs(sp.left - left) > 0.001 or abs(sp.bottom - bottom) > 0.001:
            self.fig.subplots_adjust(left=left, bottom=bottom)
            self.fig.canvas.draw_idle()
        self._align_satbar()

    def _align_satbar(self):
        """Pin the saturation bar's top and bottom to the chart's axes
        box AND place its floating overlay column. The bar is NOT in
        any layout: as a manual child of the centre panel it spans the
        lamp-strip row as well (a transparent column to the right of
        the lamps), which costs the layout ZERO minimum width — a
        layout cell would have demanded ~90 px next to the strip and
        broken the 1280 px window guarantee. The paint is transparent
        away from the track, so the strip shows clear of it; the
        headroom above the track is what lets the 100 % label sit
        centred on its tick, fully inside the widget. Recomputed on
        every canvas resize and chart-margin refit."""
        h = float(self.canvas.height()) if hasattr(self, "canvas") else 0.0
        wrap = getattr(self, "_centre_wrap", None)
        if h < 40 or wrap is None or not hasattr(self, "satbar"):
            return
        gw = 92                     # overlay column width (track + gutter)
        self.satbar.setGeometry(wrap.width() - gw, 0, gw, wrap.height())
        self.satbar.raise_()
        sp = self.fig.subplotpars
        # distance from the overlay's top (centre-panel top) DOWN to the
        # canvas: the strip row + spacing. Axes top = that + axes inset.
        from PySide6.QtCore import QPoint
        top_off = max(self.canvas.mapTo(wrap, QPoint(0, 0)).y(), 0)
        self.satbar.align_to((1.0 - sp.top) * h + top_off, sp.bottom * h)

    def _center_panel(self):
        wrap = QWidget()
        # NOTE: this bare rule is deliberately NOT scoped with
        # _paint_only — it is LOAD-BEARING for the descendants: the
        # floating SatBar overlay and the chart gutter rely on the
        # PROPAGATED 'background: transparent' beating the global dark
        # QWidget rule (scoping it regress-tested as a mis-rendered
        # bottom sliver of the sat bar). No tooltip side-effect here:
        # this wrap carries no tooltip and the tooltip-triggering
        # children (lamps) have their own scoped styles.
        wrap.setStyleSheet("background: transparent;")
        self._centre_wrap = wrap
        v = QVBoxLayout(wrap)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)

        # annunciator strip (aircraft master-warning style): the lamps
        # are always legible but unlit when inactive, and light up solid
        # to instruct. CALIBRATION is lit while calibration is needed or
        # running; SURFACE/AMBIENT POSITION are lit while the instrument
        # waits for the correct reading slot; all go dark otherwise.
        strip = QWidget()
        _paint_only(strip, "lampStrip", "background: transparent;")
        h = QHBoxLayout(strip)
        h.setContentsMargins(0, 2, 0, 2)
        h.setSpacing(8)
        self.lamps: dict[str, QLabel] = {}
        # The three reading-mode lamps double as the mode selector (the
        # left-panel combo is gone): SURFACE = emissive, REFLECTIVE,
        # AMBIENT — the selected one carries a blue border and lights
        # amber only for itself while its position is unconfirmed.
        mode_tips = {
            "surface": ("Emissive: screens and lamps measured through the "
                        "front aperture (ColorMunki wheel at 3) — the "
                        "instrument's own light-through-the-aperture "
                        "reading.<br><br><b>Click to select Emissive "
                        "mode.</b>"),
            "reflective": ("Reflective: the instrument flashes its built-in "
                           "white LED and measures the light bouncing off an "
                           "opaque surface held flush against the front "
                           "aperture (ColorMunki wheel at 3) — this reports "
                           "the colour of the surface itself (wall, print, "
                           "paint).<br><br><b>Click to select Reflective "
                           "mode.</b>"),
            "ambient": ("Ambient: overall room light entering the "
                        "instrument's ambient opening — the top diffuser on "
                        "the ColorMunki (outer detent just past "
                        "OFF).<br><br><b>Click to select Ambient mode.</b>"),
        }
        self._LAMP_MODES = {"surface": "Emissive", "reflective": "Reflective",
                            "ambient": "Ambient"}
        for key, label in (("cal", "CALIBRATION"),
                           ("surface", "EMISSIVE"),
                           ("reflective", "REFLECTIVE"),
                           ("ambient", "AMBIENT")):
            lampw = LampLabel(label)
            lampw.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lampw.setFixedHeight(36)
            lampw.setCursor(Qt.CursorShape.PointingHandCursor)
            self.lamps[key] = lampw
            h.addWidget(lampw)
        # breathing room between the CALIBRATION annunciator and the
        # three reading-mode lamps (they are two different families:
        # an instruction vs a selector)
        h.insertSpacing(1, 22)
        for key, tip in mode_tips.items():
            self.lamps[key].setToolTip(wrap_tip(tip))
        # clicking a mode lamp selects that reading mode (handled in
        # _select_mode); the CALIBRATION lamp demands a recalibration
        for key, lamp in self.lamps.items():
            if key == "cal":
                lamp.setToolTip(
                    wrap_tip("<b>Click to recalibrate</b><br>Ends and restarts the "
                    "instrument session once WITHOUT Argyll's host "
                    "calibration cache, so spotread demands a fresh "
                    "calibration — which the app then runs for every "
                    "reading mode in one wheel-at-2 effort. Normal cache "
                    "reuse resumes afterwards."))
                lamp.clicked.connect(self._on_cal_lamp_clicked)
            else:
                lamp.clicked.connect(
                    lambda _=False, k=key: self._select_mode(
                        self._LAMP_MODES[k]))
        h.addStretch(1)
        v.addWidget(strip)
        self._lamp_state = {"cal": "off", "surface": "off",
                            "reflective": "off", "ambient": "off"}
        self._apply_lamps()
        # slow shared timer: saturation-bar flashing at clip level and
        # the 1 h fade of the calibration lamp's green border
        self.blink_timer = QTimer(self)
        self.blink_timer.timeout.connect(self._blink_tick)
        self.blink_timer.start(250)

        # matplotlib figure
        self.fig = Figure(figsize=(9, 6), dpi=100)
        self.fig.set_facecolor(BG)
        self.ax = self.fig.add_subplot(111)
        # tight chart: no wasted space above/right of the plot. The
        # left/bottom gutters are set in PIXELS via _fit_chart_margins
        # whenever the canvas resizes, so the axis text never clips no
        # matter how small the window gets (fractions alone clip: a
        # 2 % margin is fine on a 900 px canvas and cuts the labels
        # off on a 300 px one).
        self._fit_chart_margins(init_only=True)
        self.ax.set_facecolor(BG)
        for sp in self.ax.spines.values():
            sp.set_color(BORDER)
        self.ax.tick_params(colors=MUTED, labelsize=10)
        self.ax.grid(True, color=BORDER, lw=0.5, alpha=0.5)
        self.ax.set_xlabel("Wavelength (nm)", color=FG, fontsize=CHART_AXIS)
        self.ax.set_ylabel("Intensity (a.u.)", color=FG, fontsize=CHART_AXIS)
        self.ax.set_xlim(self.x_lo, self.x_hi)

        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setParent(self)
        # root-cause anti-clip: whenever the canvas is resized (the
        # window can be squeezed a lot), recompute the chart gutters
        # in pixels so the axis labels always fit inside the canvas
        self.canvas.installEventFilter(self)

        # saturation annunciator bar, floating right of the chart and
        # spanning the lamp-strip row too (transparent above the
        # track — see _align_satbar for why it is NOT in a layout)
        self.satbar = SatBar(wrap)
        self.satbar.setToolTip(
            wrap_tip("<b>Sensor headroom</b> — how close the busiest "
            "sensor channel is to the END of its linear range "
            "(log scale). 100 % = the sensor's analog knee "
            "(~255 printed units on the ColorMunki): below it the "
            "reading is linear; at or above it Argyll still accepts "
            "the reading, but the brightest channels are no longer "
            "quantitative — move the instrument back from the light "
            "source. The scale is the same in adaptive and fast mode "
            "(Argyll divides each reading's counts by its own "
            "integration time — munki_imp.c). The estimate exists "
            "only for instrument families with a vetted model "
            "(ColorMunki); anything else honestly reads n/a. "
            "Reflective readings: n/a. spotread's own 'Sensor is "
            "saturated' verdict (raw counts over the per-instrument "
            "EEPROM satlimit, checked before any scaling) is the "
            "authority and pins the bar flashing red. Green = "
            "comfortable headroom, amber = getting close, steady/flashing "
            "red = at/over the knee."))
        plotrow = QWidget()
        _paint_only(plotrow, "plotRow", "background: transparent;")
        ph = QHBoxLayout(plotrow)
        # the right margin reserves the floating satbar overlay's gutter
        # without putting it in the layout (a layout cell would grow the
        # window's minimum width; see _align_satbar)
        ph.setContentsMargins(0, 0, 94, 0)
        ph.setSpacing(2)
        ph.addWidget(self.canvas, 1)
        v.addWidget(plotrow, 1)
        # first alignment pass (a Resize event will re-run it whenever
        # the window or the measured bottom gutter changes)
        self._align_satbar()

        self.line_a, = self.ax.plot([], [], color=ACCENT, lw=2.0, zorder=4)
        self.line_b, = self.ax.plot([], [], color=REF, lw=1.6, alpha=1.0, zorder=4)
        self.peak_marks, = self.ax.plot(
            [], [], linestyle="none", marker="v", markersize=9,
            markerfacecolor=ACCENT, markeredgecolor=FG,
            markeredgewidth=0.8, zorder=5)
        from .processing import rainbow_palette
        self.rainbow = self.ax.imshow(
            rainbow_palette(self.x_lo, self.x_hi), aspect="auto",
            origin="lower", extent=(self.x_lo, self.x_hi, 0.0, 1.0),
            transform=self.ax.get_xaxis_transform(),
            interpolation="bilinear", alpha=0.35, zorder=-5)
        self.rainbow_clip = PathPatch(
            Path([(self.x_lo, 0.0), (self.x_lo, 0.0), (self.x_hi, 0.0),
                  (self.x_hi, 0.0), (self.x_lo, 0.0)]),
            transform=self.ax.get_xaxis_transform(), facecolor="none",
            edgecolor="none")
        self.ax.add_patch(self.rainbow_clip)
        self.rainbow.set_clip_path(self.rainbow_clip)
        # bordeaux marker for the instrument's UNRELIABLE first bins
        # (ColorMunki: Argyll clones the 400 nm bin below 400 nm —
        # 420 reflective; those values are duplicates, not data).
        # Created/removed per redraw in _update_unreliable_span.
        self.unreliable_span = None
        self.marker_lines = [
            self.ax.axvline(self.marker1, color=ACCENT, lw=1.2, ls="--",
                            alpha=0.9, zorder=2),
            self.ax.axvline(self.marker2, color=ACCENT, lw=1.2, ls=":",
                            alpha=0.9, zorder=2)]
        self.bandspan = None
        self.band_edges = []
        self.peak_annots = []

        self.cid = self.canvas.mpl_connect(
            "button_press_event", self._on_plot_click)

        # connection problems report in the bottom status bar only
        # (the plot stays clean)
        return wrap

    def eventFilter(self, obj, ev):
        if obj is self.canvas and \
                ev.type() == ev.Type.Resize:
            # keep the axis text inside the canvas at any window size
            # (_fit_chart_margins renders, measures and re-fits)
            self._fit_chart_margins()
        return super().eventFilter(obj, ev)

    def _on_plot_click(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        wl = self._x_back(float(event.xdata))   # axis may show THz
        if event.button in (2, 3):
            self.marker2 = wl
        else:
            self.marker1 = wl
        self._refresh()

    # ===================================================================
    # right: measurements column
    # ===================================================================
    def _readout_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(372)
        scroll.setMaximumWidth(410)
        body = QWidget()
        # match the toolbar / status-bar colour — the same panel look
        # as the left column (_paint_only so the colour stays away from
        # descendants and their tooltips)
        _paint_only(body, "readoutBody", f"background: {PANEL_BODY};")
        v = QVBoxLayout(body)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(10)

        g = QGroupBox("Measurements")
        gl = QGridLayout(g)
        gl.setVerticalSpacing(4)
        gl.setHorizontalSpacing(8)
        self.val_integral = value_box(
            gl, 1, "Band integral  ∫",
            "Integral of trace A over the analysis band "
            "(shaded area), in the unit selected on the left.")
        self.val_peak = value_box(
            gl, 2, "Peak / FWHM / centroid",
            "Band max.: peak wavelength, emission value at the peak, "
            "full width at half maximum, and energy centroid.")
        self.val_markers = value_box(
            gl, 3, "Markers M1 / M2",
            "Values of the click markers: wavelength, intensity at M1 and "
            "M2, their distance Δλ and the intensity difference Δ.")
        self.val_photometric = value_box(
            gl, 4, "Photometric Y",
            "The Y tristimulus value in the unit that fits this reading "
            "mode: luminance cd/m² for an emitting screen, illuminance "
            "lx for ambient light (E = π·Y, Argyll's Lambertian-field "
            "model), reflectance factor % for a surface.")
        self.val_photometric.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self.val_radiometric = value_box(
            gl, 5, "Radiometric",
            "The radiometric counterpart of the "
            "photometric value above, derived from the measured "
            "spectral shape and the maximum spectral luminous efficacy "
            "K_m = 683 lm/W: <br><b>radiance W/(sr·m²)</b> for an "
            "emitting screen (pairs with luminance cd/m²), "
            "<b>irradiance W/m²</b> for ambient light through the "
            "diffuser (pairs with illuminance lx).<br>The absolute "
            "scale is anchored to the instrument's own calibrated Y "
            "(the printed spectrum alone is in arbitrary units — the "
            "spectrum only supplies the shape that undoes the photopic "
            "weighting). n/a for surfaces (reflectance is "
            "dimensionless) and for XYZ-only colorimeters "
            "(no spectral shape).")
        self.val_radiometric.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self.val_cct = value_box(
            gl, 6, "Correlated colour temp.",
            "CCT: the temperature of the Planckian radiator closest to "
            "this colour. Universally quoted for light sources — '2700 K "
            "warm white', '6500 K daylight'.")
        self.val_cct.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self.val_ri = value_box(
            gl, 7, "Colour rendering",
            "Colour rendering of the light source, computed by spotread "
            "from its spectrum against a reference of the same CCT — "
            "Argyll's own numbers with Argyll's own validity flags "
            "(when spotread doesn't provide them, spectro-v falls back "
            "to computing Ra/R9 from the spectrum itself; Rf/Rg come "
            "only from Argyll). "
            "<br><b>Ra</b> (CIE): the classic bulb-box general index "
            "(R1–R8 average), R9 = saturated red — cheap LEDs hide bad "
            "R9; incandescent ≈ 100; fluorescent ≈ 50–85. "
            "<br><b>Rf / Rg</b> (IES TM-30-15, printed alongside when "
            "your Argyll version provides it): Rf = fidelity over 99 "
            "modern test colours (Ra's successor; 100 = colours exactly "
            "like daylight), Rg = relative gamut area — Rg over 100 "
            "means the light over-saturates colours, under 100 washes "
            "them out. Only meaningful for Emissive/Ambient readings "
            "of lights, not for surfaces.")
        self.val_ri.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        # peak list — merged into the Measurements group, strongest first
        tip_peaks = wrap_tip(
            "Peaks of trace A (strongest first): "
            "local maxima that stand at least 4% "
            "above the saddle between their flanking "
            "valleys, and clearly above the noise "
            "floor — endpoints included, because a "
            "curve rising into 730 nm genuinely "
            "peaks there. The numbers "
            "match the ▼ markers on the plot.")
        lbl_peaks = QLabel("PEAKS (BY INTENSITY)")
        lbl_peaks.setProperty("hint", True)
        # same tooltip as the field below it — every other value_box
        # carries the tooltip on BOTH the title and the field
        lbl_peaks.setToolTip(tip_peaks)
        gl.addWidget(lbl_peaks, 8, 0, 1, 2)
        self.list_peaks = QListWidget()
        self.list_peaks.setToolTip(tip_peaks)
        # single click selects a row; the copy buttons pick up the
        # selection, and mouse selection is native on the whole list
        self.list_peaks.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_peaks.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        gl.addWidget(self.list_peaks, 9, 0, 1, 2)
        v.addWidget(g)

        g = QGroupBox("Colour coordinates")
        outer = QVBoxLayout(g)
        outer.setContentsMargins(4, 2, 4, 4)
        outer.setSpacing(4)
        tab1 = QWidget()
        _paint_only(tab1, "colourTab", "background: transparent;")
        gl = QGridLayout(tab1)
        gl.setContentsMargins(0, 4, 0, 0)
        gl.setColumnStretch(1, 1)
        gl.addWidget(QLabel("System"), 0, 0)
        self.cmb_coord = QComboBox()
        self.cmb_coord.addItems(["xyY (CIE 1931)", "CIELAB (D50)",
                                  "LCh(ab) (D50)", "XYZ raw",
                                  "Dominant wavelength"])
        self.cmb_coord.setToolTip(wrap_tip("CIE colour space used for the readout.<br>"
                                  "CIELAB/LCh come straight from the "
                                  "instrument (D50); the others are computed "
                                  "from XYZ and the CIE 1931 observer."))
        self.cmb_coord.currentTextChanged.connect(lambda *_: self._update_stats())
        gl.addWidget(self.cmb_coord, 0, 1)
        self.val_colour = value_box(
            gl, 1, "Coordinates",
            "Coordinates of the current measurement in the selected "
            "system: CIE xy chromaticity, CIELAB / LCh (D50), raw XYZ "
            "or dominant wavelength.")
        # selectable: copyable
        self.val_colour.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        lbl_sr = QLabel("sRGB preview")
        gl.addWidget(lbl_sr, 2, 0)
        self.lbl_srgb_vals = QLabel("—")
        self.lbl_srgb_vals.setAlignment(Qt.AlignmentFlag.AlignRight
                                        | Qt.AlignmentFlag.AlignVCenter)
        self.lbl_srgb_vals.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl_srgb_vals.setToolTip(wrap_tip("Exact 8-bit sRGB values shown in the "
                                      "preview. Red = outside the sRGB gamut "
                                      "(clamped)."))
        gl.addWidget(self.lbl_srgb_vals, 2, 1)
        self.swatch = QFrame()
        self.swatch.setMinimumHeight(46)
        self.swatch.setToolTip(wrap_tip("sRGB rendering of the measurement "
                               "(channels clipped if they exceed the "
                               "range). A red border means the colour "
                               "is outside the sRGB gamut."))
        gl.addWidget(self.swatch, 3, 0, 1, 2)
        outer.addWidget(tab1)
        v.addWidget(g)

        v.addStretch(1)
        scroll.setWidget(body)
        return scroll

    # ---- CIE 1931 chromaticity diagram ---------------------------------
    # (processing.chromaticity_image() renders the true-colour horseshoe;
    #  this draws the gamut triangles, the white points and the current
    #  measurement's cross-hair dot on top of it)
    # triangle colours follow the CHART's muted, less saturated palette
    # (ACCENT blue / REF orange families) — not neon marker hues
    _CHROM_TRI_COLOURS = {"sRGB": ACCENT, "Adobe RGB": "#77b98c",
                          "Display P3": "#d98a5e", "Rec. 2020": "#c9ae5f"}
    # SI prefixes the radiometric y axis picks from (value scale-out
    # factors — dividing the W-scale curve by the factor converts it
    # to the prefixed unit)
    _SI = {"k": 1e3, "": 1.0, "m": 1e-3, "µ": 1e-6, "n": 1e-9}
    _D50_WP_XY = (0.3457, 0.3585)
    _D65_WP = (0.3127, 0.3290)

    def _draw_chromaticity(self, fig, big: bool = False) -> dict:
        """Populate a matplotlib figure with the chromaticity diagram
        (CIE 1931 xy, 1976 u'v' or 1960 uv per the View menu setting;
        true-sRGB mode keeps the sRGB triangle in real colours and
        mutes the rest). Returns handles: {"cross": Line2D, "fig": fig}
        for later updates. big=True is the DETACHED-window presentation:
        ~1.5x fonts and tighter figure margins (the embedded panel keeps
        its compact look untouched)."""
        from .processing import (xy_to_chrom_space, chrom_space_labels,
                                 chrom_space_name, planckian_locus_xy)
        f = 1.5 if big else 1.0
        ax = fig.add_subplot(111)
        fig.set_facecolor(BG)
        ax.set_facecolor(BG)
        space = self.chrom_space
        extent, img, (px, py) = chromaticity_image(space=space,
                                                  true_srgb=self.true_srgb)
        # muted mode: the horseshoe blends into the dark background
        # exactly like the chart's rainbow fill (alpha 0.35); true-sRGB
        # mode: the image itself already carries the per-pixel alphas
        ax.imshow(img, origin="lower", extent=extent,
                  interpolation="bilinear", aspect="equal",
                  alpha=1.0 if self.true_srgb else 0.35, zorder=-3)
        # spectral locus / purple line
        ax.plot(np.append(px, px[0]), np.append(py, py[0]),
                color=FG, lw=1.1, zorder=-1)
        # Planckian locus (black-body radiation), gray like D50: the
        # line every measured CCT/Duv statement implicitly refers to
        plx, ply = planckian_locus_xy()
        pu, pv = xy_to_chrom_space(plx, ply, space)
        ax.plot(pu, pv, color=MUTED, lw=1.1, zorder=0)
        # gamut triangles (primaries from processing._GAMUT_PRIMS, the
        # same numbers the in-gamut readout uses) — projective space
        # changes map straight primary lines to straight lines
        from .processing import _GAMUT_PRIMS
        for name, prim in _GAMUT_PRIMS.items():
            tri = np.array(prim + (prim[0],))
            tu, tv = xy_to_chrom_space(tri[:, 0], tri[:, 1], space)
            ax.plot(tu, tv,
                    color=self._CHROM_TRI_COLOURS[name], lw=1.3,
                    label=name, zorder=2)
        # white points: D65 (gamut shared white) and D50 (instrument
        # Lab) — transformed into the current space like everything
        # else. In true-sRGB mode the sRGB triangle interior renders
        # at full strength and D65 sits on bright near-white ground:
        # BLACK then (same for its label), white otherwise. D50's
        # label rides up and right of its mark, clear of the gray
        # black-body line passing right beneath it.
        d65_col = BLACK if self.true_srgb else WHITE
        for (wx, wy), label, col, ms, mew, dx, dy, ha in (
                ((0.3127, 0.3290), "D65", d65_col, 9, 1.4, 6, -11, "left"),
                (self._D50_WP_XY, "D50", MUTED, 8, 1.2, 0, 10, "center")):
            wu, wv = xy_to_chrom_space(wx, wy, space)
            ax.plot([wu], [wv], marker="+", color=col, ms=ms, mew=mew,
                    zorder=3)
            ax.annotate(label, (wu, wv), xytext=(dx, dy),
                        textcoords="offset points", color=col,
                        fontsize=CHART_INSET_TICK * f, ha=ha)
        # the current measurement (hidden until one exists)
        cross, = ax.plot([], [], marker="o", ms=8, mfc=HOT, mec=WHITE,
                         mew=1.4, ls="none", zorder=5)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        xl, yl = chrom_space_labels(space)
        ax.set_xlabel(xl, color=FG, fontsize=CHART_INSET * f)
        ax.set_ylabel(yl, color=FG, fontsize=CHART_INSET * f)
        ax.tick_params(colors=MUTED, labelsize=7 * f)
        for sp in ax.spines.values():
            sp.set_color(BORDER)
        # title = space name only: the gamut list is already in the
        # legend, and the long form width-clips below ~460 px at the
        # big view's 12 pt
        ax.set_title(chrom_space_name(space),
                     color=FG, fontsize=CHART_INSET * f)
        leg = ax.legend(loc="upper right", fontsize=CHART_INSET_LEG * f, framealpha=1.0,
                        labelspacing=0.2, handlelength=1.4,
                        facecolor=BG, edgecolor=BORDER)
        if big:
            # tight figure margins on the DETACHED viewer — but via
            # the tight-layout ENGINE, not fixed subplots_adjust
            # fractions: the engine recomputes on every resize so the
            # (now larger) tick and axis labels can never clip against
            # the figure edge, whatever size the user drags to
            fig.set_layout_engine("tight")
            fig.get_layout_engine().set(w_pad=1.2, h_pad=1.2)
        for t in leg.get_texts():
            t.set_color(FG)
        return {"cross": cross, "ax": ax, "fig": fig}

    def _redraw_chrom_all(self):
        """Space or true-sRGB mode changed: rebuild the embedded figure
        and every detached chromaticity window in place."""
        def rebuild(fig, canvas, big):
            fig.clf()
            handles = self._draw_chromaticity(fig, big=big)
            if canvas is not None:
                canvas.draw_idle()
            return handles
        self.chrom_handles = rebuild(self.chrom_fig, None, False)
        self.chrom_canvas.draw_idle()
        for i, (dlg, _h) in enumerate(list(self._chrom_windows)):
            try:
                canvases = dlg.findChildren(FigureCanvasQTAgg)
                canvas = canvases[0] if canvases else None
                fig = canvas.figure if canvas is not None else _h.get("fig")
                if fig is None:
                    continue
                handles = rebuild(fig, canvas, True)   # detached windows keep the big presentation
                self._chrom_windows[i] = (dlg, handles)
            except RuntimeError:
                continue          # window already deleted
        self._update_chrom(getattr(self.trace_a, "xyz", None))
        self._save_settings()

    def _open_chrom_window(self):
        """Detached, resizable copy of the chromaticity diagram."""
        from PySide6.QtWidgets import QDialog
        dlg = QDialog(self)
        dlg.setWindowTitle(chrom_space_name(self.chrom_space)
                           + " chromaticity — spectro-live")
        # the OUTER portion of the window (the frame/margin ring around
        # the canvas) carries the side-panel / status-bar colour so the
        # detached viewer reads as part of the app; the canvas itself
        # keeps painting the chart's own background inside
        dlg.setStyleSheet(f"QDialog {{ background: {CARD}; }}\n" +
                         tooltip_css())
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(4, 4, 4, 4)
        fig = Figure(figsize=(6.4, 5.6), dpi=100)
        handles = self._draw_chromaticity(fig, big=True)
        canvas = FigureCanvasQTAgg(fig)
        lay.addWidget(canvas)
        dlg.resize(640, 560)
        self._chrom_windows.append((dlg, handles))
        # clean-up: the window deletes itself when the user closes it
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dlg.destroyed.connect(self._forget_chrom_window)
        # bring it in step with the current measurement, then show it
        self._update_chrom(getattr(self.trace_a, "xyz", None))
        dlg.show()

    def _forget_chrom_window(self, _obj=None):
        """A detached chromaticity window was closed (and deleted) —
        drop it from the live list."""
        try:
            from shiboken6 import isValid
            self._chrom_windows = [w for w in self._chrom_windows
                                   if isValid(w[0])]
        except ImportError:
            pass

    def _fmt_radiometric(self, er: float, unit: str, pair: str) -> str:
        """A radiometric value with an engineering prefix that keeps the
        number in comfortable digits (kW .. µW), its paired unit, and a
        note of the photometric quantity the scale is anchored to."""
        for factor, prefix in ((1e3, "k"), (1.0, ""), (1e-3, "m"),
                               (1e-6, "µ")):
            if er >= factor:
                return (f"{er/factor:,.3g} {prefix}W"
                        f"{'/m²' if unit == 'W/m²' else '/(sr·m²)'}"
                        f"  ({pair})")
        return (f"{er:,.3g} nW"
                f"{'/m²' if unit == 'W/m²' else '/(sr·m²)'}  ({pair})")

    def _update_chrom(self, xyz):
        """Move the cross-hair dot on every chromaticity view (the
        embedded tab figure and any windows the user opened)."""
        from shiboken6 import isValid
        views = [(self.chrom_handles, self.chrom_canvas)]
        for w in list(self._chrom_windows):
            try:
                if not isValid(w[0]):
                    continue
                canvases = w[0].findChildren(FigureCanvasQTAgg)
                views.append((w[1], canvases[0] if canvases else None))
            except RuntimeError:
                continue          # window already deleted
        x, y = None, None
        if xyz is not None and sum(xyz) > 0:
            try:
                x, y = xy_from_xyz(xyz)
                x, y = [float(v) for v in
                        xy_to_chrom_space(x, y, self.chrom_space)]
            except Exception:
                x, y = None, None
        for entry in views:
            handles, canvas = entry[0], entry[1]
            cross = handles["cross"]
            if x is None:
                cross.set_data([], [])
            else:
                cross.set_data([x], [y])
            if canvas is not None:
                canvas.draw_idle()

    def _statusbar(self):
        sb = self.statusBar()
        # The bar is built from PERMANENT widgets only: QStatusBar's own
        # internal message label has an Ignored horizontal size policy,
        # so any expanding permanent widget (the CAL centering spacers)
        # starves it to zero width — transient messages vanish. Our own
        # left message label keeps the bottom-left readout alive.
        # Layout (insertion order, left to right): [message (stretch 1)]
        # [CAL countdown] [spacer (stretch 1)] [device] [rate] [log].
        # The two stretch-1 items share the free space equally, keeping
        # the countdown centered between them.
        sb.setStyleSheet(
            f"QStatusBar {{ background: {CARD}; color: {MUTED}; }}\n"
            f"QStatusBar QLabel, QStatusBar QPushButton "
            f"{{ background: transparent; color: {MUTED}; }}\n"
            + tooltip_css())
        self._status_tag = 0
        self.lbl_message = QLabel("")
        self.lbl_message.setAlignment(Qt.AlignmentFlag.AlignLeft
                                      | Qt.AlignmentFlag.AlignVCenter)
        sb.addPermanentWidget(self.lbl_message, 1)
        self.lbl_cal = QLabel("CAL —")
        self.lbl_cal.setAlignment(Qt.AlignmentFlag.AlignCenter
                                  | Qt.AlignmentFlag.AlignVCenter)
        self.lbl_cal.setToolTip(
            wrap_tip("Time remaining until this instrument's calibration goes "
            "stale by Argyll's own rules (dark cal 1 h for emission "
            "modes, reflective white 24 h)."))
        sb.addPermanentWidget(self.lbl_cal, 0)
        sp_r = QLabel("")
        sp_r.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Preferred)
        sb.addPermanentWidget(sp_r, 1)
        self.lbl_device = QLabel("Instrument: —")
        self.lbl_device.setToolTip(wrap_tip("USB device currently visible to "
                                   "ArgyllCMS."))
        sb.addPermanentWidget(self.lbl_device)
        self.lbl_rate = QLabel("")
        self.lbl_rate.setToolTip(wrap_tip("Number of readings taken and the average "
                                 "time between them."))
        sb.addPermanentWidget(self.lbl_rate)
        b = QPushButton("Instrument log")
        b.setFlat(True)
        b.setToolTip(wrap_tip("Show the raw spotread output and driver state log."))
        b.clicked.connect(self.on_show_log)
        sb.addPermanentWidget(b)

    @staticmethod
    def _repolish(widget):
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    # -- copy the measurements panel -------------------------------------
    def _readout_text(self) -> str:
        """Everything in the right-hand panel as plain text (used for the
        Copy button and Ctrl+C in the peaks list)."""
        def sec(title, value):
            value = (value or "").strip()
            return f"{title}: {value}" if value else f"{title}: —"
        lines = [
            f"# {APP_NAME} measurements — "
            f"{self.last_raw.timestamp if self.last_raw else ''}"
            f"{' — ' + self.last_raw.source if self.last_raw else ''}",
            sec("Band integral ∫", self.val_integral.text()),
        ]
        # value boxes can hold multi-line text: paste them verbatim
        for name, box in (("Photometric Y", self.val_photometric),
                          ("Radiometric", self.val_radiometric),
                          ("CCT", self.val_cct),
                          ("Colour rendering", self.val_ri),
                          ("Peak / FWHM / centroid", self.val_peak),
                          ("Markers M1 / M2", self.val_markers),
                          ("Coordinates", self.val_colour)):
            t = box.text().strip()
            if "\n" in t:
                lines.append(f"{name}:")
                lines += [f"  {row}" for row in t.splitlines()]
            else:
                lines.append(sec(name, t))
        if self.lbl_srgb_vals.text() not in ("", "—"):
            lines.append(f"sRGB preview: {self.lbl_srgb_vals.text()}")
        if self.list_peaks.count():
            lines.append("Peaks (by intensity):")
            lines += ["  " + self.list_peaks.item(i).text()
                      for i in range(self.list_peaks.count())]
        return "\n".join(lines)

    def on_copy_readout(self):
        """Copy the measurements panel to the clipboard (whole panel, or
        only the selected peak rows if rows are selected)."""
        sel = [it.text() for it in self.list_peaks.selectedItems()]
        if len(sel) and len(sel) < self.list_peaks.count():
            text = "Peaks (by intensity):\n" + "\n".join("  " + s for s in sel)
        else:
            text = self._readout_text()
        QApplication.clipboard().setText(text)
        n = len(text.splitlines())
        self.status(f"Copied measurements ({n} lines) to the clipboard.")

    def keyPressEvent(self, ev):
        """Ctrl+C anywhere with peak rows selected copies those rows
        (falls back to Ctrl+C in any selectable text field)."""
        if ev.matches(QKeySequence.StandardKey.Copy) and \
                self.list_peaks.selectedItems():
            self.on_copy_readout()
            return
        super().keyPressEvent(ev)

    def status(self, text: str, ms: int | None = None):
        """Bottom-left status readout (our own permanent label — NOT
        QStatusBar.showMessage, whose internal label collapses to zero
        width beside the CAL centering spacers). Auto-clears after a
        timeout unless superseded by a newer message."""
        lbl = getattr(self, "lbl_message", None)
        if lbl is None:               # the bar is not built yet
            self.statusBar().showMessage(text)
            return
        lbl.setText(text)
        lbl.setToolTip(wrap_tip(text))
        self._status_tag += 1
        tag = self._status_tag
        ms = ms if ms is not None else (8000 if len(text) < 90 else 20000)

        def _clear():
            if tag == self._status_tag:   # not superseded since
                lbl.setText("")
                lbl.setToolTip(wrap_tip(""))
        QTimer.singleShot(ms, _clear)

    def _update_cal_countdown(self):
        """Center status-bar readout: minutes until Argyll considers
        the calibration stale (drives green CAL border too)."""
        lbl = getattr(self, "lbl_cal", None)
        if lbl is None:
            return
        if self._cal_ok_until <= 0.0:
            lbl.setText("CAL —")
            return
        left = self._cal_ok_until - time.monotonic()
        if left <= 0.0:
            lbl.setText("CAL expired")
            return
        mins = left / 60.0
        txt = (f"CAL✓ {int(mins // 60)}h {int(round(mins % 60))}m left"
               if mins >= 60 else
               f"CAL✓ {int(round(mins))} min left" if mins >= 1
               else "CAL✓ <1 min left")
        lbl.setText(txt)

    # ===================================================================
    # worker events
    # ===================================================================
    def _poll(self):
        try:
            while True:
                ev = self.events.get_nowait()
                if ev[0] == "frame":
                    self._on_frame(ev[1])
                elif ev[0] == "state":
                    self._on_state(ev[1], ev[2])
                elif ev[0] == "log":
                    self._log(ev[1])
        except queue.Empty:
            pass

    def _log(self, text):
        self.log_lines.append(text)
        self._diag_write(text + "\n")

    def _diag_write(self, text: str):
        if not self.diag_path:
            return
        try:
            with open(self.diag_path, "a", encoding="utf-8") as f:
                stamp = "" if text.startswith("=") else \
                    f"[{datetime.now():%H:%M:%S}] "
                f.write(stamp + text)
        except OSError:
            pass

    def _on_state(self, key, msg):
        self._diag_write(f"state: {key}  {msg}\n")
        if key == "instrument":
            self._on_instrument_type(msg or "")
            return
        if key == "no-device":
            msg = msg or "Device disconnected (USB)"
        elif key in ("no-argyll", "reconnecting", "starting", "restart") \
                and not msg:
            msg = key
        # the SINGLE button must always recover when a reading can no
        # longer arrive (device gone, session restart, guide prompt)
        if key in ("restart", "reconnecting", "no-device", "no-argyll"):
            self.btn_single.setText("SINGLE")
            self.btn_single.setEnabled(True)
        # session-liveness tracking for the position lamp: a reading can
        # arrive whenever a session is up (app request OR the instrument's
        # physical button), so an unconfirmed slot must be signalled then
        if key in ("no-device", "no-argyll", "no-argyl"):
            self._device_online = False
        elif key in ("reconnecting",):
            self._device_online = False
        else:
            # starting | calibrating | calibrate-confirm | measuring |
            # guide | restart | instrument — all mean the device is (about
            # to be) reachable
            self._device_online = True
        # Annunciator lamps are a proper state machine now:
        #   CALIBRATION      lit = a calibration is needed or running,
        #                    off otherwise
        #   SURFACE/AMBIENT  lit = the current wheel slot is NOT yet
        #   POSITION              confirmed by a successful reading
        #   *_confirmed flips True on the first successful frame and False
        #   whenever the position becomes unknown again (calibration,
        #   wrong-position failure, session restart).
        low = (msg or "").lower()
        pos_waiting = "position" in low or "wheel" in low
        if key == "guide" and "saturated" in low:
            # spotread itself declared the sensor CLIPPED. It is the
            # authority over the saturation bar's estimate (the bar may
            # not have reached the top on its own: spotread aborts the
            # measurement, so no brighter frame ever arrives). Pin the
            # bar at the top — at 1.5×clip so it FLASHES red — until a
            # real reading arrives.
            self.satbar.set_fraction(1.5)
        if key == "calibrating":
            self._pos_confirmed = False
            self._invalidate_cal()     # a dance is running: void the border
            low_cal = low
            m = re.search(r"\((\d+)/(\d+)\)", msg or "")
            # all-modes pass: remember the current stage, but the lamp
            # shows the FULL yellow until the wheel is really at the
            # calibration slot — the quarter-fill "loading" effect
            # starts only when spotread confirms the calibration is
            # actually running (cal_running's "Calibrating…" prompt,
            # as opposed to the pass header / the "turn the wheel"
            # prompt that both arrive BEFORE the user has moved it).
            if m:
                self._cal_stage = (int(m.group(1)) /
                                   (int(m.group(2)) + 1))
            now_running = (low_cal.startswith("calibrating")
                           and "every reading mode" not in low_cal
                           and "turn the wheel" not in low_cal)
            # The stage header itself starts the quarter-fill: 1/3 ->
            # 25 %, 2/3 -> 50 %, 3/3 -> 75 % — no solid-yellow blink
            # between stages (the worker auto-answers the wheel prompt
            # seconds after each header, so the blink was noise). Only
            # the non-pass prompts (standalone "turn the wheel") and
            # the end-of-pass summary fall back to the solid lamp.
            if now_running or m:
                self._cal_progress = getattr(self, "_cal_stage", None)
            else:
                self._cal_progress = None
            self._set_lamps(cal="solid",
                            caption=sources.cal_prompt(self.inst_type)
                            if "turn the wheel" in low_cal or
                            "set the instrument" in low_cal
                            else (msg or "Calibrating…"))
        elif key == "measuring":
            if "calibrated ✓" in low:
                # spotread just completed a FRESH dance: full window
                # and a new persisted stamp
                self._mark_cal_valid()
                self._cal_stage = None     # the pass is over
            elif "calibration still valid" in low:
                # the session started on Argyll's host cache (no dance):
                # do NOT restart the window — rebuild the remaining
                # time from the last fresh dance's persisted stamp
                self._mark_cal_valid(fresh=False)
                self._cal_stage = None
            if pos_waiting:
                if self.mode_name == "Reflective":
                    self._set_lamps(caption="Hold flat against the surface")
                elif self.mode_name == "Ambient":
                    self._set_lamps(caption="Diffuser up, facing the room")
                elif self.mode_name == "Emissive":
                    self._set_lamps(caption="Point at the target")
            else:
                self._set_lamps()          # all dark: reading normally
        elif key == "guide":
            if pos_waiting:
                self._pos_confirmed = False
            self._set_lamps()
            self._set_caption(msg or "")
        elif key == "restart":
            # A restarted session is NOT evidence that the wheel moved:
            # a USB blip cannot turn the wheel. Keep the slot
            # confirmation — if the user really did move the wheel, the
            # next attempted reading fails with spotread's own
            # wrong-position line, and THAT re-arms the lamp. Asking the
            # user to "move the wheel" they never touched would be a lie.
            self._set_lamps(caption="Restarting session…")
        elif key == "reconnecting":
            self._set_lamps(caption="Instrument lost – retrying…")
        elif key == "no-device":
            self._set_lamps(caption="Instrument not connected")
        else:
            self._set_lamps()
            self._set_caption(msg or "")
        if key != "calibrating":
            # the progress-bar lamp belongs to the all-modes pass alone;
            # any other state ends it (a plain solid lamp again)
            self._cal_progress = None
        if key == "guide":
            self.btn_single.setText("SINGLE")
            self.btn_single.setEnabled(True)
        self._update_position_lamp()
        if msg:
            self.status(msg)

    # -- multi-instrument support ----------------------------------------
    def _on_instrument_type(self, itype: str):
        """spotread named the connected instrument. Restrict the reading
        modes to what this hardware can actually do (an i1 Pro has no
        ambient diffuser; a colorimeter only reads emissive) and drop
        the ColorMunki wheel wording for other devices."""
        prev = getattr(self, "inst_type", "")
        self.inst_type = itype
        if itype:
            # the one label that can be trusted: spotread named the
            # instrument it actually opened (the startup guess comes
            # from the USB scan and can be wrong)
            self.lbl_device.setText(f"Instrument: {itype}")
        modes = list(sources.instrument_modes(itype))
        self.allowed_modes = modes
        # keep the live worker's calibrate-all pass in step with what
        # the instrument actually supports
        if isinstance(self.worker, sources.SpotreadWorker) \
                and self.worker.is_alive():
            self.worker.cal_pass_modes = tuple(
                {"Emissive": "e", "Ambient": "a", "Reflective": "r"}[m]
                for m in modes if m in ("Emissive", "Ambient",
                                        "Reflective")) or None
        if self.mode_name not in modes:
            # persisted mode unsupported by this instrument: switch
            self._diag_write(f"GUI: mode '{self.mode_name}' unsupported on "
                             f"'{itype}' -> switching to {modes[0]}\n")
            self._select_mode(modes[0])
        self.status(f"Instrument: {itype or 'unknown'} — supported modes: "
                    f"{', '.join(modes)}")
        if prev != itype:
            self._diag_write(f"GUI: instrument type '{itype}'\n")

    def _fill_device_menu(self):
        """(Re)build the Instrument → Device submenu from the USB scan.
        Counts are spotread's own -c port numbers (1-based; the scan
        order roughly matches, spotread remains the final arbiter)."""
        m = self.menu_dev
        m.clear()
        devs = [d for d in sources.usb_devices()
                if not any(n in d[2].lower()
                           for n in sources._NON_INSTRUMENTS)]
        grp = QActionGroup(self)
        a = QAction("Auto (first found)", self)
        a.setCheckable(True)
        a.setChecked(self.device_port is None)
        a.triggered.connect(lambda: self._pick_device(None))
        grp.addAction(a)
        m.addAction(a)
        m.addSeparator()
        if not devs:
            nol = QAction("(no USB device with a HID interface seen)", self)
            nol.setEnabled(False)
            m.addAction(nol)
            return
        for i, (_bus, ids, label) in enumerate(devs, 1):
            d = QAction(f"{i}.  {label}  [{ids}]", self)
            d.setCheckable(True)
            d.setChecked(self.device_port == i)
            d.triggered.connect(lambda _=False, n=i: self._pick_device(n))
            grp.addAction(d)
            m.addAction(d)

    def _pick_device(self, n):
        self.device_port = n
        self._diag_write(f"GUI: device port selected: {n}\n")
        self._save_settings()
        if self.worker and self.worker.is_alive() and \
                type(self.worker).__name__ == "SpotreadWorker":
            was_running = self.running
            self.worker.stop()
            self.worker = self._make_worker()
            self.worker.start()
            if was_running:
                self.worker.start_continuous()
                self.status("Switched instrument — restarting session "
                            "(auto-calibrating).")
            else:
                self.status("Instrument selected — starts with the next "
                            "reading.")
        else:
            self.status("Instrument selected — applies on next start.")

    # -- annunciator lamps ----------------------------------------------
    def _mode_lamp_key(self, mode_name: str):
        """The lamp that doubles as this reading mode's selector: each
        mode now has a lamp of its own (SURFACE = emissive, REFLECTIVE
        and AMBIENT) — they differ in HOW the light gets to the sensor,
        so each mode-lamp stands for the position to set."""
        return {"Emissive": "surface", "Reflective": "reflective",
                "Ambient": "ambient"}.get(mode_name, "surface")

    def _update_position_lamp(self):
        """Central renderer for the position lamp.

        It is lit IFF (a live instrument session exists) AND (the wheel
        slot for the current reading mode has not been confirmed by a
        successful reading) AND (calibration is not running). Whether a
        measurement is pending no longer matters: a reading can always
        arrive without the app asking for it — the instrument's own
        physical button is a trigger spotread honors at its menu — so an
        unconfirmed slot must be signalled even in single mode / paused.
        Every state transition funnels through here, so the lamp can
        never stay stale.
        """
        cal_active = self._lamp_state.get("cal", "off") != "off"
        want = self._device_online and not self._pos_confirmed \
            and not cal_active
        # exactly ONE mode lamp can be lit: the selected reading mode's,
        # while its position is unconfirmed. (Emissive and Reflective
        # both read through the ColorMunki's front aperture, wheel at 3;
        # ambient goes through the top diffuser, past OFF.)
        key = self._mode_lamp_key(self.mode_name)
        state = "off"
        if key is not None:
            state = "solid" if want else "off"
            for k in self._LAMP_MODES:
                new = state if k == key else "off"
                if self._lamp_state.get(k, "off") != new:
                    self._lamp_state[k] = new
                    self._apply_lamps()
            if state == "solid":
                slot = ("the ambient diffuser slot (past OFF)"
                        if key == "ambient"
                        else "position 3 (MEASURING)")
                self.status(f"Wheel not in place yet — turn it to "
                            f"{slot}…")
        return state

    def _apply_lamps(self):
        """Paint every lamp according to its state.

        CALIBRATION   lit = a calibration is needed or running (during
                      the all-modes pass the amber fill grows a quarter
                      per stage); green border while the calibration is
                      valid. Click = recalibrate.
        mode lamps    SURFACE (emissive), REFLECTIVE, AMBIENT are the
                      reading-mode selector: the selected one carries a
                      BLUE border and lights amber only for itself
                      while its position is unconfirmed. Hover
                      highlights them like the buttons they are.
        No lamp ever flashes, by design."""
        cal_valid = time.monotonic() < self._cal_ok_until
        sel_key = self._mode_lamp_key(self.mode_name)
        for key, lamp in self.lamps.items():
            state = self._lamp_state.get(key, "off")
            lit = state != "off"
            if key == "cal":
                if not lit and cal_valid:
                    # green border + dark green tint inside: the
                    # calibration is valid (state readable at a glance),
                    # but no calibration is running right now
                    lamp.setStyleSheet(
                        f"LampLabel {{ background: {OK_TINT}; color: {OK}; "
                        f"border: 2px solid {OK}; border-radius: 6px; "
                        "font-weight: bold; letter-spacing: 1px; }\n"
                        f"LampLabel:hover {{ background: {LAMP_HOVER}; "
                        f"border-color: {WHITE}; }}" + "\n" + tooltip_css())
                    continue
                if lit and self._cal_progress is not None:
                    # pass stage fill (25/50/75 %); 100 % "ready" is the
                    # green border once the pass confirms validity
                    frac = max(0.0, min(0.999, self._cal_progress))
                    lamp.setStyleSheet(
                        f"LampLabel {{ background: qlineargradient("
                        f"x1:0, y1:0, x2:1, y2:0, stop:0 {LAMP_AMBER}, "
                        f"stop:{frac:.4f} {LAMP_AMBER}, "
                        f"stop:{frac:.4f} {FIELD}, "
                        f"stop:1 {FIELD}); color: {WHITE}; "
                        f"border: 1px solid {LAMP_AMBER}; border-radius: 6px; "
                        "font-weight: bold; letter-spacing: 1px; }" + "\n" + tooltip_css())
                    continue
                base = (f"background: {LAMP_AMBER}; color: {BLACK}; " if lit
                        else f"background: {FIELD}; color: {MUTED}; ")
                lamp.setStyleSheet(
                    f"LampLabel {{ {base}"
                    f"border: 2px solid {LAMP_AMBER if lit else BORDER}; "
                    "border-radius: 6px; font-weight: bold; "
                    "letter-spacing: 1px; }\n"
                    f"LampLabel:hover {{ background: {LAMP_HOVER}; "
                    f"border-color: {WHITE}; }}" + "\n" + tooltip_css())
                continue
            # ---- the three mode-selector lamps -------------------------
            border = ACCENT if key == sel_key else BORDER
            if lit:
                lamp.setStyleSheet(
                    f"LampLabel {{ background: {LAMP_AMBER}; color: {BLACK}; "
                    f"border: 2px solid {ACCENT}; border-radius: 6px; "
                    "font-weight: bold; letter-spacing: 1px; }\n"
                    f"LampLabel:hover {{ border-color: {WHITE}; }}" + "\n" + tooltip_css())
            else:
                hover_border = (LAMP_HOVER_BORDER if key == sel_key else WHITE)
                # selected + unlit: blue border with a dark blue tint
                # inside (the selected mode reads at a glance), instead
                # of an amber that would lie (nothing is unconfirmed)
                base = (f"background: {ACCENT_TINT}; color: {ACCENT}; "
                        if key == sel_key
                        else f"background: {FIELD}; color: {MUTED}; ")
                lamp.setStyleSheet(
                    f"LampLabel {{ {base}"
                    f"border: 2px solid {border}; border-radius: 6px; "
                    "font-weight: bold; letter-spacing: 1px; }\n"
                    f"LampLabel:hover {{ background: {LAMP_HOVER}; "
                    f"border-color: {hover_border}; }}" + "\n" + tooltip_css())

    def _cal_gone_stale(self) -> bool:
        """True when the last-seen calibration has passed Argyll's
        1 h dark-cal validity window (the border must fade, since the
        next session WILL demand a fresh dance)."""
        return (self._cal_ok_until > 0.0
                and time.monotonic() >= self._cal_ok_until)

    def _mode_key(self) -> str:
        """'e'/'a'/'r' — the key under which the last fresh-calibration
        stamp is stored for the current reading mode."""
        m = (self.mode_name or "").lower()
        if m.startswith("r"):
            return "r"
        if m.startswith("a"):
            return "a"
        return "e"

    def _mark_cal_valid(self, seconds: float | None = None,
                        fresh: bool = True):
        """Mark the instrument calibration valid for `seconds` (default:
        the instrument's own window — dark cal 1 h for emission modes,
        reflective white 24 h).

        `fresh=True` — spotread just COMPLETED a calibration dance
        ("Calibration complete"): the full window starts now, and the
        wall-clock stamp is persisted so a later app restart can
        reconstruct the countdown instead of restarting it.

        `fresh=False` — spotread started on Argyll's HOST CACHE
        ("Calibration still valid", no dance): the window must NOT
        restart. The remaining time comes from Argyll's own host
        cache (the .cal file stores the REAL last-dance dates, which
        munki_restore_calibration() restores and get_n_a_cals()
        compares against wall clock — validity is judged by the
        actual calibration age, not per session). If the cache can't
        be parsed, fall back to the app's persisted stamp; with
        neither on record, the full window is the best guess."""
        if seconds is None:
            seconds = sources.cal_validity_s(self.mode_name)
        if not fresh:
            key = self._mode_key()
            # 1) Argyll's own dates — the same source spotread uses
            left = sources.argyll_cal_remaining(self.mode_name,
                                                bool(getattr(self,
                                                             "fast", False)))
            if left is not None:
                seconds = left
            else:
                # 2) this app's stamp of the last fresh dance
                stamp = self.cal_ts.get(key) if isinstance(
                    self.cal_ts, dict) else None
                if isinstance(stamp, (int, float)):
                    lft = (stamp + seconds) - time.time()
                    if lft > 0:
                        seconds = lft
            # neither: fall through with the full window (unknown age)
        self._cal_ok_until = time.monotonic() + seconds
        if fresh:
            # persist only on a REAL dance — otherwise every restart
            # would overwrite the genuine stamp
            if not isinstance(self.cal_ts, dict):
                self.cal_ts = {}
            self.cal_ts[self._mode_key()] = time.time()
            self._save_settings()
        self._apply_lamps()
        """A completed/valid calibration was just confirmed (either
        spotread said 'Calibration complete', or the session reached
        its menu / delivered a reading without any calibration demand —
        via Argyll's host cache). Green for as long as the calibration
        is actually valid: the instrument's own windows (dark cal 1 h
        for emission modes, reflective white 24 h) unless overridden."""

    def _argyll_menu_update(self):
        """Keep the Argyll-path menu items readable: show the actual
        path in use (explicit or PATH default) and disable 'reset'
        when nothing is overridden."""
        p = getattr(sources, "_ARGYLL_OVERRIDE", None)
        self.act_argyll_default.setEnabled(bool(p))
        self.act_argyll_default.setText(
            f"Use default spotread (from PATH)"
            + ("" if p else f"   [{sources.spotread_path() or 'not found'}]"))

    def _set_argyll_path(self):
        """Pick an explicit ArgyllCMS spotread binary (e.g. a different
        Argyll build). Validated before anything changes; the session
        restarts on the new binary if one is running."""
        from PySide6.QtWidgets import QFileDialog
        start = getattr(sources, "_ARGYLL_OVERRIDE", None) \
            or sources.spotread_path() or os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(
            self, "Path to the spotread binary", start,
            "spotread (spotread*);;All files (*)")
        if not path:
            return
        path = os.path.abspath(path)
        if not (os.path.isfile(path) and os.access(path, os.X_OK)):
            QMessageBox.warning(
                self, APP_NAME, f"Not an executable file:\n{path}\n\n"
                "The ArgyllCMS path is unchanged.")
            return
        sources.set_argyll_path(path)
        self.argyll_path = path
        self._save_settings()
        self._argyll_menu_update()
        w = self.worker
        if w is not None and getattr(w, "is_alive", lambda: False)():
            w.restart_session()
            self.status(f"ArgyllCMS path set to {path} — restarting the "
                        "instrument session on it…")
        else:
            self.status(f"ArgyllCMS path set to {path}.")

    def _set_argyll_path_default(self):
        """Forget the explicit ArgyllCMS path: spotread comes from PATH
        again (the session restarts on it if one is running)."""
        sources.set_argyll_path(None)
        self.argyll_path = ""
        self._save_settings()
        self._argyll_menu_update()
        w = self.worker
        if w is not None and getattr(w, "is_alive", lambda: False)():
            w.restart_session()
            self.status("ArgyllCMS path reset — using spotread from PATH; "
                        "restarting the instrument session…")
        else:
            self.status(f"ArgyllCMS path reset — using spotread from PATH "
                        f"({sources.spotread_path() or 'not found'}).")

    def _on_cal_lamp_clicked(self):
        """Click on the CALIBRATION lamp: demand a fresh calibration of
        every reading mode right now (one-shot; normal Argyll cache
        reuse resumes afterwards)."""
        w = self.worker
        fn = getattr(w, "request_calibration", None)
        if fn is not None and getattr(w, "is_alive", lambda: False)():
            self._invalidate_cal()
            fn()
            self.status("Recalibrating every reading mode — turn the "
                        "wheel to position 2 (CALIBRATION)…")
        else:
            self.status("No live instrument session — nothing to "
                        "recalibrate.")

    def _invalidate_cal(self):
        """A calibration dance just started or a calibration was
        demanded: the previous state is void."""
        if self._cal_ok_until:
            self._cal_ok_until = 0.0
        self._apply_lamps()

    def _blink_tick(self):
        """Shared slow timer tick: saturation-bar flash phase at clip
        level + the 1 h fade of the calibration green border. The lamps
        themselves are steady — no flashing by design."""
        self.satbar.tick()
        self._update_cal_countdown()
        self._center_tool_group()
        if self._cal_gone_stale():        # 1 h expiry: fade the border
            self._invalidate_cal()

    def _set_lamps(self, cal="off", surface="off", ambient="off",
                   caption="", cap_colour=None, reflective="off"):
        self._lamp_state = {"cal": cal, "surface": surface,
                            "reflective": reflective, "ambient": ambient}
        self._apply_lamps()
        if caption:
            # caption text belongs in the bottom status bar ONLY — the
            # chart area carries the annunciator lamps and nothing else
            self.status(caption)

    def _set_caption(self, text, colour=None):
        if text:
            self.status(text)

    _last_logged_size = None

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._center_tool_group()
        # forensic: the window has been seen growing wider by itself;
        # log every size change so the cause can be pinned down
        sz = (e.size().width(), e.size().height())
        if getattr(self, "_last_logged_size", None) != sz:
            self._last_logged_size = sz
            self._diag_write(f"GUI: window resized to {sz[0]}x{sz[1]}"
                             f" (min {self.minimumSizeHint().width()})"
                             f"{' [grew wider]' if sz[0] > 1281 else ''}\n")

    # frames arriving while paused less than this many seconds apart count
    # as one continuous stream (a held/closed switch), not separate presses
    _HW_STREAM_S = 20.0
    # after clicking STOP, a frame arriving within this many seconds is
    # the in-flight reading of the run that was just stopped (an early
    # STOP can't recall a measurement already in progress); anything
    # later is a straggler and is DROPPED
    _STOP_GRACE_S = 2.0

    def _on_frame(self, raw: Spectrum):
        # ---- stopped: late readings are DROPPED --------------------------
        # A frame arriving after a STOP belongs to a stopped session:
        # within the short grace window it is the in-flight reading of
        # the run just stopped (kept); later than that it is a straggler
        # — self-triggered by spotread's switch poll or a physical
        # button press — and gets dropped so the frozen trace stays
        # exactly what it was when STOP was clicked. A ⏺ SINGLE keeps
        # working while stopped: its frame is not dropped (the button is
        # disabled while the SINGLE is in flight).
        if not self.running:
            if not self.btn_single.isEnabled():
                pass     # a ⏺ SINGLE is in flight: never dropped
            elif time.monotonic() - getattr(self, "_stop_since", 0.0) \
                    > self._STOP_GRACE_S:
                self._hw_streak = 0
                self._hw_suppressing = False
                self._diag_write(
                    "GUI: frame while STOPPED dropped (late straggler, "
                    f"hw_triggered={int(bool(getattr(raw, 'hw_triggered', False)))}, "
                    f"{raw.source})\n")
                return
        else:
            self._hw_streak = 0
            self._hw_suppressing = False
        self.frame_count += 1
        self.last_raw = raw
        # a successful reading is proof the session's calibration was
        # valid at this moment — border green until spotread says
        # otherwise (it demands a dance the instant the cal goes stale).
        # fresh=False: a frame says nothing about the cal's AGE — the
        # countdown keeps running; it must not restart
        if not self._cal_ok_until:
            self._mark_cal_valid(fresh=False)
        # saturation annunciator: busiest sensor channel of the RAW
        # reading (before any unit conversion) vs the estimated clip
        self._sat_frac = saturation_fraction(
            raw, fast=getattr(self, "fast", False),
            mode_name=self.mode_name,
            inst_type=getattr(self, "inst_type", ""))
        self.satbar.set_fraction(self._sat_frac)
        self.btn_single.setText("SINGLE")
        self.btn_single.setEnabled(True)
        # a frame arrived: whatever wheel position we used is confirmed
        self._pos_confirmed = True
        self._set_lamps()
        self._update_position_lamp()
        self._diag_write(f"GUI: frame #{self.frame_count} from {raw.source}, "
                         f"{len(raw.wavelengths)} pts\n")
        now = time.monotonic()
        if self.frame_count > 1:
            self.read_times.append(now - self._t_frame)
            self.read_times = self.read_times[-20:]
        self._t_frame = now
        self._adapt_plot_range(raw)
        avg = sum(self.read_times) / len(self.read_times) if self.read_times else 0
        self.lbl_rate.setText(f"{self.frame_count} readings"
                               + (f" · {avg:.1f} s/reading" if avg > 0 else ""))
        if self.csv_log and self.csv_path:
            try:
                with open(self.csv_path, "a", encoding="utf-8") as f:
                    f.write(f'"{raw.timestamp}","{self.frame_count}",'
                            f'{raw.source}\n')
            except OSError:
                pass
        # averaging buffer
        self.frames.append(raw)
        limit = self.avg_n if (self.avg_on and self.avg_rolling) else 200
        if self.avg_on and self.avg_rolling and len(self.frames) > limit:
            del self.frames[:-limit]
        self._refresh()

    # ===================================================================
    # start / pause / single
    # ===================================================================
    def _adapt_plot_range(self, spec: Spectrum):
        """Adopt the instrument's own spectral range — different
        spectrometers report wider/narrower ranges and resolutions, and
        the axis, band spins and rainbow must follow, not assume
        380–730. Called on every frame; no-op when unchanged."""
        if len(spec.wavelengths) == 0:
            return                      # XYZ-only colorimeter frame
        lo, hi = float(spec.wavelengths[0]), float(spec.wavelengths[-1])
        if abs(lo - self.x_lo) < 1e-6 and abs(hi - self.x_hi) < 1e-6:
            return
        was_full = self.band_lo <= self.x_lo + 1e-6 \
            and self.band_hi >= self.x_hi - 1e-6
        self.x_lo, self.x_hi = lo, hi
        old_lo, old_hi = self.band_lo, self.band_hi
        self.spin_lo.setRange(lo, hi)
        self.spin_hi.setRange(lo, hi)
        if was_full or old_lo < lo or old_lo > hi:
            self.band_lo = lo
        if was_full or old_hi > hi or old_hi < lo:
            self.band_hi = hi
        self.spin_lo.setValue(self.band_lo)
        self.spin_hi.setValue(self.band_hi)
        self.ax.set_xlim(lo, hi)
        from .processing import rainbow_palette
        self.rainbow.set_data(rainbow_palette(lo, hi))
        self.rainbow.set_extent((lo, hi, 0.0, 1.0))
        self.rainbow_clip.get_path().vertices = [
            (lo, 0.0), (lo, 0.0), (hi, 0.0), (hi, 0.0), (lo, 0.0)]
        self._diag_write(f"GUI: plot range adapted to instrument: "
                         f"{lo:.1f}–{hi:.1f} nm ({len(spec.wavelengths)} pts)\n")

    def _make_worker(self):
        if self.source_name == "Simulator":
            w = sources.SimulatorWorker(self.events)
            w.profile = "fluorescent"
            return w
        mode = {"Emissive": "e", "Ambient": "a", "Reflective": "r"}[self.mode_name]
        w = sources.SpotreadWorker(self.events, mode=mode,
                                   high_res=self.high_res, fast=self.fast,
                                   port=self.device_port,
                                   force_cal=self.force_cal)
        # calibrate-all is not a user option: it is how the app
        # calibrates, always. Restrict the pass to the modes this
        # instrument actually supports.
        allowed = getattr(self, "allowed_modes", None) \
            or ("Emissive", "Ambient", "Reflective")
        w.cal_pass_modes = tuple(
            {"Emissive": "e", "Ambient": "a", "Reflective": "r"}[m]
            for m in allowed if m in ("Emissive", "Ambient",
                                      "Reflective")) or None
        return w

    def on_start_stop(self):
        # debounce: Qt fires clicked() twice on a double click — the second
        # click must not immediately pause a freshly started session
        now = time.monotonic()
        if now - getattr(self, "_last_toggle", 0.0) < 0.6:
            return
        self._last_toggle = now
        self._diag_write(f"GUI: toolbar toggle (was running={self.running})\n")
        if self.running:
            self.running = False
            self._stop_since = time.monotonic()
            if self.worker and self.worker.is_alive():
                self.worker.pause()
            self.btn_start.setText("RUN")
            self.btn_start.setIcon(self._std_icon("play"))
            self.btn_start.setObjectName("btnStart")
            self._repolish(self.btn_start)
            self.status("Stopped — instrument session kept alive; RUN "
                        "needs no new calibration.")
            # unmistakable grey banner: the loop is HALTED
            self._set_lamps(caption="⏸ STOPPED — no new readings. Press "
                            "RUN to resume (no new calibration needed).")
            return
        expected = "SimulatorWorker" if self.source_name == "Simulator" \
            else "SpotreadWorker"
        if self.worker and self.worker.is_alive() and \
                type(self.worker).__name__ == expected:
            self.worker.resume()
        else:
            if self.worker:
                self.worker.stop()
            self.worker = self._make_worker()
            self.worker.start()
            self.worker.resume()
        self.running = True
        self._begin_recording()

    def _begin_recording(self):
        """Enter the continuous-recording state (toolbar START + launch)."""
        self.worker.start_continuous()
        self._diag_write(
            "GUI: _begin_recording -> continuous="
            f"{int(self.worker.continuous.is_set())} "
            f"wants_read={int(self.worker.wants_read.is_set())} "
            f"armed={getattr(self.worker, '_armed', '?')}\n")
        # a new measurement phase starts now; whether the position lamp
        # should flash is decided centrally from (pending, confirmed)
        if hasattr(self.worker, "arm_probes"):
            self.worker.arm_probes()
        self._update_position_lamp()
        self.btn_start.setText("STOP")
        self.btn_start.setIcon(self._std_icon("stop"))
        self.btn_start.setObjectName("btnRunning")
        self._repolish(self.btn_start)
        self._start_logs()
        self.lbl_device.setText(
            f"Instrument: {sources.instrument_present() or 'spotread'}")
        wheel_hint = {
            "Reflective": "Wheel: 2 = calibrate, then back to 3; hold the "
                          "aperture flat against the surface.",
            "Ambient": "Wheel: 2 = calibrate, then the outer diffuser slot "
                       "(past OFF), diffuser facing the room.",
        }.get(self.mode_name, "Wheel: 2 = calibrate, 3 = measure.")
        self.status(f"STARTED — continuous readings. {wheel_hint}")

    def on_single(self):
        if self.worker and self.worker.is_alive():
            self.worker.request_read()
            if hasattr(self.worker, "arm_probes"):
                self.worker.arm_probes()
        else:
            self.worker = self._make_worker()
            if getattr(self.worker, "wants_read", None) is not None:
                self.worker.wants_read.set()
            self.worker.start()
            self._start_logs()
        self.btn_single.setText("…  READING")
        self.btn_single.setEnabled(False)
        # now the single read is "in flight": re-evaluate the position lamp
        self._update_position_lamp()
        self.status("Single reading requested — waiting for the "
                    "instrument…")
        # absolute fallback: even if the instrument wedges forever, give
        # the button back (the driver also restarts a stalled session)
        self._single_deadline = time.monotonic() + 150.0
        QTimer.singleShot(150_000, self._single_timeout)

    def _single_timeout(self):
        if (not self.btn_single.isEnabled()
                and time.monotonic() >= getattr(self, "_single_deadline", 0)):
            self.btn_single.setText("SINGLE")
            self.btn_single.setEnabled(True)
            self.status("Single reading did not complete in 150 s — check "
                        "the instrument log and the USB cable.")

    def _start_logs(self):
        base = os.path.expanduser("~/spectro-live-logs")
        os.makedirs(base, exist_ok=True)
        if self.diag_path is None and self.inst_log:
            # the full spotread transcript is OPT-IN (Instrument →
            # Write instrument log): nothing is written to disk
            # behind the user's back
            self.diag_path = os.path.join(
                base, f"instrument-log-{datetime.now():%Y%m%d-%H%M%S}.txt")
            self._diag_write(f"=== spectro-live session started "
                             f"{datetime.now():%Y-%m-%d %H:%M:%S} ===\n")
        if self.csv_path is None and self.csv_log:
            self.csv_path = os.path.join(
                base, f"session-{datetime.now():%Y%m%d-%H%M%S}.csv")

    # ===================================================================
    # zero / reference / files
    # ===================================================================
    def on_capture_zero(self):
        if not self.frames:
            QMessageBox.information(self, APP_NAME, "Take a reading first — "
                                    "the current reading becomes the "
                                    "zero/background.")
            return
        had_zero = self.zero_ref is not None
        # capture from the RAW (pre-zero) current reading — averaging is
        # honoured, but neither the previous zero nor any unit
        # conversion applies, so pressing Zero again over the same scene
        # RE-zeros it (display drops to ~0) instead of undoing the
        # first zero. Only "Clear zero" removes the subtraction.
        if self.chk_avg is not None and self.chk_avg.isChecked():
            buf = (self.frames[-self.spin_avg.value():]
                   if self.rb_roll.isChecked() else self.frames)
            base = average(list(buf)) if len(buf) > 1 else buf[-1].copy()
        else:
            base = self.frames[-1].copy()
        if len(base.wavelengths) == 0:
            QMessageBox.information(self, APP_NAME,
                                    "No spectrum to zero against — the "
                                    "current reading came from a "
                                    "colorimeter without spectral data.")
            return
        self.zero_ref = base
        self.zero_ref.label = "zero"
        # say WHAT was captured: with a live stream the "current
        # reading" changes every second, so a mis-captured zero (the
        # scene changed under the click) must be visible at a glance —
        # its peak and its level make the mistake obvious
        try:
            pk_i = int(np.argmax(base.values))
            pk_wl, pk_v = float(base.wavelengths[pk_i]), \
                float(base.values[pk_i])
            detail = f" (peak {pk_v:.3f} at {pk_wl:.0f} nm)"
        except Exception:
            detail = ""
        self._update_zero_button()
        if had_zero:
            self.status(f"Zero RE-captured from the current RAW reading — "
                        f"the previous zero is replaced, never restored"
                        f"{detail}.")
        else:
            self.status(f"Zero captured — the current RAW reading is now "
                        f"the reference for every following measurement, "
                        f"chromaticity and colour coordinate{detail}. "
                        "(Clear zero removes it.)")
        self._refresh()

    def on_clear_zero(self):
        self.zero_ref = None
        self._update_zero_button()
        self.status("Zero removed.")
        self._refresh()

    def _paint_swatch(self, css: str):
        """Repaint the sRGB preview swatch — always through a SCOPED
        rule: the swatch shows tooltips, and a bare rule would repaint
        the tip with the swatch's own colours (see _paint_only)."""
        _paint_only(self.swatch, "swatch", css)

    def _update_zero_button(self):
        """The ZERO button stays highlighted while a zero/background is
        active; Clear zero drops the highlight."""
        b = getattr(self, "btn_zero", None)
        if b is None:
            return
        if self.zero_ref is not None:
            b.setStyleSheet(
                f"QPushButton {{ background: {ACCENT}; color: {ACCENT_FG}; "
                "font-weight: bold; }}")
        else:
            b.setStyleSheet("")

    def on_freeze_ref(self):
        if self.trace_a is None:
            QMessageBox.information(self, APP_NAME, "Take a reading first.")
            return
        self.reference = self.trace_a.copy()
        self.status("Reference B frozen from the live trace.")
        self._refresh()

    def on_clear_ref(self):
        self.reference = None
        self._refresh()

    def on_load_ref(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Load reference spectrum", "",
            "Spectra (*.csv *.sp *.txt *.ti3);;All files (*)")
        if not p:
            return
        try:
            self.reference = sources.load_argyll(p)
            self.status(f"Reference loaded: {os.path.basename(p)}")
            self._refresh()
        except Exception as e:
            QMessageBox.critical(self, APP_NAME, f"Could not load file:\n{e}")

    def on_load_a(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Load live trace", "",
            "Spectra (*.csv *.sp *.txt *.ti3);;All files (*)")
        if not p:
            return
        try:
            self.trace_a = sources.load_argyll(p)
            # kept separately: _refresh() rebuilds trace A from
            # _processed(), which falls back to this file when no
            # frames are arriving (no device / stopped session)
            self.loaded_a = self.trace_a
            self.frames.clear()
            self.status(f"Loaded {os.path.basename(p)} as trace A")
            self.last_raw = self.trace_a
            self._refresh()
        except Exception as e:
            QMessageBox.critical(self, APP_NAME, f"Could not load file:\n{e}")

    def on_save_a(self):
        if self.trace_a is None:
            QMessageBox.information(self, APP_NAME, "Nothing to save yet.")
            return
        p, _ = QFileDialog.getSaveFileName(self, "Save trace A", "",
                                           "CSV (*.csv)")
        if p:
            sources.save_csv(p, self.trace_a,
                             zeroed=self.zero_ref is not None)
            self.status(
                f"Saved {p}" + (" (zero-corrected)" if self.zero_ref
                                else ""))

    def on_save_b(self):
        if self.reference is None:
            QMessageBox.information(self, APP_NAME, "No reference B stored.")
            return
        p, _ = QFileDialog.getSaveFileName(self, "Save reference B", "",
                                           "CSV (*.csv)")
        if p:
            sources.save_csv(p, self.reference)
            self.status(f"Saved {p}")

    def on_export_png(self):
        p, _ = QFileDialog.getSaveFileName(self, "Export plot", "",
                                          "PNG (*.png)")
        if p:
            self.fig.savefig(p, dpi=150, facecolor=self.fig.get_facecolor())
            self.status(f"Saved {p}")

    def on_show_log(self):
        win = QWidget(self, Qt.WindowType.Window)
        win.setWindowTitle("Instrument output")
        win.resize(780, 500)
        v = QVBoxLayout(win)
        t = QTextEdit()
        t.setReadOnly(True)
        t.setPlainText("\n".join(self.log_lines[-3000:]))
        t.ensureCursorVisible()
        v.addWidget(t)
        v_btn = QHBoxLayout()
        b = QPushButton("Refresh")
        b.clicked.connect(lambda b2=None: t.setPlainText(
            "\n".join(self.log_lines[-3000:])))
        v_btn.addWidget(b)
        v_btn.addStretch(1)
        v.addLayout(v_btn, 0)
        win.show()

    def on_help(self):
        QMessageBox.information(
            self, APP_NAME,
            "• <b>START</b>: connect + auto-calibrate, then repeat readings "
            "automatically. Wheel: position 2 = calibrate, 3 = measure.\n"
            "• <b>SINGLE</b>: one reading with the current session.\n"
            "• Left-click on the spectrum plot: marker M1; right-click: M2.\n"
            "• <b>Zero</b> subtracts a stored dark trace from every reading.\n"
            "• <b>Freeze B</b> and the Compare view give A−B / A/B.\n"
            "• Instrument log and CSV files land in ~/spectro-live-logs/.")

    # ===================================================================
    # processing / drawing
    # ===================================================================
    def _convert_units(self, s: Spectrum) -> Spectrum | None:
        # radiometric chart prefix (kW..nW) chosen per trace below; the
        # m default keeps the axis sane before the first frame
        self._rad_prefix = "m"
        if s is None:
            return None
        if self.unit_name == "Photopic (a.u.)":
            s = Spectrum(s.wavelengths,
                         photopic_efficacy(s.wavelengths) * s.values, s.xyz,
                         s.label, s.timestamp, s.source)
        elif self.unit_name == "Reflectance %":
            s = Spectrum(s.wavelengths, s.values * 100.0, s.xyz, s.label,
                         s.timestamp, s.source)
        elif self.unit_name == "Photometric absolute":
            # anchored to the frame's own calibrated Y, in the per-mode
            # scale the Y box uses (cd/m² / lx = π·Y). Reflective has
            # NO photometric spectral quantity (a reflectance factor is
            # dimensionless — V(λ) weighting would silently discount
            # deep blue/red reflectance) -> fall back to the plain
            # reflectance factor curve, which is what Argyll's
            # reflective spectrum already is (0–1 factor).
            if self.mode_name == "Reflective":
                return s
            if s.xyz is not None and s.xyz[1] is not None:
                y = float(s.xyz[1])
                if self.mode_name == "Ambient":
                    y *= math.pi                 # lx: E = π·Y (Argyll's model)
                conv = photometric_absolute(s, y)
                if conv is not None:
                    return conv
            # no calibrated anchor -> no honest absolute view: show
            # nothing rather than a curve with a made-up scale
            return None
        if self.unit_name == "Radiometric absolute":
            # radiometric counterpart of the mode's photometric anchor:
            # radiance W/(sr·m²) for emissive, irradiance W/m² for
            # ambient (lx = π·Y). Reflective has NO radiometric
            # quantity (reflectance is dimensionless) -> fall back to
            # the plain reflectance curve, which is what Argyll's
            # reflective spectrum already is (norm = 100).
            if self.mode_name == "Reflective":
                return s
            if s.xyz is not None and s.xyz[1] is not None:
                y = float(s.xyz[1])
                if self.mode_name == "Ambient":
                    y *= math.pi                # lx = π·Y -> W/m² pair
                conv = radiometric_absolute(s, y)
                if conv is not None:
                    # plot in an adaptive engineering prefix (kW..nW,
                    # the same tiers _fmt_radiometric uses for the
                    # scalar readout) so the y axis keeps comfortable
                    # digits whatever the light level: display-range
                    # densities land in mW, bright sources can climb
                    # to W. The SHAPE is untouched — only the fixed
                    # display scale changes.
                    peak = float(np.max(conv.values)) \
                        if len(conv.values) else 0.0
                    if peak >= 1e3:
                        self._rad_prefix = "k"
                    elif peak >= 1.0:
                        self._rad_prefix = ""
                    elif peak >= 1e-3:
                        self._rad_prefix = "m"
                    elif peak >= 1e-6:
                        self._rad_prefix = "µ"
                    else:
                        self._rad_prefix = "n"
                    conv = Spectrum(s.wavelengths,
                                    conv.values / self._SI[self._rad_prefix],
                                    conv.xyz, conv.label, conv.timestamp,
                                    conv.source)
                    return conv
            return None
        return s

    def _processed(self) -> Spectrum | None:
        if not self.frames:
            # No live data flowing (device absent, or stopped before the
            # first reading): an explicitly loaded file IS trace A. The
            # status bar already promised "Loaded … as trace A" —
            # without this fallback _refresh() would silently wipe it
            # back to none on the very next pass.
            s = getattr(self, "loaded_a", None)
            if s is None:
                return None
            s = s.copy()
            s = self._clip_unreliable(s)
            return self._convert_units(s)
        if self.chk_avg is not None and self.chk_avg.isChecked():
            buf = self.frames[-self.spin_avg.value():] \
                if self.rb_roll.isChecked() else self.frames
            s = average(buf) if len(buf) > 1 else buf[-1].copy()
        else:
            # averaging OFF: the trace is the newest reading, nothing else
            s = self.frames[-1].copy()
        if len(s.wavelengths) == 0:
            # XYZ-only colorimeter frame: no spectral maths applies, the
            # colour coordinates below carry everything
            return s
        if self.zero_ref is not None:
            s = subtract(s, self.zero_ref)
            s.label = "A (zero-corrected)"
            # the stored zero changes what the sensor reading MEANS: all
            # computations downstream (integral, peaks, chromaticity,
            # colour coordinates, exports) treat the corrected curve as
            # the new reference. Recompute XYZ from it, and drop the
            # instrument's own meta (CCT/Duv/Lab/CRI belong to the
            # UN-corrected reading — no raw fallbacks while zeroed).
            s.meta_cct = None
            s.meta_duv = None
            s.meta_lab = None
            s.meta_cri = None
            s.meta_tm30 = None
            # recompute XYZ from the corrected curve (when the clip
            # below actually removes bins, it recomputes again from
            # the trusted band — same number, honest either way)
            s.xyz = xyz_from_values(s)
        # clip the UNRELIABLE band last (and only now): subtract()
        # aligns on the UNION of the two wavelength grids, so the
        # raw full-band zero reference would interpolate its unstable
        # leading bins back into an already-clipped trace — silently
        # undoing the exclusion. Clipping the corrected result keeps
        # the zeroed curve inside the trusted band, and _clip_unreliable
        # recomputes XYZ from exactly the band that is displayed.
        s = self._clip_unreliable(s)
        if self.zero_ref is not None:
            s.label = "A (zero-corrected)"
        if self.chk_smooth is not None and self.chk_smooth.isChecked() \
                and self.spin_smooth.value() >= 3:
            s = smooth(s, self.spin_smooth.value())
        return self._convert_units(s)

    def _update_unreliable_span(self):
        """Bordeaux shading over the untrustworthy first bins — always
        shown for instruments where the boundary is known (even when
        the data is clipped, so the region stays identified), and
        converted correctly for the wavelength OR frequency axis (the
        span flips to the chart's right in THz mode)."""
        sp = getattr(self, "unreliable_span", None)
        if sp is not None:
            sp.remove()
        lo = unreliable_band_lo(getattr(self, "inst_type", ""),
                                self.mode_name)
        if lo is None or lo <= self.x_lo:
            self.unreliable_span = None
            return
        x0, x1 = float(self.x_lo), min(float(lo), self.x_hi)
        xa, xb = sorted(float(v) for v in
                        np.asarray(self._x_vals(np.array([x0, x1]))))
        self.unreliable_span = self.ax.axvspan(
            xa, xb, facecolor="#6d1a2d", alpha=0.45,
            edgecolor="none", zorder=1)

    def _clip_unreliable(self, s: Spectrum) -> Spectrum:
        """Drop the instrument's UNRELIABLE leading bins (ColorMunki:
        Argyll duplicates the 400 nm filter below 400 nm — 420 nom
        reflective — so those values are clones, not measurements).
        Off (or an instrument without a vetted boundary) = unchanged.
        On: the chart, every measurement, the colour data (XYZ
        recomputed from the trusted band) and the exports all reflect
        only the trusted region."""
        lo = (unreliable_band_lo(getattr(self, "inst_type", ""),
                                 self.mode_name)
              if getattr(self, "clip_unreliable", False) else None)
        if lo is None or s is None or len(s.wavelengths) == 0:
            return s
        keep = np.asarray(s.wavelengths) >= lo - 1e-9
        if bool(np.all(keep)):
            return s
        out = Spectrum(s.wavelengths[keep], s.values[keep], None,
                       s.label, s.timestamp, s.source)
        out.xyz = xyz_from_values(out)
        return out

    def _get_band(self):
        if self.chk_mband.isChecked():
            return min(self.marker1, self.marker2), max(self.marker1, self.marker2)
        return min(self.spin_lo.value(), self.spin_hi.value()), \
            max(self.spin_lo.value(), self.spin_hi.value())

    def _unit_labels(self) -> tuple[str, str]:
        """(axis label, integral unit) for the selected trace unit,
        with photometric units that follow the reading mode: what the
        sensor physically sees is luminance (cd/m²) from a screen,
        illuminance (lx) on the ambient diffuser, or a dimensionless
        reflectance factor from a surface."""
        if self.unit_name == "Reflectance %":
            return "Reflectance (%)", "%"
        if self.unit_name == "Photometric absolute":
            axis = {"Emissive": "Spectral luminance (cd/m²·nm⁻¹)",
                    "Ambient": "Spectral illuminance (lx·nm⁻¹)",
                    "Reflective": "Reflectance (%)"}
            unit = {"Emissive": "cd/m²", "Ambient": "lx",
                    "Reflective": "%"}[self.mode_name]
            return axis.get(self.mode_name, "Intensity (a.u.)"), unit
        if self.unit_name == "Radiometric absolute":
            axis = {"Emissive":
                        f"Spectral radiance "
                        f"({self._rad_prefix}W/(sr·m²)·nm⁻¹)",
                    "Ambient":
                        f"Spectral irradiance "
                        f"({self._rad_prefix}W/m²·nm⁻¹)",
                    "Reflective": "Reflectance (%)"}
            unit = {"Emissive": f"{self._rad_prefix}W/(sr·m²)",
                    "Ambient": f"{self._rad_prefix}W/m²",
                    "Reflective": "%"}[self.mode_name]
            return axis.get(self.mode_name, "Intensity (a.u.)"), unit
        if self.unit_name == "Photopic (a.u.)":
            # V(λ)-weighted SHAPE view: eye weighting on the
            # instrument's own arbitrary scale — a.u. by mode-agnostic
            # definition (the raw counts carry no absolute calibration,
            # so no photometric unit is claimed)
            return "Photopic weighting (a.u./nm)", "a.u."
        return "Intensity (a.u.)", "a.u."

    def _refresh(self):
        self.trace_a = self._processed()
        axis_label, _u = self._unit_labels()
        self.ax.set_ylabel(axis_label, color=FG, fontsize=CHART_AXIS)
        self._redraw()
        self._update_stats()

    def _redraw(self):
        ax, a, b = self.ax, self.trace_a, self.reference
        mode = self.view_name
        show_a, show_b, hint = a, b, ""
        if mode.startswith("Difference"):
            if a and b:
                show_a, show_b = subtract(a, b), None
            elif a:
                show_a, show_b, hint = a, None, \
                    "No reference B — showing live trace A"
        elif mode.startswith("Ratio"):
            if a and b:
                show_a, show_b = ratio_pct(a, b), None
            elif a:
                show_a, show_b, hint = a, None, \
                    "No reference B — showing live trace A"

        for line, spec in ((self.line_a, show_a), (self.line_b, show_b)):
            if spec is not None:
                xs = self._x_vals(spec.wavelengths)
                if self.curve_style == "steps":
                    line.set_drawstyle("steps-post")
                    line.set_data(xs, spec.values)
                elif self.curve_style == "smooth" and len(spec.values) >= 3:
                    line.set_drawstyle("default")
                    xs, ys = pchip_smooth(xs, spec.values)
                    line.set_data(xs, ys)
                else:
                    line.set_drawstyle("default")
                    line.set_data(xs, spec.values)
                line.set_visible(True)
            else:
                line.set_visible(False)

        lo, hi = self._get_band()
        if self.x_mode in ("freq", "ev"):
            # ascending frequency/energy runs opposite to ascending
            # wavelength
            flo, fhi = sorted((float(self._x_vals(np.array([lo]))[0]),
                              float(self._x_vals(np.array([hi]))[0])))
        else:
            flo, fhi = lo, hi
        if self.bandspan is not None:
            self.bandspan.remove()
        for ln in self.band_edges:
            ln.remove()
        full = (hi - lo) > (self.x_hi - self.x_lo) - 1e-6
        if full:
            self.bandspan, self.band_edges = None, []
        else:
            self.bandspan = ax.axvspan(flo, fhi, color=ACCENT, alpha=0.20,
                                       zorder=1)
            self.band_edges = [
                ax.axvline(flo, color=ACCENT, lw=1.4, alpha=0.9, zorder=2),
                ax.axvline(fhi, color=ACCENT, lw=1.4, alpha=0.9, zorder=2)]

        self.marker_lines[0].set_xdata(
            [float(self._x_vals(np.array([self.marker1]))[0])])
        self.marker_lines[1].set_xdata(
            [float(self._x_vals(np.array([self.marker2]))[0])])
        self._update_unreliable_span()

        all_vals = []
        if show_a is not None:
            all_vals = np.concatenate([all_vals, show_a.values])
        if show_b is not None:
            all_vals = np.concatenate([all_vals, show_b.values])
        if len(all_vals):
            vmin, vmax = float(all_vals.min()), float(all_vals.max())
            pad = 0.06 * (vmax - vmin) or 0.5
            if vmax <= vmin:
                pad = max(abs(vmax) * 0.1, 0.5)
        if self.x_mode in ("freq", "ev"):
            # reciprocal axis (f = c/λ or E = hc/λ): ascending energy
            # runs opposite to ascending wavelength, so the axis is
            # λ_hi .. λ_lo upside down and the rainbow image columns —
            # generated for ascending wavelength — flip to match
            # (red-low-energy left, violet-high-energy right).
            k = (self._C_NM_THZ if self.x_mode == "freq"
                 else self._HC_NM_EV)
            fx0, fx1 = k / self.x_hi, k / self.x_lo
            pal = rainbow_palette(self.x_lo, self.x_hi)[:, ::-1, :]
        else:
            fx0, fx1 = self.x_lo, self.x_hi
            pal = rainbow_palette(self.x_lo, self.x_hi)
        self.rainbow.set_data(pal)
        self.rainbow.set_extent((fx0, fx1, 0.0, 1.0))
        ax.set_xlim(fx0, fx1)
        if len(all_vals):
            # y axis always starts at the bottom: 0 (or exactly the
            # lowest recorded value when the trace dips below zero)
            ax.set_ylim(0.0 if vmin >= 0 else float(vmin), vmax + pad)

        src = show_a if show_a is not None else show_b
        # NOTE: the rainbow clip polygon lives in get_xaxis_transform()
        # coordinates, where y is a FRACTION of the axis (0..1), so the
        # data values must be converted — this is what makes the colour
        # fill follow the curve exactly, including negative values (the
        # fill reaches all the way down to the axis floor).
        y0, y1 = ax.get_ylim()
        span = (y1 - y0) or 1.0
        def style_polygon(spec):
            """(xs, ys) of the curve the way it is actually DISPLAYED:
            smooth interpolation, steps, or raw points — so the rainbow
            fill tracks the drawn curve exactly."""
            if spec is None or len(spec.values) == 0:
                return spec.wavelengths, spec.values
            xs = self._x_vals(spec.wavelengths)
            if self.curve_style == "smooth" and len(spec.values) >= 3:
                return pchip_smooth(xs, spec.values)
            if self.curve_style == "steps":
                xs2 = np.repeat(xs, 2)[1:]
                ys = np.repeat(spec.values, 2)[:-1]
                return np.append(xs2, xs[-1]), \
                    np.append(ys, spec.values[-1])
            return xs, spec.values

        if src is not None:
            def frac(v):
                f = (float(v) - y0) / span
                return f if f > 0.0 else 0.0
            xs, ys = style_polygon(src)
            xs = np.asarray(xs, dtype=float)
            ys = np.asarray(ys, dtype=float)
            if len(xs) == 0:
                # XYZ-only colorimeter frame: no curve exists — keep
                # the rainbow fully clamped (no fill at all)
                verts = [(fx0, 0.0), (fx1, 0.0),
                         (fx1, 0.0), (fx0, 0.0)]
            else:
                # the fill hugs the DISPLAYED curve and extends from it
                # straight DOWN to the bottom of the plot area — no
                # matter where the 0 line sits or how the trace
                # starts/ends. Drop verticals at both ends of the data
                # (never diagonals to the plot corners), and work in
                # ascending-x order so the polygon is well-formed even
                # on the frequency axis.
                if xs[0] > xs[-1]:
                    xs, ys = xs[::-1], ys[::-1]
                verts = ([(float(xs[0]), 0.0), (float(xs[0]), frac(ys[0]))]
                         + [(float(x), frac(y)) for x, y in zip(xs[1:], ys[1:])]
                         + [(float(xs[-1]), 0.0)])
        else:
            verts = [(fx0, 0.0), (fx1, 0.0),
                     (fx1, 0.0), (fx0, 0.0)]
        self.rainbow_clip.set_path(Path(verts))
        if hint:
            self.status(hint)
        # margin fit AFTER the labels are known (unit selection /
        # data range change the y tick text width)
        self._fit_chart_margins()
        self.canvas.draw_idle()

    def _update_stats(self):
        a = self.trace_a
        lo, hi = self._get_band()
        if a is None:
            self.val_integral.setText("—")
            self.val_peak.setText("—")
            self.val_markers.setText("—")
            self.val_photometric.setText("—")
            self.val_radiometric.setText("—")
            self.val_cct.setText("—")
            self.val_ri.setText("—")
            self.val_colour.setText("—")
            self._paint_swatch(
                f"background: {FIELD}; border: 2px solid {BORDER};")
            self.list_peaks.clear()
            self.peak_marks.set_visible(False)
            self._update_chrom(None)
            return
        _axis, yunit = self._unit_labels()
        if a.xyz is None:
            self.val_colour.setText("—")
            self.val_radiometric.setText("—")
            self._paint_swatch(
                f"background: {FIELD}; border: 2px solid {BORDER};")
            self._update_chrom(None)
            return
        self._update_chrom(a.xyz)
        if len(a.wavelengths) == 0:
            # XYZ-only (colorimeter): no spectrum exists — grey out the
            # spectral numbers, keep the colour section
            self.val_integral.setText("— (no spectrum)")
            self.val_peak.setText("—")
            self.val_markers.setText("—")
            self.val_radiometric.setText("—  (needs a spectrum)")
            self.list_peaks.clear()
            self.peak_marks.set_visible(False)
        else:
            I = integrate(a, lo, hi)
            self.val_integral.setText(f"{I:,.1f} {yunit}·nm   "
                                      f"({lo:.0f}–{hi:.0f} nm)")
            p = peak_info(a, lo, hi)
            self.val_peak.setText(
                f"λp {p['wl']:.1f} nm · {p['val']:.4f}\n"
                f"FWHM {p['fwhm']:.1f} nm · centroid {p['centroid']:.1f} nm")
            v1, v2 = a.at(self.marker1), a.at(self.marker2)
            self.val_markers.setText(
                f"M1 {self.marker1:.1f} nm {v1:.4f}\n"
                f"M2 {self.marker2:.1f} nm {v2:.4f}\n"
                f"Δλ {abs(self.marker2 - self.marker1):.1f} nm · "
                f"Δ {v2 - v1:+.4f}")

            # peaks: strongest first (the order IS the ranking — no
            # index numbers), wavelength leading each row
            peaks = find_peaks(a, prominence=0.04, n_max=6)
            self.list_peaks.clear()
            wls, vs = [], []
            for i, (w, _v) in enumerate(peaks, 1):
                info = peak_info(a, w - 15.0, w + 15.0)
                self.list_peaks.addItem(
                    f"{w:7.1f} nm  E {a.at(w):9.4f}   "
                    f"FWHM {info['fwhm']:6.1f} nm")
                wls.append(w)
                vs.append(a.at(w))
            for ann in self.peak_annots:
                ann.remove()
            self.peak_annots = []
            self.peak_marks.set_data(self._x_vals(np.array(wls)), vs)
            self.peak_marks.set_visible(bool(wls))
            for i, (w, v) in enumerate(zip(wls, vs), 1):
                self.peak_annots.append(self.ax.annotate(
                    str(i), (float(self._x_vals(np.array([w]))[0]), v), xytext=(2, 8),
                    textcoords="offset points",
                    color=FG, fontsize=CHART_ANNO, fontweight="bold", ha="center",
                    zorder=6))

        # colour coordinates
        lab = getattr(a, "meta_lab", None)
        if lab is None and (self.last_raw is not None
                          and self.zero_ref is None):
            lab = getattr(self.last_raw, "meta_lab", None)
        # instrument measurements that stay the same across coordinate
        # systems: photometric Y (unit depends on reading mode), CCT and
        # — for light sources — the colour rendering index
        if a.xyz is None:
            self.val_photometric.setText("—")
            self.val_cct.setText("—")
            self.val_ri.setText("—")
            self.val_colour.setText("—")
            self._paint_swatch(
                f"background: {FIELD}; border: 2px solid {BORDER};")
            return
        cct = getattr(a, "meta_cct", None)
        if cct is None and (self.last_raw is not None
                          and self.zero_ref is None):
            cct = getattr(self.last_raw, "meta_cct", None)
        duv = getattr(a, "meta_duv", None)
        if duv is None and (self.last_raw is not None
                          and self.zero_ref is None):
            duv = getattr(self.last_raw, "meta_duv", None)
        cct_caution = ""
        if cct is None:
            # McCamy's cubic is a fit to the Planckian locus over
            # ~2856–25000 K; printed unqualified outside that range,
            # the number is fiction. Mark it instead of hiding it.
            cct = cct_from_xyz(a.xyz)
            if not (2856.0 <= cct <= 25000.0):
                cct_caution = ("\n(computed — outside McCamy's\n"
                               "fit range, treat as rough)")
        x, y = xy_from_xyz(a.xyz)
        if self.mode_name == "Reflective":
            self.val_photometric._head.setText("REFLECTANCE Y")
            self.val_photometric.setText(f"{a.xyz[1]*100:.2f} %")
        elif self.mode_name == "Ambient":
            self.val_photometric._head.setText("ILLUMINANCE Y")
            # E = π·Y: Argyll's own ambient model (a Lambertian field
            # through the diffuser) — see spotread.c
            self.val_photometric.setText(
                f"{a.xyz[1]*math.pi:,.1f} lx  (E = π·Y)")
        else:
            self.val_photometric._head.setText("LUMINANCE Y")
            self.val_photometric.setText(f"{a.xyz[1]:,.2f} cd/m²")
        # the radiometric counterpart of the photometric figure —
        # irradiance W/m² on the ambient diffuser (pairs with the lx
        # above), radiance W/(sr·m²) for an emitting spot (pairs with
        # the cd/m² above). Anchored to the same absolute Y; the
        # spectrum supplies only the shape. A surface has no honest
        # radiometric value (a reflectance factor is dimensionless).
        if self.mode_name == "Reflective":
            self.val_radiometric._head.setText("RADIOMETRIC")
            self.val_radiometric.setText(
                "(n/a for surfaces)")
        elif len(a.wavelengths) == 0:
            self.val_radiometric._head.setText("RADIOMETRIC")
            self.val_radiometric.setText("—  (needs a spectrum)")
        else:
            if self.mode_name == "Ambient":
                self.val_radiometric._head.setText("IRRADIANCE Eₑ")
                anchor = a.xyz[1] * math.pi       # lx (Argyll's model)
                unit = "W/m²"
                pair = "E = π·Y anchor"
            else:
                self.val_radiometric._head.setText("RADIANCE Lₑ")
                anchor = a.xyz[1]                  # cd/m²
                unit = "W/(sr·m²)"
                pair = "Y anchor"
            er = radiometric_from_photometric(a, anchor if anchor > 0
                                              else None)
            if er is None:
                self.val_radiometric.setText("—")
            else:
                self.val_radiometric.setText(
                    self._fmt_radiometric(er, unit, pair))
        if cct:
            txt = f"{cct:,.0f} K"
            if duv is not None:
                txt += f" · Duv {duv:+.4f}"
                if abs(duv) > 0.006:
                    txt += "\n(off the Planckian locus —"
                    txt += "\n CCT only loosely correlated)"
            txt += cct_caution
            self.val_cct.setText(txt)
        else:
            self.val_cct.setText("—")
        # CRI: only for actual light sources, and needs the spectrum
        if self.mode_name == "Reflective":
            self.val_ri.setText("(n/a for surfaces)")
        elif len(a.wavelengths) == 0 and getattr(a, "meta_cri", None) is None \
                and (self.last_raw is None or
                     getattr(self.last_raw, "meta_cri", None) is None):
            self.val_ri.setText("—  (needs a spectrum)")
        else:
            mcri = getattr(a, "meta_cri", None)
            if mcri is None and (self.last_raw is not None
                              and self.zero_ref is None):
                mcri = getattr(self.last_raw, "meta_cri", None)
            mtm30 = getattr(a, "meta_tm30", None)
            if (mtm30 is None and self.last_raw is not None
                    and self.zero_ref is None):
                mtm30 = getattr(self.last_raw, "meta_tm30", None)
            if mcri is not None:
                # spotread's own value wins — it carries Argyll's validity
                # flag (truncated spectral range etc.)
                r9 = (f"   ·   R9 {mcri['r9']:,.0f}"
                      if mcri["r9"] is not None else "")
                flag = (f"  ({mcri['flag'].lower()})"
                        if mcri["flag"] else "")
                line = f"Ra {mcri['ra']:,.1f}{r9}{flag}"
            else:
                cri = cri_from_spectrum(a)
                if cri is None:
                    line = "—"
                else:
                    r9 = (f"   ·   R9 {cri['R9']:,.0f}"
                          if cri["R9"] is not None else "")
                    line = f"Ra {cri['Ra']:,.1f}{r9}"
            # IES TM-30-15 rides the same reading when the instrument
            # reported it (no computed fallback: the 99-sample fidelity
            # engine is Argyll's, not ours — absent means absent)
            if mtm30 is not None:
                tflag = (f"  ({mtm30['flag'].lower()})"
                         if mtm30["flag"] else "")
                line += (f"\nRf {mtm30['rf']:,.1f}   ·   "
                         f"Rg {mtm30['rg']:,.1f}{tflag}")
            self.val_ri.setText(line)
        sys_name = self.cmb_coord.currentText()
        if sys_name == "xyY (CIE 1931)":
            ctext = f"xy {x:.4f} {y:.4f}\nY {a.xyz[1]:.5f}"
        elif sys_name == "CIELAB (D50)":
            if lab is not None:
                ctext = (f"L* {lab[0]:.2f}\na* {lab[1]:+.2f}   "
                         f"b* {lab[2]:+.2f}\n(D50, from instrument)")
            else:
                ctext = "Lab not reported by the\ninstrument for this reading"
        elif sys_name == "LCh(ab) (D50)":
            if lab is not None:
                L, a_, b_ = lab
                C = math.hypot(a_, b_)
                h = math.degrees(math.atan2(b_, a_)) % 360.0
                ctext = f"L* {L:.2f}   C* {C:.2f}\nh°ab {h:.1f}   (D50)"
            else:
                ctext = "LCh not available\n(needs instrument Lab)"
        elif sys_name == "XYZ raw":
            ctext = (f"X {a.xyz[0]:.5f}\nY {a.xyz[1]:.5f}   "
                     f"Z {a.xyz[2]:.5f}")
        else:
            dwl, dominant = dominant_wavelength(a.xyz)
            if dwl is None:
                ctext = "not defined (near white point)"
            else:
                tag = "dominant" if dominant else "complementary"
                ctext = (f"λd {dwl:.0f} nm ({tag})\n"
                         f"xy {x:.4f} {y:.4f}")
        self.val_colour.setText(ctext)
        # sRGB preview: Emissive/Ambient show the chromaticity at maximum
        # brightness (light sources: their absolute luminance is not
        # screenable). Reflective shows the surface HONESTLY: the swatch
        # emits light in proportion to the measured reflectance — a 100 %
        # white is 255,255,255 and a 50 % surface emits exactly half the
        # photons (~code 188; codes ≠ light because of display gamma).
        # Fallback: if a channel would exceed the sRGB range, the preview
        # is renormalised to max brightness instead of clipping.
        if self.mode_name == "Reflective":
            lin_max = float(linear_channels(a.xyz).max())
            if lin_max > 1.002:       # 0.2% tol: white noise won't boost
                rgb, oog, over, black = srgb_preview(a.xyz)
                note = (f"surface reflectance Y = {a.xyz[1] * 100:.1f} % "
                        "(brightest channel boosted into sRGB range)")
            else:
                rgb, oog, over, black = srgb_preview(a.xyz, normalize=False)
                note = (f"surface reflectance Y = {a.xyz[1] * 100:.1f} % "
                        "— swatch brightness = reflectance")
        else:
            rgb, oog, over, black = srgb_preview(a.xyz)
            note = f"measured luminance Y = {a.xyz[1]:.2f} cd/m²"
        # the 8-bit values next to the preview label: red when the
        # chromaticity is outside the sRGB gamut (mirrors the border)
        self.lbl_srgb_vals.setText(f"({rgb[0]}, {rgb[1]}, {rgb[2]})")
        _paint_only(self.lbl_srgb_vals, "srgbVals",
                    f"color: {HOT if (oog and not black) else FG};"
                    " font-weight: bold;")
        border = HOT if oog else BORDER
        border_w = 4 if oog else 2
        self._paint_swatch(
            f"background: rgb({rgb[0]},{rgb[1]},{rgb[2]}); "
            f"border: {border_w}px solid {border}; border-radius: 6px;")
        if black:
            self.swatch.setToolTip(
                wrap_tip("No meaningful colour — the measurement is too dark "
                f"({note}).\nTake a reading with more light."))
        elif oog:
            g = gamut_report(a.xyz)
            if g["Adobe RGB"]:
                where = "within Adobe RGB"
            elif g["Display P3"]:
                where = "within Display P3"
            elif g["Rec. 2020"]:
                where = "within Rec. 2020"
            else:
                where = "outside Rec. 2020"
            self.swatch.setToolTip(
                wrap_tip(f"sRGB preview: rgb({rgb[0]}, {rgb[1]}, {rgb[2]}) "
                "(channels clipped).\n"
                f"The measured colour exceeds the sRGB gamut ({where}).\n"
                f"Preview {note}. The red border marks out-of-gamut."))
        else:
            self.swatch.setToolTip(
                wrap_tip(f"sRGB preview: rgb({rgb[0]}, {rgb[1]}, {rgb[2]}).\n"
                f"The measured colour is inside the sRGB gamut.\n"
                f"Preview {note}."))

    # ===================================================================
    def closeEvent(self, e):
        try:
            if self.worker:
                self.worker.stop()
                self.worker.join(timeout=3)
        except Exception:
            pass
        super().closeEvent(e)


def run(source=None, start=False):
    app = QApplication([APP_NAME])
    app.setStyle("Fusion")
    # styled_qss: the QSS @ARROW_*@ tokens become real image paths, so
    # the combo/spin arrows are painted from our own chevrons — the
    # only arrow mechanism that works identically on every platform
    # once a stylesheet rules the widget (style fallbacks are not).
    app.setStyleSheet(styled_qss())
    # the app-wide QFont comes from the SAME base as the QSS's
    # font-size rule (QSS wins on styled widgets; this still governs
    # tooltips and anything that slips past the stylesheet)
    font = app.font()
    font.setPointSize(int(FONT_BASE.rstrip("pt")))
    app.setFont(font)
    win = MainWindow()
    if source == "simulator":
        win.source_name = "Simulator"
        win.act_source.setText("Source: Simulator  (CLI testing only)")
    win.show()
    # --start needs no action here: MainWindow.__init__() auto-connects
    # the instrument and enters continuous recording on launch. Calling
    # on_start_stop() from here would see running=True and PAUSE the
    # freshly auto-started session (the old --start bug).
    app.exec()
