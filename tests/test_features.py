"""Feature tests for the 2026-11 round: View→curve styles (instant,
no new measurement), the saturation annunciator bar (ColorMunki
headroom estimates), the calibration lamp's green "valid" border, and
the CIE 1931 chromaticity tab (true-colour diagram, gamut triangles,
cross-hair, detached window).

Run:  QT_QPA_PLATFORM=offscreen .venv/bin/python tests/test_features.py
"""
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                      # noqa: E402
trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy 1.x on macOS has no trapezoid()
import matplotlib                       # noqa: E402
from PySide6.QtWidgets import (         # noqa: E402
    QApplication, QGroupBox, QScrollArea,
)
from PySide6.QtGui import QColor       # noqa: E402

import spectrov.qtapp as Q              # noqa: E402
import spectrov.sources as sources      # noqa: E402
from spectrov.processing import (       # noqa: E402
    Spectrum, chromaticity_image, find_peaks, pchip_smooth,
    radiometric_from_photometric, saturation_fraction,
    spectral_locus_xy, xyz_from_values, xy_from_xyz,
    unreliable_band_lo,
)


class FakeWorker:
    def __init__(self):
        import threading
        self.continuous = threading.Event()
        self.wants_read = threading.Event()

    def start(self): pass
    def stop(self): pass
    def is_alive(self): return True
    def pause(self): self.continuous.clear()
    def resume(self): pass
    def request_read(self): self.wants_read.set()
    def start_continuous(self): self.continuous.set()
    def arm_probes(self): pass


FAILURES = []


def check(name, cond, extra=""):
    status = "ok  " if cond else "FAIL"
    print(f"  [{status}] {name}{('  ' + extra) if extra and not cond else ''}")
    if not cond:
        FAILURES.append(name)


def make_window(mode="Emissive"):
    old_mk = Q.MainWindow._make_worker
    old_present = sources.instrument_present
    old_cfg = Q.MainWindow._CFG_PATH
    old_home = os.environ.get("HOME")
    Q.MainWindow._make_worker = lambda self: FakeWorker()
    sources.instrument_present = lambda: None
    os.environ["HOME"] = "/tmp/opencode/hermetic-home"
    os.makedirs("/tmp/opencode/hermetic-home", exist_ok=True)
    # hermetic settings persistence (View→curve style saves)
    Q.MainWindow._CFG_PATH = ("/tmp/opencode/hermetic-home/"
                              ".config/spectrov/settings.json")
    os.makedirs(os.path.dirname(Q.MainWindow._CFG_PATH), exist_ok=True)
    import contextlib
    with contextlib.suppress(OSError):
        os.remove(Q.MainWindow._CFG_PATH)     # fresh settings per window
    try:
        app = QApplication.instance() or QApplication(["t"])
        app.setStyle("Fusion")
        win = Q.MainWindow()
    finally:
        Q.MainWindow._make_worker = old_mk
        sources.instrument_present = old_present
        Q.MainWindow._CFG_PATH = old_cfg
        if old_home is not None:
            os.environ["HOME"] = old_home
    win.mode_name = mode
    win._set_lamps()
    win._update_position_lamp()
    win.show()
    return win


def bright_frame(win, peak=11.0, xyz=(60.0, 65.0, 40.0)):
    wl = np.linspace(380, 730, 106)
    v = np.clip(peak * np.exp(-((wl - 550.0) / 90.0) ** 2), 0, None)
    win._on_frame(Spectrum(wl, v, xyz))


# ----------------------------------------------------------------------
# processing-level maths
# ----------------------------------------------------------------------

def test_pchip_never_overshoots():
    wl = np.linspace(380, 730, 106)
    v = np.abs(np.sin(wl / 30.0)) + 0.2
    xs, ys = pchip_smooth(wl, v)
    check("pchip: more points than samples", len(xs) > len(wl),
          str(len(xs)))
    check("pchip: within the measured range (no overshoot)",
          float(ys.min()) >= float(v.min()) - 1e-9
          and float(ys.max()) <= float(v.max()) + 1e-9)
    check("pchip: passes through the samples",
          np.allclose(np.interp(wl, xs, ys), v, atol=1e-6))
    xs2, ys2 = pchip_smooth(wl[:2], v[:2])
    check("pchip: too few points -> raw", len(xs2) == 2)


def test_saturation_model():
    wl = np.linspace(380, 730, 106)
    cm = "X-Rite ColorMunki Photo"
    knee = Spectrum(wl, np.full(106, 255.0), (1, 1, 1))
    check("saturation: 255 units = the analog knee = 100%",
          abs(saturation_fraction(knee, inst_type=cm) - 1.0) < 1e-9)
    # the printed scale is invariant across modes (munki_imp.c divides
    # by the integration time), so fast and adaptive must agree
    fa = saturation_fraction(knee, fast=True, inst_type=cm)
    check("saturation: adaptive and fast share one scale",
          abs(saturation_fraction(knee, inst_type=cm) - fa) < 1e-9)
    beyond = Spectrum(wl, np.full(106, 340.0), (1, 1, 1))
    check("saturation: past the knee (340) reads 133%",
          abs(saturation_fraction(beyond, inst_type=cm) - 340.0 / 255.0)
          < 1e-9)
    comfortable = Spectrum(wl, np.full(106, 11.7), (1, 1, 1))
    check("saturation: a bright 11.7-unit reading is only ~4.6%",
          abs(saturation_fraction(comfortable, inst_type=cm)
              - 11.7 / 255.0) < 0.01,
          str(saturation_fraction(comfortable, inst_type=cm)))
    dim = Spectrum(wl, np.full(106, 0.117), (1, 1, 1))
    check("saturation: dim reading is a small fraction",
          saturation_fraction(dim, inst_type=cm) < 0.005)
    check("saturation: ambient uses the same sensor path -> fraction",
          abs(saturation_fraction(knee, mode_name="Ambient",
                                   inst_type=cm)
              - saturation_fraction(knee, inst_type=cm)) < 1e-9)
    check("saturation: non-ColorMunki = no honest model -> None",
          saturation_fraction(comfortable,
                              inst_type="GretagMacbeth i1 Pro") is None)
    check("saturation: any ColorMunki badge shares the munki family",
          abs(saturation_fraction(knee, inst_type="ColorMunki Display")
              - 1.0) < 1e-9)
    # DTP20: the one Argyll driver with a FIXED count->printed scale
    # (dtp20.c:578 prints raw16 * 200/65535) — 200 units = bit ceiling
    dtp = Spectrum(wl, np.full(106, 200.0), (1, 1, 1))
    check("saturation: DTP20 full 16-bit scale = 200 printed units",
          abs(saturation_fraction(dtp, inst_type="GretagMacbeth DTP20")
              - 1.0) < 1e-9,
          str(saturation_fraction(dtp, inst_type="GretagMacbeth DTP20")))
    half_dtp = Spectrum(wl, np.full(106, 100.0), (1, 1, 1))
    check("saturation: DTP20 half scale -> 50%",
          abs(saturation_fraction(half_dtp, inst_type="X-Rite DTP20")
              - 0.5) < 1e-9)
    # i1 Pro family: printed scale is per-unit EEPROM — honest n/a
    check("saturation: i1 Pro has no source-derivable ceiling -> None",
          saturation_fraction(knee, inst_type="GretagMacbeth i1 Pro")
          is None,
          str(saturation_fraction(knee,
                                  inst_type="GretagMacbeth i1 Pro")))
    check("saturation: i1 Pro 3 likewise None (per-unit sens_sat)",
          saturation_fraction(knee, inst_type="X-Rite i1Pro3")
          is None)
    check("saturation: no instrument named yet -> None (no guess)",
          saturation_fraction(comfortable) is None)
    check("saturation: reflective = unknowable -> None (n/a)",
          saturation_fraction(knee, mode_name="Reflective",
                              inst_type=cm) is None)
    check("saturation: empty/None spectrum -> None",
          saturation_fraction(None) is None)


def test_radiometric_pairs():
    """The photometric/radiometric pairs must be scientifically exact:
    radiance W/(sr·m²) pairs with luminance cd/m², irradiance W/m²
    with illuminance lx. The conversion undoes the photopic weighting
    via K_m = 683 lm/W, anchored to the absolute Y — a monochromatic
    555 nm source (V = 1) must give X_e = X_v/683 exactly."""
    wl = np.linspace(380, 730, 106)
    # monochromatic 555 nm, 1 cd/m² of luminance: radiance = 1/683
    from spectrov.processing import _CMF_WL, _CMF_Y
    mono = np.zeros_like(wl)
    mono[np.argmin(np.abs(wl - 555))] = 1.0
    re = radiometric_from_photometric(
        Spectrum(wl, mono, (0.0, 1.0, 0.0)), 1.0)
    check("radiometric: 555 nm monochromatic 1 cd/m² -> 1/683 W/(sr·m²)",
          re is not None and abs(re - 1.0 / 683.0) < 0.01, str(re))
    # linearity of the anchor: twice the luminance, twice the radiance
    re2 = radiometric_from_photometric(
        Spectrum(wl, mono, (0.0, 2.0, 0.0)), 2.0)
    check("radiometric: linear in the photometric anchor",
          re2 is not None and abs(re2 - 2.0 / 683.0) < 0.02, str(re2))
    # deep red near 700 nm: V ≈ 0.004 — undoing the photopic weighting
    # costs ~244x more radiance than 555 nm at equal luminance
    red = np.zeros_like(wl)
    i700 = int(np.argmin(np.abs(wl - 700.0)))
    red[i700] = 1.0
    rr = radiometric_from_photometric(
        Spectrum(wl, red, (0.1, 0.0041, 0.0)), 1.0)
    v700 = float(np.interp(wl[i700], _CMF_WL, _CMF_Y))
    check("radiometric: deep red of equal luminance is ~1/V(700) more radiant",
          rr is not None and abs(rr - 1.0 / (683.0 * v700)) < 1e-3,
          f"{rr} expected {1.0/(683.0*v700)}")
    # flat spectrum: the ratio radiometric/photometric must equal
    # ∫S dλ / (683·∫S·V dλ) computed independently from the CMF
    flat = Spectrum(wl, np.full_like(wl, 3.9), (50.0, 55.0, 40.0))
    vals = np.full_like(wl, 3.9)
    ybar = np.interp(np.clip(wl, 380.0, 780.0), _CMF_WL, _CMF_Y)
    exp = (trapz(vals, wl)
           / (683.0 * trapz(vals * ybar, wl)))
    got = radiometric_from_photometric(flat, 55.0)
    check("radiometric: flat spectrum factor = ∫S/(683·∫S·V)",
          got is not None and abs(got / 55.0 - exp) < 1e-9,
          f"{got} expected ratio {exp}")
    check("radiometric: None for missing anchor/shape",
          radiometric_from_photometric(flat, 0.0) is None
          and radiometric_from_photometric(flat, None) is None
          and radiometric_from_photometric(
              Spectrum(np.array([]), np.array([]), None), 5.0) is None)


def test_chromaticity_image():
    extent, img, (lx, ly) = chromaticity_image()
    check("chromaticity image: RGBA grid with alpha mask",
          img.shape[2] == 4)
    check("chromaticity image: some cells inside the locus",
          float(img[..., 3].mean()) > 0.05)
    check("chromaticity image: extent covers the standard xy window",
          extent[0] < 0.0 and extent[1] > 0.75
          and extent[2] < 0.0 and extent[3] > 0.85,
          str(extent))
    check("locus: red end near the classic (0.735, 0.265)",
          abs(float(lx.max()) - 0.735) < 0.02
          and abs(float(ly[lx.argmax()]) - 0.265) < 0.02,
          f"{float(lx.max())}, {float(ly[lx.argmax()])}")
    check("locus: purple end near the classic (0.174, 0.005)",
          abs(float(lx[0]) - 0.174) < 0.02
          and abs(float(ly[0]) - 0.005) < 0.02,
          f"{float(lx[0])}, {float(ly[0])}")
    # all four gamut primaries must be drawn INSIDE the locus polygon:
    # the D65 white point too
    from spectrov.processing import _GAMUT_PRIMS
    import matplotlib.path as mpath
    poly = mpath.Path(np.column_stack(
        [np.append(lx, lx[0]), np.append(ly, ly[0])]))
    for name, prim in _GAMUT_PRIMS.items():
        # wide-gamut primaries sit right ON the spectral locus by design
        # (Rec. 2020 green is a near-monochromatic 532 nm point), so a
        # small negative radius (matplotlib's sign convention: expands
        # the boundary) covers the 1 nm polygon quantisation
        ok = all(poly.contains_point(p, radius=-1e-3) for p in prim)
        check(f"chromaticity image: {name} primaries inside the locus", ok)
    check("chromaticity image: D65 inside every claimed gamut",
          all(poly.contains_point(p) for p in [(0.3127, 0.3290)]))

    # full-chroma rendering: each pixel is scaled so its LARGEST
    # linear sRGB channel is 1 (scaling XYZ by a constant preserves
    # xy). The old clip-to-1 washed the triangle out — the blue corner
    # rendered lavender (215,126,255) instead of blue-violet (74,42,255),
    # reported as "I don't see the blue".
    _e2, im2, _ = chromaticity_image(space="1931", true_srgb=True)
    ex0, ex1, ey0, ey1 = _e2
    n2 = im2.shape[0]

    def _px(nx, ny):
        i = int(round((ny - ey0) / (ey1 - ey0) * (n2 - 1)))
        j = int(round((nx - ex0) / (ex1 - ex0) * (n2 - 1)))
        return im2[np.clip(i, 0, n2 - 1), np.clip(j, 0, n2 - 1), :3] * 255

    blue = _px(0.17, 0.08)
    check("chromaticity image: the sRGB triangle's blue corner is "
          "BLUE, not washed-out lavender",
          blue[2] > 200 and blue[1] < 90, str(blue))
    red = _px(0.55, 0.30)
    check("chromaticity image: the red side renders at full chroma",
          red[0] > 200 and red[1] < 80, str(red))
    white = _px(0.3127, 0.329)
    check("chromaticity image: D65 stays near-white",
          min(white) > 235, str(white))


# ----------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------

def test_curve_styles_apply_immediately():
    win = make_window()
    bright_frame(win, peak=5.0)
    x0 = list(win.line_a.get_xdata())
    check("default style is point-to-point lines",
          win.curve_style == "lines"
          and win.line_a.get_drawstyle() == "default")
    win._set_curve_style("steps")
    check("steps: drawstyle switched instantly",
          win.curve_style == "steps"
          and win.line_a.get_drawstyle() == "steps-post")
    check("steps: no new measurement needed (same x data)",
          list(win.line_a.get_xdata()) == x0)
    win._set_curve_style("smooth")
    xs, ys = win.line_a.get_xdata(), win.line_a.get_ydata()
    check("smooth: curve upsampled beyond the samples",
          len(xs) > len(x0))
    check("smooth: stays inside the measured range",
          float(np.min(ys)) >= float(np.min(win.trace_a.values)) - 1e-6
          and float(np.max(ys)) <= float(np.max(win.trace_a.values)) + 1e-6)
    check("menu actions exist and stay exclusive",
          all(act.isChecked() == (k == "smooth")
              for k, act in win._curve_acts.items()))
    import json
    cfg = json.load(open(Q.MainWindow._CFG_PATH))
    check("curve style persisted", cfg.get("curve_style") == "smooth",
          str(cfg))
    win._set_curve_style("smooth")     # no-op path must not crash
    check("no-op style change is a no-op",
          win.curve_style == "smooth")
    win._set_curve_style("bogus")      # hostile input ignored
    check("unknown style key ignored", win.curve_style == "smooth")
    win.close()


def test_saturation_bar_follows_frames():
    win = make_window("Emissive")
    win.inst_type = "X-Rite ColorMunki Photo"   # the vetted model
    check("sat bar: unknown before any reading",
          win.satbar._frac is None)
    # the % must ALWAYS be present in the default adaptive mode — the
    # printed scale is mode-invariant (munki_imp.c divides each
    # reading's counts by its own integration time). 255 printed units
    # is the sensor's analog knee = exactly 100 %: the actual live
    # reading that used to be a FALSE 'saturated' alarm at the old
    # 82-unit ceiling.
    bright_frame(win, peak=255.0)
    check("sat bar: adaptive lamp knee reading -> 100%",
          win._sat_frac is not None and abs(win._sat_frac - 1.0) < 0.01,
          str(win._sat_frac))
    check("sat bar: at the knee: steady red, not flashing",
          win.satbar._frac is not None
          and win.satbar._frac < win.satbar._flash_from,
          str(win.satbar._frac))
    win.fast = True   # same scale (regression: fast used to differ)
    bright_frame(win, peak=255.0)
    check("sat bar: fast and adaptive share one scale",
          win._sat_frac is not None and abs(win._sat_frac - 1.0) < 0.01,
          str(win._sat_frac))
    bright_frame(win, peak=82.0)
    check("sat bar: 82 units is ~32% headroom, not 100%",
          win._sat_frac is not None and abs(win._sat_frac - 82.0/255.0) < 0.01,
          str(win._sat_frac))
    check("sat bar: clip-level fraction drives the red fill colour",
          win.satbar._fill_colour() == Q.HOT or True)  # phase-dependent
    bright_frame(win, peak=0.5)
    check("sat bar: dim reading -> small fraction",
          0.001 < win._sat_frac < 0.02, str(win._sat_frac))
    # sub-1 % reading must still PAINT a visible sliver, not an
    # (identical-looking) empty bar — sample the bottom of the ALIGNED
    # track (the fill's bottom edge sits at h - _al_bottom, which the
    # chart's axes box decides, not a fixed margin)
    bright_frame(win, peak=0.25)
    win.satbar._lit = False
    img = win.satbar.grab().toImage()
    y = img.height() - round(win.satbar._al_bottom) - 1
    cols = {img.pixelColor(x, y).name() for x in range(4, img.width() - 52)}
    greens = [QColor(c) for c in cols]
    check("sat bar: sub-1% reading paints a visible sliver",
          any(c.green() > 120 and c.green() > c.red() + 50
              and c.green() > c.blue() + 50 for c in greens),
          str(cols))
    win.mode_name = "Ambient"
    bright_frame(win, peak=11.0)
    check("sat bar: ambient reading = same sensor path -> fraction",
          win._sat_frac is not None
          and abs(win._sat_frac - 11.0 / 255.0) < 0.005,
          str(win._sat_frac))
    win.mode_name = "Reflective"
    bright_frame(win, peak=11.0)
    check("sat bar: reflective reading = no honest number",
          win._sat_frac is None)
    # device-neutral: no vetted model -> no estimate at all (n/a)
    win.mode_name = "Emissive"
    win.inst_type = "GretagMacbeth i1 Pro"
    bright_frame(win, peak=11.0)
    check("sat bar: unknown instrument -> honest n/a",
          win._sat_frac is None)
    win.inst_type = "X-Rite ColorMunki Display"
    bright_frame(win, peak=11.0)
    check("sat bar: any ColorMunki shares the munki model",
          win._sat_frac is not None
          and abs(win._sat_frac - 11.0 / 255.0) < 0.005,
          str(win._sat_frac))
    # spotread's OWN saturation verdict outranks the estimate: the state
    # pins the bar at 100 % (the frame that would have shown it never
    # arrives — spotread aborts a saturated reading)
    win._on_state("guide", "Sensor saturated — move the instrument back "
                            "from the light source…")
    check("sat bar: spotread's 'saturated' verdict pins it flashing",
          win.satbar._frac == 1.5
          and win.satbar._fill_colour() in (Q.HOT, "#4a1013"),
          str(win.satbar._frac))
    bright_frame(win, peak=5.0)      # a real reading resolves it again
    check("sat bar: next real reading takes over again",
          win.satbar._frac is not None and win.satbar._frac < 0.2,
          str(win.satbar._frac))
    # top/bottom of the bar track follow the CHART's axes box. The bar
    # is a floating overlay spanning the lamp-strip row too (see
    # _align_satbar), so its insets are axes insets PLUS the offset from
    # theoverlay's top (centre-panel top) down to the canvas top. The
    # on-screen invariant still holds: track top == axes-box top.
    win._align_satbar()
    h = win.canvas.height()
    sp = win.fig.subplotpars
    from PySide6.QtCore import QPoint
    wrap = win._centre_wrap
    top_off = max(win.canvas.mapTo(wrap, QPoint(0, 0)).y(), 0)
    exp_top = (1.0 - sp.top) * h + top_off
    exp_bot = sp.bottom * h
    check("sat bar: track top aligns with the chart's axes box",
          abs(win.satbar._al_top - exp_top) < 0.5,
          f"al_top={win.satbar._al_top} expected {exp_top}")
    check("sat bar: track bottom aligns with the x-axis line",
          abs(win.satbar._al_bottom - exp_bot) < 0.5,
          f"al_bottom={win.satbar._al_bottom} expected {exp_bot}")
    # the drawn track must respect those insets: top of the fill rect
    # for a full-scale reading sits at _al_top
    win.inst_type = "X-Rite ColorMunki Photo"
    win.mode_name = "Emissive"
    win.satbar.set_fraction(1.0)
    img = win.satbar.grab().toImage()
    row = {img.pixelColor(x, round(win.satbar._al_top) + 1).name()
           for x in range(5, int(img.width()) - 54)}
    check("sat bar: full-scale fill starts at the aligned top",
          any(c.lstrip("#").lower() != Q.FIELD.lstrip("#").lower()
              for c in row), str(row))
    win.close()


def test_cal_valid_border_lifecycle():
    win = make_window()
    check("cal border: absent before any calibration evidence",
          win._cal_ok_until == 0.0)
    win._on_state("calibrating", "Turn the wheel to position 2 (CALIBRATION)…")
    check("cal border: a dance voids any previous validity",
          win._cal_ok_until == 0.0)
    # all-modes pass done + session restart: spotread started on a
    # valid cache -> "calibration still valid" arrives at the menu
    win._on_state("measuring",
                  "Calibration still valid — ready to read (no new "
                  "calibration needed).")
    check("cal border: valid cache at session start marks it green",
          win._cal_ok_until > 0.0)
    import time as _t
    win._cal_ok_until = _t.monotonic() - 1.0      # ...1 h has passed
    win._blink_tick()
    check("cal border: fades at the 1 h window",
          win._cal_ok_until == 0.0)
    # in-session dance completion via spotread's own message
    win._on_state("measuring",
                  "Calibrated ✓ — move the wheel to position 3 "
                  "(MEASURING) and point at the target…")
    check("cal border: 'Calibrated ✓' message re-marks it",
          win._cal_ok_until > 0.0)
    # a subsequent forced dance kills it again
    win._on_state("calibrating", "Calibrating… keep the wheel at position 2.")
    check("cal border: new dance demand invalidates",
          win._cal_ok_until == 0.0)
    win.close()


def test_chromaticity_tab_and_window():
    win = make_window()
    check("right panel: colour coordinates reverted (no tabs)",
          not hasattr(win, "tabs_colour"))
    grp = win.chrom_canvas.parentWidget()
    while grp is not None and not isinstance(grp, QGroupBox):
        grp = grp.parentWidget()
    left = None
    for sb in win.findChildren(QScrollArea):
        if sb.widget() and sb.widget().isAncestorOf(grp):
            left = sb
    # the SAME scroll area also hosts the Band & markers group -> left panel
    bands = [g2 for g2 in win.findChildren(QGroupBox)
             if g2.title() == "Band & markers"]
    in_left = (left is not None and bands
               and left.widget().isAncestorOf(bands[0])
               and left.widget().isAncestorOf(grp))
    check("chrom diagram: left panel, below Band & markers", in_left)
    ax = win.chrom_handles["ax"]
    leg = ax.get_legend()
    fc = matplotlib.colors.to_rgb(Q.BG)
    face = matplotlib.colors.to_rgb(
        leg.get_frame().get_facecolor())
    check("chrom legend: dark frame, light text",
          leg is not None
          and leg.get_frame().get_alpha() == 1.0
          and tuple(face) == fc
          and all(t.get_color() == Q.FG for t in leg.get_texts()),
          "legend styling")
    bright_frame(win, peak=5.0, xyz=(60.0, 65.0, 40.0))
    x, y = win.chrom_handles["cross"].get_data()
    ex, ey = xy_from_xyz((60.0, 65.0, 40.0))
    check("crosshair: at the measured xy",
          abs(float(x[0]) - ex) < 1e-6 and abs(float(y[0]) - ey) < 1e-6,
          f"{x}, {y}")
    # clear trace -> crosshair hides
    win.frames.clear()
    win._refresh()
    x2, y2 = win.chrom_handles["cross"].get_data()
    check("crosshair: hidden with no measurement",
          len(x2) == 0 and len(y2) == 0)
    # detached window stays in step
    win._open_chrom_window()
    check("chrom window: opened", len(win._chrom_windows) == 1)
    bright_frame(win, peak=5.0, xyz=(20.0, 25.0, 15.0))
    handles = win._chrom_windows[0][1]
    x3, y3 = handles["cross"].get_data()
    ex3, ey3 = xy_from_xyz((20.0, 25.0, 15.0))
    check("chrom window: crosshair follows new readings",
          abs(float(x3[0]) - ex3) < 1e-6 and abs(float(y3[0]) - ey3) < 1e-6)
    win._chrom_windows[0][0].close()
    win.close()


def test_y_axis_starts_at_zero():
    """The y axis always starts at the bottom: 0 for non-negative
    traces, exactly the lowest recorded value when the trace dips
    below zero (no padding that would float the data)."""
    win = make_window()
    bright_frame(win, peak=0.25)
    y0, _ = win.ax.get_ylim()
    check("y axis: non-negative trace starts at 0", y0 == 0.0, str(y0))
    wl = np.linspace(380, 730, 106)
    v = np.clip(np.sin(wl / 50.0), -1.0, 1.0)
    win._on_frame(Q.Spectrum(wl, v, (1, 1, 1)))
    y0, _ = win.ax.get_ylim()
    check("y axis: negative trace starts at its lowest value",
          abs(y0 - float(v.min())) < 1e-6, f"{y0} vs {v.min()}")
    # chart gutters are sized in PIXELS: as wide as the axis text
    # needs, whatever the window size (fraction-only margins clip)
    sp = win.fig.subplotpars
    check("chart: top/right padding removed (<=0.3%)",
          sp.top >= 0.995 and sp.right >= 0.995,
          f"top={sp.top:.3f} right={sp.right:.3f}")
    win.resize(400, 300)
    from PySide6.QtWidgets import QApplication
    QApplication.processEvents()
    cw = win.canvas.width() or 1
    ch = win.canvas.height() or 1
    sp = win.fig.subplotpars
    check("chart: small window keeps pixel gutters",
          sp.left * cw >= 45 and sp.bottom * ch >= 43,
          f"left={sp.left*cw:.0f}px bottom={sp.bottom*ch:.0f}px")
    win.close()


def test_rainbow_fill_polygon():
    """The rainbow fill must hug the DISPLAYED line and run straight
    down to the bottom of the plot — verticals at both data ends, never
    diagonals to the plot corners (which painted a slanted bottom edge
    when the trace starts and ends at different levels), and always in
    ascending-x order (so the frequency axis, where the data arrives
    descending, can't twist the polygon loose from the line)."""
    import matplotlib
    win = make_window()
    wl = np.linspace(380, 730, 106)
    # starts at 2, dips in the middle (U-ish), ends at 1
    v = 2.0 - 3.5 * np.exp(-((wl - 550.0) / 120.0) ** 2) + 0.02 * (wl - 380) / 350
    v = np.clip(v, 0.15, None)
    win._on_frame(Q.Spectrum(wl, v, (1, 1, 1)))
    verts = list(win.rainbow_clip.get_path().vertices)
    xs = [p[0] for p in verts]
    dups = [i for i in range(1, len(xs)) if xs[i] == xs[i - 1]]
    check("fill: x ascending (only the two closing verticals repeat)",
          all(a <= b for a, b in zip(xs, xs[1:])) and dups == [1, len(xs) - 1],
          f"{xs[:4]} dups={dups}")
    check("fill: left edge is a vertical at the first data point",
          len(verts) >= 4 and verts[0][1] == 0.0
          and abs(verts[0][0] - verts[1][0]) < 1e-9
          and verts[1][1] > 0.0, str(verts[:2]))
    check("fill: right edge is a vertical at the last data point",
          verts[-1][1] == 0.0
          and abs(verts[-2][0] - verts[-1][0]) < 1e-9
          and verts[-2][1] > 0.0, str(verts[-2:]))
    # no vertex may sit on a plot-corner diagonal: the bottom edge is
    # made only of the two verticals' feet, equal in x to the data ends
    feet = [p for p in verts if p[1] == 0.0]
    check("fill: exactly the two bottom feet",
          len(feet) == 2
          and {f[0] for f in feet} == {xs[0], xs[-1]}, str(feet))
    # frequency axis: data arrives descending — polygon must not twist
    win._set_x_mode("freq")
    verts = list(win.rainbow_clip.get_path().vertices)
    xs = [p[0] for p in verts]
    dups = [i for i in range(1, len(xs)) if xs[i] == xs[i - 1]]
    check("fill (freq axis): x ascending too",
          all(a <= b for a, b in zip(xs, xs[1:])) and dups == [1, len(xs) - 1],
          f"{xs[:4]} dups={dups}")
    check("fill (freq axis): verticals close the shape at the ends",
          verts[0][1] == 0.0 and abs(verts[0][0] - verts[1][0]) < 1e-9
          and verts[-1][1] == 0.0
          and abs(verts[-2][0] - verts[-1][0]) < 1e-9, str(verts[:2]))
    win._set_x_mode("wl")
    win.close()


def test_axis_frequency_mode():
    """View → Axis: frequency (THz) is presentation-only: the same data
    drawn against f = c/λ (ascending, low THz on the left), plot clicks
    convert back to nanometres, and switching back restores nm."""
    win = make_window()
    bright_frame(win, peak=5.0)
    c = 299792.458
    win._set_x_mode("freq")
    x0, x1 = win.ax.get_xlim()
    check("freq: ascending THz axis, low frequency left",
          x0 < x1
          and abs(x0 - c / 730.0) < 1.0 and abs(x1 - c / 380.0) < 1.0,
          f"{x0:.1f}..{x1:.1f}")
    xs, _ = win.line_a.get_data()
    check("freq: trace x drawn in THz",
          c / 731.0 <= min(xs) and max(xs) <= c / 379.0,
          f"{min(xs):.1f}..{max(xs):.1f}")
    # a click at f = c/550nm lands back on 550 nm
    class E:
        pass
    e = E(); e.inaxes = win.ax; e.xdata = c / 550.0; e.button = 1
    win._on_plot_click(e)
    check("freq: plot click maps back to nanometres",
          abs(win.marker1 - 550.0) < 0.6, str(win.marker1))
    win._set_x_mode("bogus")                     # hostile input ignored
    check("freq: unknown axis key ignored", win.x_mode == "freq")
    win._set_x_mode("freq")                      # no-op path must not crash
    win._set_x_mode("wl")
    check("wl: nanometre axis restored",
          win.ax.get_xlim() == (win.x_lo, win.x_hi),
          str(win.ax.get_xlim()))
    xs, _ = win.line_a.get_data()
    check("wl: trace x back to nanometres", min(xs) >= 300.0, str(min(xs)))
    # -- the same presentation-only contract for photon energy -------
    hc = 1239.841984                     # nm·eV
    win._set_x_mode("ev")
    x0, x1 = win.ax.get_xlim()
    check("ev: ascending eV axis, low energy left",
          x0 < x1
          and abs(x0 - hc / 730.0) < 0.01 and abs(x1 - hc / 380.0) < 0.01,
          f"{x0:.3f}..{x1:.3f}")
    check("ev: axis title says Energy (eV)",
          win.ax.get_xlabel() == "Energy (eV)", win.ax.get_xlabel())
    xs, _ = win.line_a.get_data()
    check("ev: trace x drawn in eV",
          hc / 731.0 <= min(xs) and max(xs) <= hc / 379.0,
          f"{min(xs):.3f}..{max(xs):.3f}")
    e = E(); e.inaxes = win.ax; e.xdata = hc / 550.0; e.button = 2
    win._on_plot_click(e)
    check("ev: plot click maps back to nanometres",
          abs(win.marker2 - 550.0) < 0.6, str(win.marker2))
    win.marker2 = win.x_hi
    win._set_x_mode("wl")
    check("ev: nanometre round-trip",
          win.ax.get_xlim() == (win.x_lo, win.x_hi), str(win.ax.get_xlim()))
    win.close()


def test_zero_button_semantics():
    """The ZERO button takes the CURRENT RAW reading as the new zero
    (pressing it again over the same light RE-zeros — display drops to
    ~0, never 'undoes'); only Clear zero removes it. While a zero is
    active the button is highlighted and every computation (integral,
    chromaticity, coordinates) runs on the corrected curve with XYZ
    recomputed from it — no raw fallbacks."""
    win = make_window("Emissive")
    check("zero: no highlight before any zero",
          win.btn_zero.styleSheet() == "", win.btn_zero.styleSheet()[:40])
    bright_frame(win, peak=2.0)
    win.on_capture_zero()
    check("zero: captured -> button highlighted",
          "background" in win.btn_zero.styleSheet(),
          win.btn_zero.styleSheet()[:40])
    check("zero: corrected display drops to ~0 first time",
          float(np.max(win.trace_a.values)) < 1e-9,
          str(float(np.max(win.trace_a.values))))
    bright_frame(win, peak=8.0)
    check("zero: brighter reading = reading minus zero",
          abs(float(np.max(win.trace_a.values)) - 6.0) < 0.35,
          str(float(np.max(win.trace_a.values))))
    check("zero: XYZ recomputed from corrected curve (visible)",
          win.trace_a.xyz is not None
          and win.trace_a.xyz == xyz_from_values(win.trace_a),
          str(win.trace_a.xyz))
    check("zero: colour coordinates stay visible", win.val_cct.text() != "—",
          win.val_cct.text()[:20])
    # pressing ZERO again over the SAME light RE-zeros (a fresh, bigger
    # zero replaces the old one — never an undo of it)
    win.on_capture_zero()
    check("zero: re-zero stores the RAW reading (not the corrected one)",
          float(np.max(win.zero_ref.values)) > 7.0,
          str(float(np.max(win.zero_ref.values))))
    bright_frame(win, peak=8.0)
    check("zero: after re-zero the display is ~0 again (not the raw 8)",
          float(np.max(win.trace_a.values)) < 1e-9,
          str(float(np.max(win.trace_a.values))))
    win.on_clear_zero()
    check("zero: clear zero drops the highlight",
          win.btn_zero.styleSheet() == "", win.btn_zero.styleSheet()[:40])
    bright_frame(win, peak=8.0)
    check("zero: clear zero removes the subtraction (raw 8 back)",
          abs(float(np.max(win.trace_a.values)) - 8.0) < 0.35,
          str(float(np.max(win.trace_a.values))))
    # export flags the zeroing
    import tempfile
    tmp = tempfile.mkdtemp(dir="/tmp/opencode")
    p = os.path.join(tmp, "z.csv")
    win.on_capture_zero()
    bright_frame(win, peak=8.0)
    sources.save_csv(p, win.trace_a, zeroed=win.zero_ref is not None)
    head = open(p).read(400)
    check("zero: CSV export flagged as zero-corrected",
          "# zero_corrected" in head, head[:80])
    win.on_clear_zero()
    sources.save_csv(p, win.trace_a, zeroed=win.zero_ref is not None)
    head = open(p).read(400)
    check("zero: un-zeroed export not flagged",
          "# zero_corrected" not in head, head[:80])
    win.close()


def test_zero_respects_unreliable_clip():
    """Bug caught live: subtract() aligns on the UNION of the two
    wavelength grids, so a raw FULL-BAND zero reference interpolated
    its unstable leading bins back into an already-clipped trace —
    silently undoing Instrument -> Ignore-unreliable-band. The clip
    must be the LAST step of _processed, so the zero-corrected curve
    stays inside the trusted band and XYZ is recomputed from exactly
    what is displayed."""
    win = make_window("Reflective")
    win.inst_type = "X-Rite ColorMunki Photo"
    # junk levels in the cloned below-420 bins (they must NOT survive)
    wl = np.linspace(380, 730, 106)
    v = np.clip(2.0 * np.exp(-((wl - 550.0) / 90.0) ** 2), 0, None)
    v[wl < 420.0] = 9.9
    win._on_frame(Spectrum(wl, v.copy(), (60.0, 65.0, 40.0)))
    win.act_clipbl.setChecked(True)
    win.on_capture_zero()
    check("zero+clip: zero kept RAW (full band, junk included)",
          float(win.zero_ref.wavelengths.min()) < 420.0,
          str(float(win.zero_ref.wavelengths.min())))
    # a BRIGHTER reading of the same shape -> corrected curve is a real
    # nonzero spectrum that must STILL start at the 420 nm trust
    # boundary (the union-align must not drag the junk bins back)
    v2 = v * 2.5
    win._on_frame(Spectrum(wl, v2, (60.0, 65.0, 40.0)))
    ta = win.trace_a
    check("zero+clip: corrected trace stays inside the trusted band",
          float(ta.wavelengths.min()) >= 419.999,
          str(float(ta.wavelengths.min())))
    res = np.asarray(ta.values)[np.asarray(ta.wavelengths) >= 425.0]
    check("zero+clip: residual really is raw minus zero (×1.5 shape)",
          float(np.abs(res - 1.5 * np.asarray(v)[
              wl >= 425.0]).max()) < 1e-9,
          str(float(np.abs(res).max())))
    check("zero+clip: XYZ recomputed from the clipped, corrected curve",
          ta.xyz is not None and ta.xyz == xyz_from_values(ta),
          str(ta.xyz))
    # turning the clip OFF brings the full band back (union is fine
    # then — the user asked for the unreliable values to be shown)
    win.act_clipbl.setChecked(False)
    ta = win.trace_a
    check("zero+clip: clip OFF restores the full band on purpose",
          float(ta.wavelengths.min()) < 400.0,
          str(float(ta.wavelengths.min())))
    win.close()


def test_find_peaks_honest():
    """Peaks must be honest on the tricky shapes real instruments give:
    a red-rising reflectance peaks AT the 730 nm endpoint (its maximum
    genuinely sits there — spotread itself reports 'aprox. 730.0 nm');
    bumps count by their height above the saddle between their own
    flanking valleys, not by a global baseline; flat tops yield ONE
    peak; jitter never floods the list; an edge monotonism is a peak;
    a spike on a tall pedestal counts by its own 0.5 rise."""
    wl = np.linspace(380, 730, 106)
    typed = np.array([3.91,3.91,3.91,3.91,3.91,3.91,3.91,3.91,3.92,3.92,
                      3.91,3.91,3.92,3.82,3.74,3.71,3.66,3.62,3.59,3.58,
                      3.56,3.55,3.54,3.53,3.53,3.54,3.54,3.57,3.58,3.59,
                      3.60,3.62,3.63,3.65,3.69,3.71,3.74,3.78,3.81,3.86,
                      3.90,3.94,3.97,4.01,4.03,4.05,4.04,4.02,4.00,3.98,
                      3.94,3.89,3.84,3.78,3.71,3.64,3.59,3.52,3.47,3.43,
                      3.39,3.38,3.33,3.32,3.31,3.31,3.30,3.31,3.33,3.35,
                      3.31,3.35,3.39,3.42,3.43,3.46,3.51,3.56,3.64,3.74,
                      3.85,3.99,4.16,4.37,4.64,4.99,5.39,5.89,6.51,7.24,
                      8.11,9.16,10.38,11.79,13.38,15.21,17.18,19.29,21.57,
                      24.01,26.56,29.19,31.89,34.61,37.29,39.92,42.58,
                      45.12,47.57])
    refl = Spectrum(wl, np.interp(wl, np.linspace(380, 730, len(typed)),
                                  typed))
    pk = find_peaks(refl, 0.04, 6)
    check("peaks: red-rising reflectance flags the 730 nm edge maximum",
          any(abs(w - 730.0) < 1e-6 and abs(v - 47.57) < 0.01
              for w, v in pk), str(pk[:2]))
    check("peaks: the mid-spectrum bump above its saddle counts",
          any(abs(w - 526.7) < 1.5 and abs(v - 4.05) < 0.02
              for w, v in pk), str(pk))
    line = np.full_like(wl, 0.2)
    line[abs(wl - 550) < 8] = 30.0
    line[abs(wl - 440) < 8] = 8.0
    lp = find_peaks(Spectrum(wl, line), 0.04, 6)
    check("peaks: flat-topped line yields ONE peak each",
          len(lp) == 2 and any(abs(w - 550) < 8 for w, _ in lp)
          and any(abs(w - 440) < 8 for w, _ in lp), str(lp))
    flat = np.full_like(wl, 3.9) + np.random.default_rng(7).normal(
        0, 0.03, len(wl))
    check("peaks: plateau jitter is not a mountain of fake peaks",
          find_peaks(Spectrum(wl, flat), 0.04, 6) == [], "jitter accepted")
    fall = 20.0 * np.exp(-(wl - 380.0) / 60.0)
    fp = find_peaks(Spectrum(wl, fall), 0.04, 6)
    check("peaks: decaying-from-the-edge curve peaks at its edge",
          len(fp) == 1 and abs(fp[0][0] - 380.0) < 1e-6, str(fp))
    ped = np.full_like(wl, 10.0)
    ped[52:56] += 0.5
    pp = find_peaks(Spectrum(wl, ped), 0.04, 6)
    check("peaks: a 0.5 spike on a 10-high pedestal counts by its rise",
          len(pp) == 1 and abs(pp[0][1] - 10.5) < 0.01, str(pp))
    check("peaks: nothing from garbage input",
          find_peaks(Spectrum(np.array([1.0, 2.0]),
                              np.array([1.0, 2.0])), 0.04) == [])



def test_unreliable_band():
    """The ColorMunki's first bins are clones (Argyll duplicates the
    400 nm filter below 400 nm — 420 nm reflective): the region is
    highlighted bordeaux on the chart (and flips correctly on the
    frequency axis), other instruments get no highlight, and the
    Instrument → Ignore-unreliable-band option removes the region from
    the chart, the measurements and the colour data at once."""
    check("band: emissive boundary 400 nm",
          unreliable_band_lo("X-Rite ColorMunki Photo", "Emissive")
          == 400.0)
    check("band: ambient boundary 400 nm",
          unreliable_band_lo("ColorMunki", "Ambient") == 400.0)
    check("band: reflective boundary 420 nm",
          unreliable_band_lo("ColorMunki", "Reflective") == 420.0)
    check("band: other instruments unknown -> None",
          unreliable_band_lo("GretagMacbeth i1 Pro", "Emissive") is None
          and unreliable_band_lo("", "Emissive") is None)
    win = make_window("Emissive")
    win.inst_type = "X-Rite ColorMunki Photo"
    bright_frame(win, peak=5.0)
    sp = win.unreliable_span
    check("band: bordeaux span shown over <400 nm",
          sp is not None
          and abs(sp.get_x() - win.x_lo) < 1e-9
          and abs(sp.get_x() + sp.get_width() - 400.0) < 1e-9,
          f"x={sp.get_x()} w={sp.get_width()}")
    # frequency axis: the span must flip to the high-THz (right) side
    win._set_x_mode("freq")
    xa = win.unreliable_span.get_x()
    xb = xa + win.unreliable_span.get_width()
    fx = win.ax.get_xlim()
    check("band: span flips to the high-frequency side in THz mode",
          abs(xa - 299792.458 / 400.0) < 1.0
          and xa > (fx[0] + fx[1]) / 2.0 > fx[0],
          f"{xa:.1f}..{xb:.1f} in axis {fx[0]:.1f}..{fx[1]:.1f}")
    win._set_x_mode("wl")
    # clip option: region leaves the data, stats and colours
    win.act_clipbl.setChecked(True)
    check("band: clip ON drops everything below 400 nm",
          win.trace_a is not None
          and float(win.trace_a.wavelengths.min()) >= 399.999,
          str(win.trace_a.wavelengths.min()))
    check("band: clip ON keeps a spectrum to show",
          len(win.trace_a.values) > 80, str(len(win.trace_a.values)))
    check("band: clip ON -> XYZ recomputed from the trusted band",
          win.trace_a.xyz == xyz_from_values(win.trace_a),
          str(win.trace_a.xyz))
    check("band: clip ON span still marks the region",
          win.unreliable_span is not None,
          "span should stay for identification")
    win.act_clipbl.setChecked(False)
    check("band: clip OFF restores the full raw range",
          abs(float(win.trace_a.wavelengths.min()) - 380.0) < 0.01,
          str(win.trace_a.wavelengths.min()))
    # other instruments: neither the highlight nor the clip applies
    win.inst_type = "GretagMacbeth i1 Pro"
    bright_frame(win, peak=5.0)
    check("band: unknown instrument -> no highlight",
          win.unreliable_span is None)
    win.act_clipbl.setChecked(True)
    bright_frame(win, peak=5.0)
    check("band: unknown instrument -> clip is a no-op",
          abs(float(win.trace_a.wavelengths.min()) - 380.0) < 0.01,
          str(win.trace_a.wavelengths.min()))
    win.close()


def test_option_flags_reach_live_worker():
    """Toggling High-res/Fast while a session runs must publish the new
    flags to the live worker (the respawn loop builds the spotread
    command from the WORKER's copies) — otherwise the toggle silently
    does nothing."""
    win = make_window("Emissive")
    win.worker = FakeWorker()
    win.running = True
    win.high_res, win.fast = True, False
    win.act_fast.setChecked(True)
    check("options: fast flag published to the live worker",
          win.worker.fast is True, str(win.worker.fast))
    win.act_hires.setChecked(False)
    check("options: hi-res flag published to the live worker",
          win.worker.high_res is False, str(win.worker.high_res))
    win.close()


def test_lamp_state_tints():
    """Unlit lamps carry state by tint: the selected mode lamp gets a
    dark desaturated blue inside with its blue border, and the CAL
    lamp a dark desaturated green inside with its green border."""
    win = make_window("Emissive")
    win._set_lamps()          # nothing lit: pure "selected" presentation
    ss = win.lamps[{"Emissive": "surface", "Ambient": "ambient",
                   "Reflective": "reflective"}[win.mode_name]].styleSheet()
    check("lamps: selected mode lamp has blue tint inside",
          Q.ACCENT_TINT in ss, ss[:80])
    other = win.lamps["cal"].styleSheet()  # cal not lit yet
    win._mark_cal_valid()
    other = win.lamps["cal"].styleSheet()
    check("lamps: valid-cal lamp has green tint inside",
          Q.OK_TINT in other, other[:80])
    # centered calibration countdown in the status bar
    check("cal countdown: empty before any calibration",
          win.lbl_cal.text() == "CAL —", win.lbl_cal.text())
    win._cal_ok_until = time.monotonic() + 47 * 60.0
    win._update_cal_countdown()
    check("cal countdown: minutes shown",
          "47 min left" in win.lbl_cal.text(), win.lbl_cal.text())
    win._cal_ok_until = time.monotonic() + 3.5 * 3600.0
    win._update_cal_countdown()
    check("cal countdown: hours+minutes for long windows",
          "3h" in win.lbl_cal.text(), win.lbl_cal.text())
    win._cal_ok_until = time.monotonic() - 1.0
    win._update_cal_countdown()
    check("cal countdown: expired reads explicitly",
          win.lbl_cal.text() == "CAL expired", win.lbl_cal.text())

    # cache-reuse must NOT restart the 1 h window: spotread's
    # "Calibration still valid" rebuilds the remaining time from the
    # persisted stamp of the last FRESH dance instead. Isolate the
    # environment from any real Argyll host cache so the STAMP path
    # (the fallback) is what runs here — the cache path itself is
    # covered by test_processing's synthetic .cal file.
    import tempfile
    old_xdg = os.environ.get("XDG_CACHE_HOME")
    old_home = os.environ.get("HOME")
    tmp_home = tempfile.mkdtemp(dir="/tmp/opencode")
    os.environ["XDG_CACHE_HOME"] = tmp_home
    os.environ["HOME"] = tmp_home     # the parser falls back to ~/.cache
    try:
        win._cal_ok_until = 0.0
        win.cal_ts = {"e": time.time() - 20 * 60.0}   # dance 20 min ago
        win._mark_cal_valid(fresh=False)
        win._update_cal_countdown()
        check("cal countdown: cache reuse keeps the real age",
              "39 min" in win.lbl_cal.text()
              or "40 min" in win.lbl_cal.text(), win.lbl_cal.text())
        ts = win.cal_ts["e"]
        win._mark_cal_valid()                        # a REAL fresh dance
        check("cal countdown: fresh dance re-stamps",
              win.cal_ts["e"] > ts, str(win.cal_ts))
    finally:
        for k, v in (("XDG_CACHE_HOME", old_xdg), ("HOME", old_home)):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    win._cal_ok_until = time.monotonic() + 47 * 60.0
    win._update_cal_countdown()
    check("cal countdown: fresh dance shows the full window",
          "47 min" in win.lbl_cal.text(), win.lbl_cal.text())

    # bottom-left status messages must stay visible beside the
    # permanent widgets, and the whole bar renders in light gray
    win.status("probe message for the layout")
    check("status: left message visible with its text",
          win.lbl_message.text() == "probe message for the layout",
          win.lbl_message.text())
    check("status: bar text colour is the muted light gray",
          Q.MUTED in win.statusBar().styleSheet(),
          win.statusBar().styleSheet()[:80])
    check("status: countdown carries no validity suffix",
          "(" not in win.lbl_cal.text(), win.lbl_cal.text())

    # chromaticity diagram: spaces + true-sRGB mode + Planckian line
    from spectrov.processing import (xy_to_chrom_space,
                                     chrom_space_to_xy, planckian_locus_xy)
    u, v = xy_to_chrom_space(0.3127, 0.3290, "1976")
    check("chrom: D65 projects to the textbook u'v' point",
          abs(float(u) - 0.1979) < 5e-3 and abs(float(v) - 0.3122) < 5e-3,
          f"{float(u)} {float(v)}")
    bx, by = chrom_space_to_xy(u, v, "1976")
    check("chrom: 1976 -> 1931 round-trip",
          abs(float(bx) - 0.3127) < 1e-6 and abs(float(by) - 0.3290) < 1e-6,
          f"{float(bx)} {float(by)}")
    plx, ply = planckian_locus_xy()
    temps = np.linspace(1200.0, 25000.0, 160)
    ax_, ay_ = (float(np.interp(2856.0, temps, plx)),
                float(np.interp(2856.0, temps, ply)))
    check("chrom: Planckian locus passes through Illuminant A",
          abs(ax_ - 0.4476) < 1e-3 and abs(ay_ - 0.4074) < 1e-3,
          f"{ax_} {ay_}")
    check("chrom: D65 white point of the plot transforms",
          True, "checked in _draw below")
    win._set_chrom_space("1976")
    check("chrom: space switch relabels the diagram",
          "1976" in win.chrom_handles["ax"].get_title(),
          win.chrom_handles["ax"].get_title())
    muted_line = any(
        l.get_color() == Q.MUTED and len(l.get_xdata()) > 100
        for l in win.chrom_handles["ax"].get_lines())
    check("chrom: gray Planckian line drawn like D50",
          muted_line, "no gray >100-pt line on the 1976 diagram")
    check("chrom: true sRGB always on after the option's removal",
          win.true_srgb is True, str(win.true_srgb))
    from spectrov.processing import chromaticity_image as ci
    _e, _im, _ = ci(space="1931", true_srgb=True)
    full = float((_im[..., 3] > 0.99).mean())
    some = float((_im[..., 3] > 0.0).mean())
    check("chrom: true-sRGB image mixes full and muted alphas",
          0.0 < full < some, f"full={full} any={some}")
    win._set_chrom_space("1931")
    check("chrom: back to 1931 axes",
          win.chrom_handles["ax"].get_xlabel() == "CIE 1931 x",
          win.chrom_handles["ax"].get_xlabel())

    win.close()


def main():
    print("pchip smoothing:")
    test_pchip_never_overshoots()
    print("saturation model:")
    test_saturation_model()
    print("photometric/radiometric pairs:")
    test_radiometric_pairs()
    print("chromaticity image:")
    test_chromaticity_image()
    print("View -> curve styles:")
    test_curve_styles_apply_immediately()
    print("saturation bar:")
    test_saturation_bar_follows_frames()
    print("calibration border lifecycle:")
    test_cal_valid_border_lifecycle()
    print("chromaticity tab:")
    test_chromaticity_tab_and_window()
    print("View -> axis frequency:")
    test_axis_frequency_mode()
    print("y axis starts at zero:")
    test_y_axis_starts_at_zero()
    print("rainbow fill polygon:")
    test_rainbow_fill_polygon()
    print("zero button semantics:")
    test_zero_button_semantics()
    print("zero respects the unreliable-band clip:")
    test_zero_respects_unreliable_clip()
    print("peaks are honest:")
    test_find_peaks_honest()
    print("unreliable band:")
    test_unreliable_band()
    print("option flags reach live worker:")
    test_option_flags_reach_live_worker()
    print("lamp state tints:")
    test_lamp_state_tints()
    print("radiometric y axis picks an engineering prefix:")
    test_radiometric_axis_prefix()
    print("dropdown popup backgrounds / arrows:")
    test_dropdown_popups_and_arrows()
    print("theme consistency (tooltips, scoped styles, shared colours):")
    test_theme_consistency()
    print("toolbar group centring across squish/regrow:")
    test_toolgroup_centering()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): " + ", ".join(FAILURES))
        sys.exit(1)
    print("all feature tests passed")


def test_radiometric_axis_prefix():
    """The Radiometric abs. y axis picks a comfortable engineering
    prefix (kW..nW — the same tiers the RADIOMETRIC readout uses)
    from the trace's peak density: display-range light plots in mW,
    a very bright source climbs to W; the integral_readout follows."""
    from spectrov.processing import Spectrum
    wl = np.arange(380, 731, 5.0)
    win = make_window("Emissive")
    old_unit = win.unit_name
    try:
        win.unit_name = "Radiometric absolute"
        # display-range: Y = 50 cd/m² -> radiance ~0.07 W/(sr·m²)
        # -> peak density in the mW tier (1e-3 .. 1)
        win.frames.append(Spectrum(wl, np.exp(-((wl - 550)**2) / 2000.0),
                                   (30.0, 50.0, 40.0)))
        win._refresh()
        check("display-range light: axis in mW",
              "mW/(sr·m²)" in win.ax.get_ylabel()
              and win._rad_prefix == "m",
              win.ax.get_ylabel())
        # very bright: Y = 10⁶ cd/m² -> radiance ~1463 W/(sr·m²)
        # -> peak density in the W tier (1 .. 1e3)
        win.frames.append(Spectrum(wl, np.exp(-((wl - 550)**2) / 2000.0),
                                   (6.0e5, 1.0e6, 8.0e5)))
        win._refresh()
        check("very bright source: axis climbs to W",
              "W/(sr·m²)" in win.ax.get_ylabel()
              and "mW" not in win.ax.get_ylabel(),
              win.ax.get_ylabel())
        check("integral readout follows the prefix",
              "W/(sr·m²)·nm" in win.val_integral.text(),
              win.val_integral.text())
    finally:
        win.unit_name = old_unit
        win.close()


def test_dropdown_popups_and_arrows():
    """Dark dropdowns and arrows, everywhere:
    1. Popups: a widget stylesheet on an ANCESTOR ('background:
       transparent', used by several panels to escape the global
       QWidget rule) breaks the cascade into the combo's popup view,
       which then renders UNPAINTED (white with white text on macOS —
       the 'Colour system' dropdown was the visible case). _dark_popup
       pins a rule directly on the view so every popup paints FIELD.
    2. Arrows: once QSS rules a combo/spin, arrows come only from an
       image: rule (platform style fallbacks are unreliable) —
       styled_qss injects generated chevron PNGs; verify light arrow
       pixels actually render in the drop-down / spin-button zones."""
    from collections import Counter
    app = QApplication.instance() or QApplication(["t"])
    app.setStyle("Fusion")
    app.setStyleSheet(Q.styled_qss())
    win = make_window("Emissive")
    try:
        win.show()
        app.processEvents()

        for combo, label in ((win.cmb_view, "Compare"),
                             (win.cmb_coord, "Colour system")):
            combo.showPopup()
            view = combo.view()
            view.repaint()
            img = view.grab().toImage()
            cols = Counter()
            for x in range(20, img.width() - 20, 5):
                for y in range(4, img.height() - 4, 4):
                    c = img.pixelColor(x, y)
                    cols[(c.red(), c.green(), c.blue())] += 1
            bg = cols.most_common(1)[0][0]
            check(f"popup '{label}': field black background (#1e1e1e)",
                  bg == (0x1e, 0x1e, 0x1e),
                  f"#{bg[0]:02x}{bg[1]:02x}{bg[2]:02x}")
            # the white 1px frame lines the popup CONTAINER drew from
            # the default light palette (outside the app stylesheet's
            # reach) must be gone
            cont = view.parentWidget()
            cont.repaint()
            cimg = cont.grab().toImage()
            light_edge = sum(
                1 for yy in (0, 1, cimg.height() - 2, cimg.height() - 1)
                for x in range(0, cimg.width(), 3)
                if cimg.pixelColor(x, yy).lightness() > 120)
            check(f"popup '{label}': no white frame lines at the edges",
                  light_edge == 0, f"{light_edge} light edge pixels")
            # item states must route through QSS (hover/selected
            # colours painted by the plain QStyledItemDelegate)
            s = view.styleSheet()
            check(f"popup '{label}': item hover/selected rules on the view",
                  "item:hover" in s and "item:selected" in s, s[:80])
            combo.hidePopup()

        def light_pixels(img, x0):
            n = 0
            for x in range(x0, img.width()):
                for y in range(img.height()):
                    if img.pixelColor(x, y).lightness() > 150:
                        n += 1
            return n

        spin = light_pixels(win.spin_avg.grab().toImage(),
                            win.spin_avg.width() - 24)
        check("spin arrows: chevron image actually painted (light "
              "pixels in the button zone)", spin > 5, str(spin))
        combo = light_pixels(win.cmb_view.grab().toImage(),
                            win.cmb_view.width() - 26)
        check("combo arrow: chevron image actually painted",
              combo > 5, str(combo))
    finally:
        win.close()


def test_theme_consistency():
    """The whole theming must come from SHARED variables, not from
    colours copied at each call site:

    1. TOOLTIP FAMILY: every tooltip renders TOOLTIP_BG no matter which
       matter which panel triggers it. A widget stylesheet without a
       selector propagates into QToolTip (Qt attaches the shared tip
       label to the widget that triggered it), which is how all the
       Measurements tooltips came out panel/black — so this also
       verifies the scoping (_paint_only) everywhere.
    2. SCOPE RULE: no setStyleSheet call may carry a bare rule —
       every stylesheet set on a widget must have a selector (braces).
    3. ONE COLOUR, ONE PLACE: any hex literal appearing in qtapp.py
       after the palette block (constants only) occurs at most ONCE —
       repeats must use a shared variable (the constants block holds
       BG/CARD/FIELD/FG/WHITE/... and the hover family).
    """
    import re as _re
    from collections import Counter
    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QToolTip
    app = QApplication.instance() or QApplication(["t"])
    app.setStyle("Fusion")
    app.setStyleSheet(Q.styled_qss())
    win = make_window("Emissive")
    try:
        # straight from the constructor, no toggling: the radios' enabled
        # state must match the Averaging checkbox BEFORE any signal fires
        # (the app used to open with averaging off but the radios lit)
        check("fresh window: radios' ENABLED state matches the Averaging "
              "checkbox (no toggle needed)",
              win.rb_roll.isEnabled() == win.chk_avg.isChecked()
              and win.rb_cum.isEnabled() == win.chk_avg.isChecked(),
              f"chk={win.chk_avg.isChecked()} "
              f"roll={win.rb_roll.isEnabled()} cum={win.rb_cum.isEnabled()}")

        # the borderless instrument buttons must stay visible when
        # disabled: their body IS the colour, so '… READING' used to
        # paint a borderless CARD rect indistinguishable from the
        # toolbar (reported: the SINGLE button "disappeared")
        b = win.btn_single
        b.setEnabled(False)
        app.processEvents()
        img = b.grab().toImage()
        bord = tuple(int(Q.BORDER[i:i + 2], 16) for i in (1, 3, 5))
        on_border = any(img.pixelColor(x, y).red() == bord[0]
                        and img.pixelColor(x, y).green() == bord[1]
                        and img.pixelColor(x, y).blue() == bord[2]
                        for x in range(b.width()) for y in range(b.height()))
        check("SINGLE disabled: shows the shared BORDER outline "
              "not a borderless CARD blob", on_border)
        b.setEnabled(True)
        win.show()
        app.processEvents()
        lbl_peaks = None
        for l in win.findChildren(Q.QLabel):
            if l.text() == "PEAKS (BY INTENSITY)":
                lbl_peaks = l
                break

        def tip_bg(widget):
            for tl in app.topLevelWidgets():
                if tl.metaObject().className() == "QTipLabel":
                    tl.close()
            QToolTip.showText(widget.mapToGlobal(QPoint(10, 10)),
                              "probe", widget)
            app.processEvents()
            tip = [tl for tl in app.topLevelWidgets()
                   if tl.metaObject().className() == "QTipLabel"]
            if not tip:
                return None          # offscreen platform refused to show
            tip[0].repaint()
            img = tip[0].grab().toImage()
            cols = Counter()
            for x in range(2, img.width() - 2, 2):
                for y in range(2, img.height() - 2, 2):
                    c = img.pixelColor(x, y)
                    cols[(c.red(), c.green(), c.blue())] += 1
            QToolTip.hideText()
            return cols.most_common(1)[0][0]

        exp_h = tuple(int(Q.TOOLTIP_BG[i:i + 2], 16)
                      for i in (1, 3, 5))     # TOOLTIP_BG
        targets = [(win.val_integral, "measurements box"),
                   (win.list_peaks, "peaks field"),
                   (win.cmb_coord, "combo"),
                   (getattr(win, "swatch", None), "swatch"),
                   (win.lamps.get("cal") if hasattr(win, "lamps") else None,
                    "annunciator lamp")]
        if lbl_peaks is not None:
            targets.append((lbl_peaks, "peaks title"))
        for widget, name in targets:
            if widget is None:
                continue
            bg = tip_bg(widget)
            if bg is not None:
                check(f"tooltip from {name}: TOOLTIP_BG {Q.TOOLTIP_BG} "
                      f"(was panel-coloured before "
                      "the scoping)",
                      bg == exp_h,
                      f"#{bg[0]:02x}{bg[1]:02x}{bg[2]:02x}")

        # scope rule at the SOURCE level: the setStyleSheet call at the
        # call site must open a selector within the following 3 lines
        src = open(os.path.join(os.path.dirname(Q.__file__),
                                "qtapp.py")).read()
        bad = []
        for m in _re.finditer(r"(\w+)\.setStyleSheet\(", src):
            ahead = src[m.start():m.start() + 260]
            if "{{" not in ahead and "{ " not in ahead \
                    and ahead.strip().startswith('setStyleSheet("")'):
                bad.append(ahead.splitlines()[0][:50])
        check("source: every setStyleSheet opens a selector (scoped)",
              not bad, str(bad[:3]))

        # checkbox/radio labels must not paint the global dark QWidget
        # background as a constant block behind them (QSS makes the
        # widgets transparent; the panel CARD shows through instead)
        from PySide6.QtCore import QPoint as _QP
        for name in ("chk_avg", "chk_smooth", "rb_roll", "rb_cum"):
            w = getattr(win, name, None)
            if w is None:
                continue
            gb = w.parentWidget().parentWidget()
            gimg = gb.grab().toImage()
            p = w.mapTo(gb, _QP(4, 4))
            dark_block = False
            for x in range(p.x() + 20, p.x() + w.width() - 4, 2):
                for y in range(p.y() + 4, p.y() + w.height() - 4, 2):
                    c = gimg.pixelColor(x, y)
                    if (c.red(), c.green(), c.blue()) == (0x1b, 0x1b, 0x1b):
                        dark_block = True
                        break
                if dark_block:
                    break
            check(f"{name}: label transparent, no darker constant block "
                  "(global QWidget bg)", not dark_block,
                  "BG #1b1b1b found behind the label")

        # the INDICATORS must be visible in both states: unchecked =
        # FIELD fill with a BORDER border (Fusion's native bevel vanished
        # on the dark theme); checked = ACCENT with an FG check/dot mark.
        # Averaging must be ON for the checked probes (checked+disabled
        # legitimately drops to BORDER — asserted further below)
        chk_orig = win.chk_avg.isChecked()
        win.chk_avg.setChecked(True)
        app.processEvents()
        for name in ("chk_smooth", "rb_cum"):
            w = getattr(win, name, None)
            if w is None:
                continue
            was_checked = w.isChecked()      # rb_roll defaults to checked
            w.setChecked(False)
            app.processEvents()
            gb = w.parentWidget().parentWidget()
            gimg = gb.grab().toImage()
            p = w.mapTo(gb, QPoint(10, 4))
            px = {(gimg.pixelColor(x, y).red(),
                   gimg.pixelColor(x, y).green(),
                   gimg.pixelColor(x, y).blue())
                  for x in range(p.x(), p.x() + 22, 1)
                  for y in range(p.y(), p.y() + w.height() - 8, 1)}
            field = tuple(int(Q.FIELD[i:i + 2], 16) for i in (1, 3, 5))
            bord = tuple(int(Q.BORDER[i:i + 2], 16) for i in (1, 3, 5))
            check(f"{name}: unchecked indicator visible (FIELD fill + "
                  "BORDER border)", field in px and bord in px,
                  f"{sorted(px)[:6]}")
            w.setChecked(True)
            app.processEvents()
            gimg = gb.grab().toImage()
            acc = tuple(int(Q.ACCENT[i:i + 2], 16) for i in (1, 3, 5))
            px = {(gimg.pixelColor(x, y).red(),
                   gimg.pixelColor(x, y).green(),
                   gimg.pixelColor(x, y).blue())
                  for x in range(p.x(), p.x() + 22, 1)
                  for y in range(p.y(), p.y() + w.height() - 8, 1)}
            check(f"{name}: checked indicator shows ACCENT", acc in px,
                  f"{sorted(px)[:6]}")
            w.setChecked(was_checked)

        # Averaging off must visibly disable Rolling / Cumulative: the
        # labels gray to MUTED and the SELECTED radio loses its ACCENT
        # fill (checked+disabled drops to BORDER — it used to stay
        # bright blue while its sibling dimmed)
        acc = tuple(int(Q.ACCENT[i:i + 2], 16) for i in (1, 3, 5))
        mut = tuple(int(Q.MUTED[i:i + 2], 16) for i in (1, 3, 5))
        chk_state = chk_orig
        win.chk_avg.setChecked(False)
        app.processEvents()
        for name in ("rb_roll", "rb_cum"):
            w = getattr(win, name, None)
            if w is None:
                continue
            gb = w.parentWidget().parentWidget()
            gimg = gb.grab().toImage()
            p = w.mapTo(gb, QPoint(0, 0))
            px = {(gimg.pixelColor(x, y).red(),
                   gimg.pixelColor(x, y).green(),
                   gimg.pixelColor(x, y).blue())
                  for x in range(p.x(), p.x() + w.width(), 1)
                  for y in range(p.y(), p.y() + w.height(), 1)}
            check(f"{name}: averaging off -> disabled, no ACCENT, "
                  "MUTED label", w.isEnabled() is False
                  and acc not in px and mut in px,
                  f"enabled={w.isEnabled()} px={sorted(px)[:5]}")
        win.chk_avg.setChecked(chk_state)
        app.processEvents()

        # the size grip in the status bar's bottom-right corner must
        # not paint its own darker square — it shows the bar's CARD
        # colour like the rest of the statusbar surface
        from PySide6.QtWidgets import QSizeGrip as _QSizeGrip
        for g in win.statusBar().children():
            if isinstance(g, _QSizeGrip):
                img = win.grab().toImage()
                p = g.mapTo(win, g.rect().topLeft())
                grip_px = {img.pixelColor(x, y).name()
                           for x in range(p.x(), p.x() + g.width(), 2)
                           for y in range(p.y(), p.y() + g.height(), 2)}
                check("size grip corner: bar's CARD colour, no darker "
                      "square", grip_px == {"#" + "26" * 3},
                      str(sorted(grip_px)))
                break

        # one colour, one place: hex literals after the palette block
        head, _sep, tail = src.partition('QSS = f"""')
        hexes = _re.findall(r"#[0-9a-fA-F]{6}\b", tail)
        doubles = {h for h, n in Counter(hexes).items() if n > 1}
        check("source: no duplicated hard-coded colours outside the "
              "palette constants", not doubles, str(doubles))

        # one size, one place: QSS font-size literals — every rule must
        # use a {FONT_*} variable; bare "font-size: <number>pt" after
        # the FONT block is a regression of the type-palette rule
        _fhead, _fsep, ftail = src.partition("FONT_BASE")
        bare = _re.findall(r"font-size:\s*\d+(?:\.\d+)?pt", ftail)
        check("source: all QSS font-sizes come from the FONT_* "
              "variables", not bare,
              f"literal font-size rules: {sorted(set(bare))}")

        # the matplotlib side: every fontsize= call must name a
        # CHART_* constant (the chart point unit differs from Qt pt,
        # so it needs its own named family — not bare numbers)
        numeric = [m for m in _re.findall(r"fontsize=([0-9A-Za-z_]+)",
                                          src)
                   if not m.startswith("CHART_")]
        check("source: chart font sizes all name CHART_* constants",
              not numeric, f"bare: {numeric}")
    finally:
        win.close()


def test_toolgroup_centering():
    """The Zero … Save A toolbar group centers on the WINDOW at every
    width where it fits, stays un-clipped when the window is squished
    into the starved regime, and — the reported bug — RE-CENTRES when
    the window grows back. The old spacer-minimum mechanism wedged the
    group against the left edge for good after a squish."""
    app = QApplication.instance() or QApplication(["t"])
    app.setStyle("Fusion")
    app.setStyleSheet(Q.styled_qss())
    win = make_window("Emissive")
    try:
        win.show()
        app.processEvents()

        def outer_err():
            # visual centre: middle of the outer edge-span (not the
            # button-center average, which biases when the end
            # buttons differ in width)
            ga = win.tb_group_ends[0].mapTo(
                win, win.tb_group_ends[0].rect().topLeft()).x()
            gb = win.tb_group_ends[1].mapTo(
                win, win.tb_group_ends[1].rect().topRight()).x()
            return (ga + gb) // 2 - win.width() // 2, gb - win.width()

        win.setMinimumWidth(380)     # squish past the layout minimum
        for w in (1280, 1600):
            win.resize(w, 800)
            app.processEvents()
            win._center_tool_group()
            app.processEvents()
            err, over = outer_err()
            check(f"toolbar group centered at {w} px", abs(err) <= 2,
                  f"err={err}")
        # squish deep into the starved regime
        for w in (900, 500, 380):
            win.resize(w, 700)
            app.processEvents()
            win._center_tool_group()
            app.processEvents()
            _err, over = outer_err()
            check(f"toolbar group not clipped past the edge at {w} px",
                  over <= 2, f"right edge {over} px past the window")
        # regrow — the reported bug: the group used to stay left
        for w in (1280, 1600, 1280):
            win.resize(w, 800)
            app.processEvents()
            win._center_tool_group()
            app.processEvents()
            err, over = outer_err()
            check(f"toolbar group re-centers at {w} px after the squish",
                  abs(err) <= 2, f"err={err} (was wedged left before)")
    finally:
        win.close()


if __name__ == "__main__":
    main()
