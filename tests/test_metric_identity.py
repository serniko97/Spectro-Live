"""Two meta-correctness suites:

1. test_averaging_metric_identity — every metric displayed for an
   AVERAGED trace must equal what the same metric gives for the
   precomputed mean spectrum (and for the mean saved to CSV and
   reloaded), to floating tolerance. Averaging, saving, importing and
   live display are different code paths; if they drift apart the user
   sees different numbers for the same physical measurement.

2. test_reference_crossvalidation — our own colour math in
   processing.py, checked against independent references: the
   colour-science package (BSD-3) and scipy. Values are COMPUTED
   INDEPENDENTLY here and compared against tolerances, in the spirit
   of cross-validating against a reference implementation.

Run hermetic:
  QT_QPA_PLATFORM=offscreen HOME=/tmp/opencode/hermetic-home \
  .venv/bin/python tests/test_metric_identity.py
"""
import os
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(TESTS_DIR))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("HOME", "/tmp/opencode/hermetic-home")
os.makedirs("/tmp/opencode/hermetic-home", exist_ok=True)

import numpy as np                                     # noqa: E402
trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy 1.x on macOS has no trapezoid()
import threading                                       # noqa: E402
from PySide6.QtWidgets import QApplication             # noqa: E402

import spectrov.qtapp as Q                              # noqa: E402
from spectrov import sources                           # noqa: E402
from spectrov.processing import (                      # noqa: E402
    Spectrum, average, cri_from_spectrum, cct_from_xyz,
    dominant_wavelength, pchip_smooth, xy_from_xyz, radiometric_from_photometric,
)

FAILURES = []


def check(name, cond, extra=""):
    tag = "ok  " if cond else "FAIL"
    print(f"  [{tag}] {name}" + (f"  -> {extra}" if extra and not cond
                                else (f"    ({extra})" if extra else "")))
    if not cond:
        FAILURES.append(name)


def _make_window():
    class FakeWorker:
        def __init__(self):
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

    old_mk = Q.MainWindow._make_worker
    old_home = os.environ.get("HOME")
    Q.MainWindow._make_worker = lambda self: FakeWorker()
    os.environ["HOME"] = "/tmp/opencode/hermetic-home"
    old_present = sources.instrument_present
    sources.instrument_present = lambda: None
    try:
        app = QApplication.instance() or QApplication(["t"])
        app.setStyle("Fusion")
        win = Q.MainWindow()
    finally:
        Q.MainWindow._make_worker = old_mk
        if old_home is not None:
            os.environ["HOME"] = old_home
        sources.instrument_present = old_present
    win.inst_type = "X-Rite ColorMunki Photo"
    win.mode_name = "Emissive"
    return win


def _spawn(win, spec):
    spec.timestamp = "12:00:00"
    spec.source = "Test"
    win._on_frame(spec)
    return spec


def _readouts(win):
    """A dict of the numeric readouts as PARSED NUMBERS — text formats
    change, the physics underneath must not."""
    import re

    def num(txt):
        m = re.search(r"[-+]?[0-9]+\.?[0-9]*(?:e[-+]?[0-9]+)?",
                      txt.replace(",", ""))
        return float(m.group()) if m else None

    out = {
        "integral": num(win.val_integral.text()),
        "photometric": num(win.val_photometric.text()),
        "radiometric": num(win.val_radiometric.text()),
        "cct": num(win.val_cct.text()),
        "colour": num(win.val_colour.text().splitlines()[0]
                      if win.val_colour.text() else ""),
        "colour_last": num(win.val_colour.text().splitlines()[-1]
                           if win.val_colour.text() else ""),
        "peaks": win.list_peaks.count(),
        "trace": np.asarray(win.trace_a.values, dtype=float)
                 if win.trace_a is not None and len(win.trace_a.values)
                 else None,
        "xyz": (tuple(win.trace_a.xyz)
                if (win.trace_a is not None
                    and getattr(win.trace_a, "xyz", None) is not None)
                else None),
    }
    return out


# ---------------------------------------------------------------------------
# Suite 1 — averaging metric identity
# ---------------------------------------------------------------------------

def test_averaging_metric_identity():
    np.random.seed(20261106)
    wl = np.linspace(380.0, 730.0, 106)
    shape = (np.exp(-((wl - 450.0) / 70.0) ** 2)
             + 0.4 * np.exp(-((wl - 610.0) / 30.0) ** 2))
    frames = []
    for k in range(5):
        vals = 100.0 * shape + np.random.normal(0.0, 0.8, len(wl))
        frames.append(Spectrum(wl, vals, (1.20 + 0.02 * k,
                                          1.30 + 0.01 * k,
                                          2.10 - 0.01 * k)))
    mean_vals = np.mean([f.values for f in frames], axis=0)
    mean_xyz = tuple(np.mean([f.xyz for f in frames], axis=0))
    mean_spec = Spectrum(wl, mean_vals, mean_xyz)

    print("test_averaging_metric_identity:")
    # --- sanity: our own average() must equal an independent np.mean
    avg = average(frames)
    check("average() == np.mean of the frames",
          np.max(np.abs(avg.values - mean_vals)) < 1e-9,
          str(np.max(np.abs(avg.values - mean_vals))))
    check("average() XYZ == mean of the XYZ",
          max(abs(a - b) for a, b in zip(avg.xyz, mean_xyz)) < 1e-9)

    # --- window 1: the GUI fed all frames, averaging ON (rolling N=5)
    w1 = _make_window()
    w1.chk_avg.setChecked(True)
    w1.spin_avg.setValue(5)
    w1.rb_roll.setChecked(True)
    for k, f in enumerate(frames):
        _spawn(w1, f)
    w1._refresh()
    r1 = _readouts(w1)

    # --- window 2: the GUI fed the precomputed mean as ONE frame
    w2 = _make_window()
    _spawn(w2, mean_spec)
    w2._refresh()
    r2 = _readouts(w2)

    check("GUI rolling-average trace == precomputed mean",
          r1["trace"] is not None and r2["trace"] is not None
          and np.max(np.abs(r1["trace"] - r2["trace"])) < 5e-4,
          str(None if r1["trace"] is None or r2["trace"] is None
              else np.max(np.abs(r1["trace"] - r2["trace"]))))
    ig1, ig2 = r1["integral"], r2["integral"]
    check("integral: averaged GUI == mean-frame GUI",
          ig1 is not None and ig2 is not None and abs(ig1 - ig2) < 0.5,
          f"{ig1} vs {ig2}")
    for key in ("photometric", "radiometric", "cct", "colour"):
        a, b = r1[key], r2[key]
        check(f"{key}: averaged GUI == mean-frame GUI",
              a is not None and b is not None and abs(a - b) < 1e-6,
              f"{a} vs {b}")
    check("peak count: averaged GUI == mean-frame GUI",
          r1["peaks"] == r2["peaks"], f"{r1['peaks']} vs {r2['peaks']}")
    # XYZ must survive the averaging path too (feeds xy/CCT/measurement)
    check("averaged GUI keeps mean XYZ",
          r1["xyz"] is not None
          and max(abs(a - b) for a, b in zip(r1["xyz"], mean_xyz)) < 1e-9,
          str(r1["xyz"]))

    # --- window 3: the averaged trace SAVED to CSV and RELOADED
    path = os.path.join("/tmp/opencode", "avg_identity.csv")
    sources.save_csv(path, w1.trace_a)
    loaded = sources.load_csv(path)
    _spawn_layer = None
    w3 = _make_window()
    _spawn(w3, loaded)
    w3._refresh()
    r3 = _readouts(w3)
    check("CSV roundtrip keeps the values (6-decimal writer)",
          r3["trace"] is not None and r1["trace"] is not None
          and np.max(np.abs(r3["trace"] - r1["trace"])) < 2e-4,
          str(None if r3["trace"] is None or r1["trace"] is None
              else np.max(np.abs(r3["trace"] - r1["trace"]))))
    for key in ("photometric", "radiometric"):
        a, b = r1[key], r3[key]
        check(f"{key}: live averaged == saved+reloaded",
              a is not None and b is not None and abs(a - b) < 1e-3,
              f"{a} vs {b}")
    ig3 = r3["integral"]
    check("integral: live averaged == saved+reloaded",
          ig1 is not None and ig3 is not None and abs(ig1 - ig3) < 0.5,
          f"{ig1} vs {ig3}")
    check("loaded CSV carries its stored XYZ",
          getattr(loaded, "xyz", None) is not None,
          "load_csv returns a Spectrum without xyz — metrics relying on "
          "the absolute anchor silently change on reload")

    # --- cumulative averaging over all frames == same mean (N >= count)
    w4 = _make_window()
    w4.chk_avg.setChecked(True)
    w4.rb_cum.setChecked(True)
    for f in frames:
        _spawn(w4, f)
    w4._refresh()
    r4 = _readouts(w4)
    check("cumulative averaging == the same mean as rolling (all frames)",
          r4["trace"] is not None and r1["trace"] is not None
          and np.max(np.abs(r4["trace"] - r1["trace"])) < 5e-4,
          str(None if r4["trace"] is None or r1["trace"] is None
              else np.max(np.abs(r4["trace"] - r1["trace"]))))

    for w in (w1, w2, w3, w4):
        w.close()


# ---------------------------------------------------------------------------
# Suite 2 — cross-validation against independent reference implementations
# ---------------------------------------------------------------------------

def test_reference_crossvalidation():
    print("test_reference_crossvalidation:")
    import colour
    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]

    # --- xy from XYZ ------------------------------------------------------
    for xyz in ((0.9642, 1.0, 0.8251),      # D50 white
                (0.3127 * 2, 0.3290 * 2, 1.07 * 2 / 4),
                (1.4457, 1.4225, 0.3955)):
        ours = xy_from_xyz(xyz)
        theirs = colour.XYZ_to_xy(np.asarray(xyz))
        check(f"xy_from_xyz({xyz[0]:.3f},..) matches colour",
              abs(ours[0] - theirs[0]) < 1e-6
              and abs(ours[1] - theirs[1]) < 1e-6,
              f"{ours} vs {tuple(theirs)}")

    # --- CCT (McCamy cubic, independently reimplemented here) ------------
    for xyz in ((0.9642, 1.0, 0.8251),      # D50 white
                (0.9505, 1.0, 1.089),       # D65 white
                (0.4476, 0.4074, 0.1450)):  # warm ~2900K-ish
        x, y = xy_from_xyz(xyz)
        ours = cct_from_xyz(xyz)
        theirs = float(colour.xy_to_CCT((x, y), method="McCamy 1992"))
        check(f"McCamy CCT({x:.4f}, {y:.4f}) matches colour",
              abs(ours - theirs) < 0.5,
              f"{ours} K vs {theirs} K")
    # the GUI must flag computed CCTs OUTSIDE McCamy's ~2856–25000 K
    # fit range instead of printing a bare (fictitious) number
    win = _make_window()
    for xyz, flagged in (((0.9642, 1.0, 0.8251), False),   # D50, in range
                         ((0.1, 0.02, 0.005), True)):      # far off-locus
        win.frames.clear()
        win._on_frame(Spectrum(np.linspace(380, 730, 106),
                               np.abs(np.sin(np.linspace(0.3, 9, 106)))
                               + 0.05, xyz))
        win._refresh()
        txt = win.val_cct.text()
        check(f"CCT domain caution {'shown' if flagged else 'silent'} "
              f"for XYZ {xyz}",
              ("fit range" in txt) == flagged, txt)
    win.close()

    # --- CMF ybar table vs colour's 1-nm CIE 1931 2° observer -------------
    from spectrov.processing import _CMF_WL, _CMF_Y, _CMF_X
    col_wl = np.asarray(cmfs.wavelengths, dtype=float)
    col_y = np.asarray(cmfs.values)[:, 1]
    col_x = np.asarray(cmfs.values)[:, 0]
    ref_y = np.interp(_CMF_WL, col_wl, col_y)
    ref_x = np.interp(_CMF_WL, col_wl, col_x)
    y555 = float(np.interp(555.0, _CMF_WL, _CMF_Y))
    check("CMF table: ybar(555 nm) == 1.0 exactly",
          abs(y555 - 1.0) < 1e-9, str(y555))
    check("CMF table matches colour's CIE 1931 ybar on our 5-nm grid",
          np.max(np.abs(_CMF_Y - ref_y)) < 2e-4,
          str(np.max(np.abs(_CMF_Y - ref_y))))
    check("CMF table matches colour's CIE 1931 xbar (no column swap)",
          np.max(np.abs(_CMF_X - ref_x)) < 2e-4,
          str(np.max(np.abs(_CMF_X - ref_x))))

    # --- XYZ from spectrum: ratios vs colour's sd_to_XYZ ------------------
    # colour's ASTME308 path only accepts 1/5/10/20 nm grids, so both
    # sides work on a 5 nm grid (ours: trapezoid; colour: ASTM
    # rectangular summation — small method difference, tolerance 2 %)
    wl = np.arange(380.0, 731.0, 5.0)
    for label, vals in (
            ("smooth warm LED", np.exp(-((wl - 450.0) / 60.0) ** 2)
             + 0.5 * np.exp(-((wl - 600.0) / 40.0) ** 2)),
            ("spiky fluorescent", np.exp(-((wl - 435.0) / 8.0) ** 2)
             + 0.9 * np.exp(-((wl - 545.0) / 8.0) ** 2)
             + 0.6 * np.exp(-((wl - 610.0) / 8.0) ** 2))):
        from spectrov.processing import xyz_from_values
        s = Spectrum(wl, vals, None)
        ours = xyz_from_values(s)
        sd = colour.SpectralDistribution(
            dict(zip(np.round(wl, 3),
                     np.round(vals, 6))),
            interpolator=colour.LinearInterpolator)
        theirs = colour.sd_to_XYZ(
            sd, cmfs=cmfs,
            illuminant=colour.SDS_ILLUMINANTS["E"])
        # colour normalizes Y to 100: compare X/Y and Z/Y ratios
        ox, oy, oz = ours[0] / ours[1], 1.0, ours[2] / ours[1]
        tx = theirs[0] / theirs[1]
        tz = theirs[2] / theirs[1]
        check(f"xyz_from_values ratios match colour sd_to_XYZ ({label})",
              abs(ox - tx) < 0.02 and abs(oz - tz) < 0.02,
              f"X/Y {ox:.4f} vs {tx:.4f}, Z/Y {oz:.4f} vs {tz:.4f}")

    # --- sRGB preview vs colour's XYZ_to_sRGB ------------------------------
    # FRAME FIRST: our pipeline assumes D50-referred instrument XYZ
    # (Argyll's spectral XYZ is D50-adapted) and applies Bradford
    # D50->D65 internally; colour's XYZ_to_sRGB expects D65-referred
    # input. Compare in the SAME frame: adapt with colour's own
    # Bradford matrix, then convert.
    from spectrov.processing import srgb_preview, _BRADFORD_D50_D65, \
        _XYZ_TO_RGB_D65
    SD = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]
    Mk = colour.adaptation.matrix_chromatic_adaptation_VonKries(
        colour.xy_to_XYZ(SD["D50"]), colour.xy_to_XYZ(SD["D65"]),
        transform="Bradford")
    check("our Bradford D50->D65 matrix == colour's (±2e-3)",
          float(np.max(np.abs(Mk - _BRADFORD_D50_D65))) < 2e-3,
          str(float(np.max(np.abs(Mk - _BRADFORD_D50_D65)))))
    for label, xyz in (("D50 white", (0.9642, 1.0, 0.8251)),
                       ("warm surface", (0.2045, 0.1850, 0.0793))):
        ours = srgb_preview(xyz, normalize=False)[0]
        theirs = colour.XYZ_to_sRGB(Mk @ np.asarray(xyz)) * 255.0
        ok = all(abs(a - t) <= 2 for a, t in zip(ours, theirs))
        check(f"srgb_preview matches colour XYZ_to_sRGB ({label}, "
              f"D50-referred frame)",
              ok, f"{ours} vs {np.round(theirs, 1)}")

    # --- CRI anchor values from the literature --------------------------
    # Colour's bundled standard illuminants; Ra anchors are textbook:
    # Illuminant A ~ 100 (the reference IS a Planckian of its own CCT),
    # FL2 (cool white standard fluorescent) ~ 64, FL11 ~ 83.
    for name, anchor, tol in (("A", 100.0, 2.5), ("FL2", 64.0, 3.0),
                              ("FL11", 83.0, 3.0)):
        sd = colour.SDS_ILLUMINANTS[name]
        wl_s = np.asarray(sd.wavelengths, dtype=float)
        vals = np.asarray(sd.values, dtype=float)
        m = (wl_s >= 380.0) & (wl_s <= 730.0)   # the instrument's band
        s = Spectrum(wl_s[m], vals[m], None)
        r = cri_from_spectrum(s)
        got = r["Ra"] if r else None
        check(f"CRI {name}: Ra == {anchor} (±{tol}) on the 380–730 band",
              got is not None and abs(got - anchor) <= tol,
              str(got))

    # --- dominant wavelength vs colour -----------------------------------
    # NOTE: the sRGB red primary (0.735, 0.265) is deliberately NOT a
    # test point: it lies ON the purple boundary of the xy diagram, so
    # the dominant/complementary question is degenerate there.
    for label, xyz in (("saturated blue", (0.180, 0.072, 1.0)),
                       ("red-orange", (0.60, 0.35, 0.05)),
                       ("green laser-ish", (0.0, 1.0, 0.0))):
        ours_wl, ours_dom = dominant_wavelength(xyz)
        xyw = np.array([1/3, 1/3])          # our white point: E
        try:
            dw, _r, _xyr = colour.dominant_wavelength(xy_from_xyz(xyz), xyw)
            theirs = abs(float(dw))
        except Exception:                   # noqa: BLE001
            theirs = None
        check(f"dominant_wavelength({label}) vs colour (±5 nm)",
              theirs is not None and abs(abs(ours_wl) - theirs) <= 5.0
              and ours_dom is True,
              f"ours {ours_wl} (dominant={ours_dom}) vs colour {theirs}")

    # --- pchip smoothing vs scipy's PchipInterpolator --------------------
    try:
        from scipy.interpolate import PchipInterpolator
        y = np.array([0.0, 0.1, 1.0, 0.4, 0.05, 0.9, 0.1, 0.0])
        xs = np.arange(len(y)) * 5.0
        dense = np.linspace(xs[0], xs[-1], 200)
        up = int(round(200 / (len(y) - 1)))
        w_sm, v_sm = pchip_smooth(xs, y, up=up if up >= 2 else 2)
        theirs = PchipInterpolator(xs, y)(w_sm)
        check("pchip_smooth == scipy PchipInterpolator",
              np.max(np.abs(v_sm - theirs)) < 1e-9,
              str(np.max(np.abs(v_sm - theirs))))
    except ImportError:
        check("scipy available for pchip cross-check", False, "no scipy")

    # --- radiometric factor vs colour's independent V(λ) ---------------
    # flat spectrum: Xe = Y·∫S/(683·∫S·V), with V recomputed from
    # colour's own 1-nm CMF table — an independent copy of the standard
    wl5 = np.arange(380.0, 731.0, 5.0)
    flat = 3.7
    ours = radiometric_from_photometric(
        Spectrum(wl5, np.full_like(wl5, flat), None), 100.0)
    v = np.interp(wl5, col_wl, col_y)
    v = v / float(np.interp(555.0, col_wl, col_y))
    theirs = 100.0 * (trapz(np.full_like(wl5, flat), wl5)
                      / (683.0 * trapz(np.full_like(wl5, flat) * v,
                                              wl5)))
    check("radiometric flat-spectrum factor matches colour's V(λ)",
          ours is not None and abs(ours - theirs) / theirs < 0.01,
          f"{ours} vs {theirs}")


if __name__ == "__main__":
    test_averaging_metric_identity()
    test_reference_crossvalidation()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): " + "; ".join(FAILURES))
        sys.exit(1)
    print("all metric-identity tests passed")
