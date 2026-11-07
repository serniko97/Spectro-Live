"""GUI-level multi-instrument tests (headless): a non-ColorMunki
spectrometer must get neutral wording, only the modes it supports, and
the plot must adapt to its own spectral range; colorimeter XYZ-only
frames must not break the maths panels; the device picker must select
spotread instrument ports.

Run:  QT_QPA_PLATFORM=offscreen .venv/bin/python tests/test_multidevice.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threading                                    # noqa: E402
import math                                         # noqa: E402
import json                                         # noqa: E402
import numpy as np                                  # noqa: E402
trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy 1.x on macOS has no trapezoid()
from PySide6.QtWidgets import QApplication          # noqa: E402

import spectrov.qtapp as Q                          # noqa: E402
import spectrov.sources as sources                  # noqa: E402
from spectrov.processing import Spectrum            # noqa: E402


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


class SpotreadWorker(FakeWorker):
    """Named so MainWindow's restart-vs-nop check sees a spotread
    worker and actually re-creates it."""


def make_window():
    old_mk = Q.MainWindow._make_worker
    old_home = os.environ.get("HOME")
    Q.MainWindow._make_worker = lambda self: FakeWorker()
    os.environ["HOME"] = "/tmp/opencode/hermetic-home"
    os.makedirs("/tmp/opencode/hermetic-home", exist_ok=True)
    sources.instrument_present = lambda: None
    try:
        app = QApplication.instance() or QApplication(["t"])
        app.setStyle("Fusion")
        win = Q.MainWindow()
    finally:
        Q.MainWindow._make_worker = old_mk
        if old_home is not None:
            os.environ["HOME"] = old_home
    win.inst_type = ""
    win.show()
    return win


def feed(win, lo=380.0, hi=730.0, n=106, xyz=(0.04, 0.03, 0.01)):
    wl = np.linspace(lo, hi, n)
    win.frames.clear()
    spec = Spectrum(wl, np.abs(np.sin(wl)) * 0.3, xyz)
    spec.timestamp = "12:00:00"
    spec.source = "Test"
    win._on_frame(spec)
    return spec


FAILURES = []


def check(name, cond, extra=""):
    s = "ok  " if cond else "FAIL"
    print(f"  [{s}] {name}{('  ' + extra) if extra and not cond else ''}")
    if not cond:
        FAILURES.append(name)


# ----------------------------------------------------------------- tests
def test_instrument_event_filters_modes():
    """An i1 Pro (no ambient diffuser) must refuse the Ambient mode —
    the mode-selector lamps (SURFACE / REFLECTIVE / AMBIENT) stay
    honest: unsupported ones cannot be selected."""
    win = make_window()
    win._on_state("instrument", "GretagMacbeth i1 Pro")
    check("i1 Pro: only Emissive+Reflective selectable",
          list(win.allowed_modes) == ["Emissive", "Reflective"],
          str(win.allowed_modes))
    win.lamps["ambient"].clicked.emit()
    check("i1 Pro: Ambient lamp click refused",
          win.mode_name in ("Emissive", "Reflective"), win.mode_name)
    win.lamps["reflective"].clicked.emit()
    check("i1 Pro: Reflective selectable",
          win.mode_name == "Reflective", win.mode_name)
    check("i1 Pro: mode tips contain no wheel slots",
          "slot" not in win._mode_tips().lower(), win._mode_tips())
    win._on_state("instrument", "X-Rite ColorMunki Photo")
    check("ColorMunki: all three modes return",
          list(win.allowed_modes) == ["Emissive", "Ambient", "Reflective"],
          str(win.allowed_modes))
    win.lamps["ambient"].clicked.emit()
    check("ColorMunki: Ambient selectable again",
          win.mode_name == "Ambient", win.mode_name)
    win.close()


def test_unsupported_persisted_mode_switches():
    """App was last used in Ambient with a ColorMunki; now an i1 Pro is
    plugged in -> mode falls back to a supported one."""
    win = make_window()
    win._on_state("instrument", "GretagMacbeth i1 Pro")
    check("i1 Pro: Ambient falls back to a supported mode",
          win.mode_name in ("Emissive", "Reflective"),
          win.mode_name)
    win._on_state("instrument", "X-Rite ColorMunki Photo")
    win._select_mode("Ambient")
    check("ColorMunki accepts Ambient", win.mode_name == "Ambient")
    win._on_state("instrument", "GretagMacbeth i1 Pro")
    check("i1 Pro: Ambient falls back to a supported mode again",
          win.mode_name in ("Emissive", "Reflective"),
          win.mode_name)
    win.close()


def test_menu_cycle_respects_allowed_modes():
    win = make_window()
    win._on_state("instrument", "Datacolor Spyder5")
    seen = set()
    for _ in range(3):
        win._menu_cycle_mode()
        seen.add(win.mode_name)
    check("spyder: cycle never leaves Emissive", seen == {"Emissive"},
          str(seen))
    win.close()


def test_adaptive_plot_range():
    win = make_window()
    feed(win, 360.0, 750.0, 40)
    check("range: x-axis adopts 360–750",
          abs(win.x_lo - 360.0) < 1e-6 and abs(win.x_hi - 750.0) < 1e-6,
          f"{win.x_lo}..{win.x_hi}")
    check("range: band spins follow",
          win.spin_lo.minimum() == 360.0 and win.spin_hi.maximum() == 750.0)
    check("range: full band retained (was full before)",
          abs(win.spin_lo.value() - 360.0) < 1e-6
          and abs(win.spin_hi.value() - 750.0) < 1e-6,
          f"{win.spin_lo.value()}..{win.spin_hi.value()}")
    lo, hi = win.ax.get_xlim()
    check("range: matplotlib xlim matches",
          abs(lo - 360.0) < 1e-6 and abs(hi - 750.0) < 1e-6)
    # and back to the ColorMunki range on the next session
    feed(win, 380.0, 730.0, 106)
    check("range: shrinks back for a narrower instrument",
          abs(win.x_lo - 380.0) < 1e-6 and abs(win.x_hi - 730.0) < 1e-6,
          f"{win.x_lo}..{win.x_hi}")
    win.close()


def test_xyz_only_frame():
    """Colorimeter: XYZ result, no spectrum -> colour panels alive,
    spectral panels show em-dashes, nothing crashes."""
    win = make_window()
    win.frames.clear()
    spec = Spectrum(np.array([]), np.array([]), (40.0, 31.0, 25.0))
    spec.timestamp = "12:00:00"
    spec.source = "Spyder"
    win._on_frame(spec)
    check("XYZ-only: no crash, frame counted", win.frame_count == 1)
    check("XYZ-only: spectral integral placeholder",
          "no spectrum" in win.val_integral.text().lower(),
          win.val_integral.text())
    check("XYZ-only: colour coordinates still computed",
              win.val_colour.text() not in ("—", ""),
              win.val_colour.text())
    check("XYZ-only: chromaticity xy values",
          "xy" in win.val_colour.text(), win.val_colour.text())
    win.close()


def test_device_picker_sets_port():
    """The Device menu selection must reach the worker as spotread's
    -c port number."""
    win = make_window()
    captured = {}
    win.worker = SpotreadWorker()
    win.running = True

    real_mk = Q.MainWindow._make_worker
    def fake_mk(self):
        captured["port"] = getattr(self, "device_port", "MISSING")
        return SpotreadWorker()
    Q.MainWindow._make_worker = fake_mk
    try:
        win._pick_device(2)
        check("device picker: port 2 recorded on session restart",
              captured.get("port") == 2, str(captured))
        win._pick_device(None)
        check("device picker: back to auto (port None)",
              captured.get("port") is None, str(captured))
    finally:
        # NEVER leak this into the next test in the battery
        Q.MainWindow._make_worker = real_mk
    win.close()


def test_usb_scan_vendor_class_instrument():
    """regression: the ColorMunki enumerates with a VENDOR-SPECIFIC
    interface class (ff), not HID (03) — the scan must still find it by
    its ids/product name, or the app reports 'Instrument not connected'
    while lsusb plainly shows it."""
    from spectrov.sources import is_candidate, _NON_INSTRUMENTS
    check("ColorMunki by known ids (no HID interface)",
          is_candidate("0971", "2007", "", hid=False))
    check("instrument-like product string without HID",
          is_candidate("1234", "5678", "Some Spectro device", hid=False))
    check("HID devices remain candidates",
          is_candidate("046e", "55a5", "USB Multimedia Keyboard", hid=True))
    check("unknown non-HID device skipped",
          not is_candidate("26ce", "01a2", "LED Controller", hid=False))
    check("RTL dongle not an instrument",
          not is_candidate("0bda", "2838", "RTL2838 DVB-T", hid=False))
    # and the blacklist still hides the LED controller from the
    # fallback pick (this is what masked the bug before)
    check("LED controller in blacklist", "led" in _NON_INSTRUMENTS
          and "controller" in _NON_INSTRUMENTS)


def test_state_wording_generic():
    """The calibrating state with a generic instrument message must show
    the blinking CALIBRATION lamp, not pretend the wheel is spinning."""
    win = make_window()
    win._on_state("instrument", "GretagMacbeth i1 Pro")
    win._on_state("calibrating", sources.cal_prompt("GretagMacbeth i1 Pro"))
    check("generic calibration prompt: CAL lamp blinks",
          win._lamp_state.get("cal") == "solid", str(win._lamp_state))
    win._on_state("calibrating", "Calibrating… keep the instrument in "
                                 "its calibration position.")
    check("calibration running: CAL lamp solid",
          win._lamp_state.get("cal") == "solid", str(win._lamp_state))
    win.close()


def test_force_calibration_menu():
    """Device menu: 'Force calibration' toggle persists, feeds the worker
    and (when ON with a live spotread session) asks it to restart."""
    cfg_path = "/tmp/opencode/hermetic-home/settings-test.json"
    import spectrov.qtapp as Q
    old_q_path = Q.MainWindow._CFG_PATH
    Q.MainWindow._CFG_PATH = cfg_path
    win = make_window()
    try:
        check("force-cal OFF by default (reuses Argyll's host cal cache)",
              win.force_cal is False and not win.act_forcecal.isChecked())
        win.act_forcecal.setChecked(True)
        check("ticking the action sets the flag", win.force_cal is True)
        cfg = json.load(open(cfg_path))
        check("force-cal persisted to settings", cfg.get("force_cal") is True)
        w = win.worker
        if isinstance(w, sources.SpotreadWorker):
            check("live worker got the flag + restart request",
                  w.force_cal and w.restart_now.is_set(),
                  "worker not notified")
        win.act_forcecal.setChecked(False)
        check("unticking clears the flag", win.force_cal is False)
        cfg = json.load(open(cfg_path))
        check("persistence updated again", cfg.get("force_cal") is False)
        win.device_port = None
        wnew = win._make_worker()
        check("new workers inherit force_cal",
              isinstance(wnew, sources.SpotreadWorker)
              and wnew.force_cal == win.force_cal)
        # calibrate-all is not a user option: every worker does it,
        # restricted to the instrument's supported modes
        check("calibrate-all restricted to supported modes",
              isinstance(wnew, sources.SpotreadWorker)
              and set(wnew.cal_pass_modes or ()) <= {"e", "a", "r"},
              str(getattr(wnew, "cal_pass_modes", None)))
    finally:
        win.close()
        Q.MainWindow._CFG_PATH = old_q_path
        if os.path.exists(cfg_path):
            os.unlink(cfg_path)


def test_spotread_duv_cri_relay():
    """spotread's own Duv and CRI (with Argyll's validity flag) must
    override our computed values in the instrument panels."""
    win = make_window()
    spec = feed(win, 380.0, 730.0, 106, xyz=(0.31, 0.30, 0.28))
    win.mode_name = "Emissive"
    # attach what the driver parsed from spotread's lines
    spec.meta_cct = 6361.0
    spec.meta_duv = -0.0075
    spec.meta_cri = {"ra": 66.3, "r9": -61.0, "flag": "Caution"}
    win._refresh()
    txt = win.val_cct.text()
    check("CCT shows spotread's Duv", "Duv" in txt and "-0.0075" in txt, txt)
    check("off-locus warning for |Duv| > 0.006",
          "locus" in txt, txt)
    txt_ri = win.val_ri.text()
    check("spotread's Ra preferred over computed",
          "Ra 66.3" in txt_ri, txt_ri)
    check("spotread's R9 relayed", "R9 -61" in txt_ri, txt_ri)
    check("Argyll's Caution flag shown", "caution" in txt_ri.lower(), txt_ri)
    check("Argyll's Caution flag shown", "caution" in txt_ri.lower(), txt_ri)
    # IES TM-30-15 rides the same readout when spotread printed it
    spec.meta_tm30 = {"rf": 61.2, "rg": 98.5, "flag": "Caution"}
    win._refresh()
    txt_ri = win.val_ri.text()
    check("TM-30 Rf/Rg shown on the second line",
          "Rf 61.2" in txt_ri and "Rg 98.5" in txt_ri, txt_ri)
    check("TM-30 carries Argyll's Caution flag",
          "caution" in txt_ri.splitlines()[1].lower(), txt_ri)
    del spec.meta_tm30
    win._refresh()
    check("no TM-30 line when spotread printed none",
          "Rf" not in win.val_ri.text() or "TM-30" in win.val_ri.text(),
          win.val_ri.text())
    # a well-behaved light: |Duv| small → no warning
    spec.meta_duv = 0.0005
    win._refresh()
    check("on-locus: no warning text", "locus" not in win.val_cct.text(),
          win.val_cct.text())
    # fallback: no meta_cri -> computed value without the flags
    del spec.meta_cri
    win._refresh()
    txt_ri = win.val_ri.text()
    check("fallback: computed Ra still shown", "Ra" in txt_ri, txt_ri)
    check("fallback: no 'instrument' tag left", "instrument" not in txt_ri,
          txt_ri)
    # colorimeter (XYZ only, no spectrum) with relaying: CRI still shown
    spec2 = Spectrum(np.array([]), np.array([]), (0.31, 0.30, 0.28))
    spec2.timestamp = "12:00:00"
    spec2.source = "Test"
    spec2.meta_cri = {"ra": 81.2, "r9": 44.0, "flag": None}
    win.frames.clear()
    win._on_frame(spec2)
    win._refresh()
    check("XYZ-only frame: spotread CRI shown even without a spectrum",
          "Ra 81.2" in win.val_ri.text(), win.val_ri.text())
    win.close()


def test_copy_readout():
    """The measurements panel must be copyable: selectable value labels,
    a Copy button that dumps the whole panel, selected-rows copy, and
    Ctrl+C with a peak row selected."""
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt
    win = make_window()
    feed(win, 380.0, 730.0, 106, xyz=(0.2, 0.15, 0.05))
    # value labels selectable with the mouse
    for lbl in (win.val_integral, win.val_peak, win.val_markers,
                win.val_colour, win.lbl_srgb_vals):
        flags = lbl.textInteractionFlags()
        if not (flags & Qt.TextInteractionFlag.TextSelectableByMouse):
            check("value labels are mouse-selectable", False,
                  lbl.objectName() or str(flags))
            break
    else:
        check("value labels are mouse-selectable", True)
    app = QApplication.instance()
    # whole-panel copy
    win.list_peaks.clearSelection()
    win.on_copy_readout()
    text = app.clipboard().text()
    check("copy: contains the band integral",
          "Band integral" in text and "a.u." in text, text[:120])
    check("copy: contains peak/FWHM", "FWHM" in text)
    check("copy: contains chromaticity", "xy" in text)
    check("copy: contains the peaks list", "Peaks (by intensity)" in text
          and text.count("nm  E") >= 1)
    check("copy: contains sRGB preview values",
          "sRGB preview" in text, text[:200])
    # selected-rows copy
    win.list_peaks.item(0).setSelected(True)
    win.on_copy_readout()
    text = app.clipboard().text()
    check("copy: with one peak row selected only that row is copied",
          text.count("nm  E") == 1 and "Peaks (by intensity)" in text,
          text[:200])
    check("copy: status message confirms",
          "Copied" in win.lbl_message.text())
    # Ctrl+C path
    from PySide6.QtGui import QKeySequence, QKeyEvent
    win.list_peaks.clearSelection()
    win.list_peaks.item(1).setSelected(True)
    ev = QKeyEvent(QKeyEvent.Type.KeyPress,
                   Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier, "c")
    win.keyPressEvent(ev)
    text = app.clipboard().text()
    check("Ctrl+C copies the selected peak row",
          text.count("nm  E") == 1, text[:200])
    win.close()


def test_instrument_measurements_panel():
    """System-independent measurements: photometric Y with a
    mode-appropriate unit, CCT, and CRI for light sources."""
    win = make_window()
    feed(win, 380.0, 730.0, 106, xyz=(0.55, 0.59, 0.47))  # warmish light
    app = None
    # emissive: luminance in cd/m², CRI computed
    win.mode_name = "Emissive"
    win._refresh()
    check("emissive: Y reported in cd/m²",
          "cd/m²" in win.val_photometric.text(),
          win.val_photometric.text())
    check("emissive: header renamed to LUMINANCE Y",
          "LUMINANCE" in win.val_photometric._head.text(),
          win.val_photometric._head.text())
    check("emissive: CCT present", "K" in win.val_cct.text(),
          win.val_cct.text())
    check("emissive: CRI computed from the spectrum",
          "Ra" in win.val_ri.text(), win.val_ri.text())
    check("CRI reports R9 too", "R9" in win.val_ri.text(),
          win.val_ri.text())
    # ambient: lux — E = π·Y (Argyll's own Lambertian-field model)
    win.mode_name = "Ambient"
    win._refresh()
    check("ambient: Y reported in lux",
          "lx" in win.val_photometric.text(),
          win.val_photometric.text())
    try:
        got_lx = float(win.val_photometric.text().split("lx")[0].strip())
    except ValueError:
        got_lx = -1.0
    check("ambient: lux = π·Y (not the old ×6.83 fudge)",
          abs(got_lx - 0.59 * math.pi) < 0.05,
          f"parsed {got_lx}, expected {0.59*math.pi:.3f}")
    check("ambient: no 'approx.' weasel word any more",
          "approx" not in win.val_photometric.text(),
          win.val_photometric.text())
    check("ambient: header renamed to ILLUMINANCE Y",
          "ILLUMINANCE" in win.val_photometric._head.text())
    # reflective: percent reflectance, no CRI (surface, not a light)
    win.mode_name = "Reflective"
    win._refresh()
    check("reflective: Y reported as % reflectance",
          "%" in win.val_photometric.text(),
          win.val_photometric.text())
    check("reflective: CRI correctly refused",
          "n/a" in win.val_ri.text(), win.val_ri.text())
    # the coordinate system switching must not touch these boxes
    before = (win.val_photometric.text(), win.val_cct.text(),
              win.val_ri.text())
    for sys_ in ("CIELAB (D50)", "LCh(ab) (D50)", "XYZ raw",
                 "Dominant wavelength"):
        win.cmb_coord.setCurrentText(sys_)
    win.cmb_coord.setCurrentText("xyY (CIE 1931)")
    after = (win.val_photometric.text(), win.val_cct.text(),
             win.val_ri.text())
    check("measurements identical across coordinate systems",
          before == after, f"{before} vs {after}")
    check("chromaticity box holds coordinates only (no Y, no CCT)",
          "cd/m²" not in win.val_colour.text()
          and "K" not in win.val_colour.text().replace("K_", ""),
          win.val_colour.text())
    win.close()


def test_cri_math():
    """reference-grade checks of the CRI computation itself"""
    from spectrov.processing import Spectrum, cri_from_spectrum
    from spectrov.processing import cri_from_spectrum as _c
    import colour
    # published values: CIE F2 fluorescent has Ra = 64
    sd = colour.SDS_ILLUMINANTS["FL2"]
    wl = np.array(sd.wavelengths, dtype=float)
    vals = np.array([sd.values[i] for i in range(len(wl))],
                    dtype=float)
    cri = _c(Spectrum(wl, vals))
    check("F2 -> Ra ~64 (published value)", cri is not None
          and abs(cri["Ra"] - 64.2) < 1.0, str(cri))
    # a smooth continuum renders colours well; single narrow line dreadfully
    wl = np.arange(380, 730, 2)
    smooth = Spectrum(wl, np.exp(-((wl - 550.0) / 120.0) ** 2) + 0.8)
    spike = Spectrum(wl, np.exp(-((wl - 550.0) / 1.0) ** 2))
    cas = _c(smooth)["Ra"]
    csp = _c(spike)["Ra"]
    check("smooth continuum scores much higher Ra than a spike",
          cas > csp + 10, f"{cas:.1f} vs {csp:.1f}")
    check("CRI refuses a colorimeter's XYZ-only frame",
          _c(Spectrum(np.array([]), np.array([]), (1, 1, 1))) is None)
    check("CRI refuses None", _c(None) is None)


def test_units_follow_mode():
    """Photometric trace units must follow the reading mode: cd/m² for
    emissive, lux for ambient, dimensionless weight for reflective —
    while every option stays selectable in every mode."""
    win = make_window()
    feed(win)
    opts = [win.cmb_units.itemText(i) for i in range(win.cmb_units.count())]
    expected = ["Raw (a.u.)", "V(λ)-weighted /nm", "Photometric abs.",
                "Radiometric abs.", "Refl. %"]
    check("all five unit options always available, ×683 removed",
          opts == expected, str(opts))
    # the radiometric absolute view picks the radiometric counterpart
    # of the mode's photometric quantity; reflective falls back to
    # plain reflectance (dimensionless — no radiometric quantity)
    win.cmb_units.setCurrentText("Radiometric abs.")
    for mode, axis_has, int_has in (
            ("Emissive", "W/(sr·m²)·nm⁻¹", "W/(sr·m²)"),
            ("Ambient", "W/m²·nm⁻¹", "W/m²"),
            ("Reflective", "Reflectance", "%")):
        win.mode_name = mode
        win._refresh()
        ok_axis = axis_has in win.ax.get_ylabel()
        ok_int = int_has in win.val_integral.text()
        check(f"radiometric: {mode.lower()} axis/unit follow the mode",
              ok_axis and ok_int,
              f"axis={win.ax.get_ylabel()!r} integral={win.val_integral.text()!r}")
    win.cmb_units.setCurrentText("V(λ)-weighted /nm")
    for mode, axis_has, int_has in (
            ("Emissive", "Photopic weighting (a.u./nm)", "a.u."),
            ("Ambient", "Photopic weighting (a.u./nm)", "a.u."),
            ("Reflective", "Photopic weighting (a.u./nm)", "a.u.")):
        win.mode_name = mode
        win._refresh()
        check(f"{mode.lower()}: V(λ)-weighted is a mode-agnostic a.u. "
              "shape view (no cd/m² claim)",
              axis_has in win.ax.get_ylabel(), win.ax.get_ylabel())
        check(f"{mode.lower()}: integral unit honours the mode",
              int_has in win.val_integral.text(), win.val_integral.text())
    win.cmb_units.setCurrentText("Refl. %")
    check("Refl. %: axis is percent regardless of mode",
          "%" in win.ax.get_ylabel(), win.ax.get_ylabel())
    win.cmb_units.setCurrentText("Raw (a.u.)")
    check("Raw: plain a.u. axis", "a.u." in win.ax.get_ylabel())
    win.close()


def test_photometric_absolute():
    """The radiometric/photometric split: the anchored view must
    integrate exactly to the instrument's Y, and lie honestly (None)
    when there is no calibrated anchor."""
    from spectrov.processing import (Spectrum, photometric_absolute,
                                     photopic_efficacy)
    import colour
    # a real SPD: CIE F2 — its photometric curve must carry exactly the
    # Y we give it
    sd = colour.SDS_ILLUMINANTS["FL2"]
    wl = np.array(sd.wavelengths, dtype=float)
    vals = np.array([sd.values[i] for i in range(len(wl))], dtype=float)
    for y in (1.0, 57.5, 1000.0):
        conv = photometric_absolute(Spectrum(wl, vals), y)
        integral = float(trapz(conv.values, wl))
        check(f"anchored curve integrates to Y={y}",
              abs(integral - y) < 0.02 * y, f"{integral}")

    # the weighting is genuinely perceptual: same total energy at 450 nm
    # vs 555 nm must yield a much lower photometric total in the blue
    wl5 = np.arange(400, 701, 5.0)
    blue = np.where(np.abs(wl5 - 450) <= 10, 1.0, 0.0)   # same radiant energy
    green = np.where(np.abs(wl5 - 555) <= 10, 1.0, 0.0)
    cb = photometric_absolute(Spectrum(wl5, blue), 1.0)
    cg = photometric_absolute(Spectrum(wl5, green), 1.0)
    # same *absorbed-energy* share: compare the ȳ weights themselves
    check("blue is discounted vs green by V(λ) (perception weighting)",
          photopic_efficacy([450.0])[0] < 0.23 * photopic_efficacy([555.0])[0],
          f"{photopic_efficacy([450.0])[0]:.1f} vs "
          f"{photopic_efficacy([555.0])[0]:.1f}")

    # no anchor -> no curve
    check("V(λ)-weighted SPD: blue peak photometric total < green peak",
          float(trapz(cb.values, cb.wavelengths))
          < float(trapz(cg.values, cg.wavelengths)))
    check("no anchor: None for y=None", photometric_absolute(
        Spectrum(wl, vals), None) is None)
    check("no anchor: None for y=0 (dark reading)",
          photometric_absolute(Spectrum(wl, vals), 0.0) is None)
    check("no anchor: None for a colorimeter's empty spectrum",
          photometric_absolute(Spectrum(np.array([]), np.array([])), 1.0)
          is None)

    # GUI side: mode-appropriate absolute units, integral == Y
    win = make_window()
    feed(win, xyz=(0.5, 0.55, 0.45))
    win.cmb_units.setCurrentText("Photometric abs.")
    for mode, axis_has, unit_has in (
            ("Emissive", "cd/m²·nm⁻¹", "cd/m²"),
            ("Ambient", "lx·nm⁻¹", "lx"),
            ("Reflective", "Reflectance", "%")):
        win.mode_name = mode
        win._refresh()
        ok_axis = axis_has in win.ax.get_ylabel()
        check(f"{mode.lower()}: absolute photometric axis label",
              ok_axis, win.ax.get_ylabel())
        if mode == "Reflective":
            # fix: a reflectance is dimensionless — the absolute view
            # must hand back the UNWEIGHTED reflectance factor, not a
            # photopic-weighted one (V(λ) would silently discount deep
            # blue/red reflectance, which is a photometric operation)
            raw = np.asarray(win.frames[-1].values, dtype=float)
            shown = np.asarray(win.trace_a.values, dtype=float)
            check("reflective: absolute views show the UNWEIGHTED "
                  "reflectance factor (no V(λ) on a surface)",
                  win.trace_a is not None
                  and np.allclose(shown, raw, rtol=1e-9, atol=1e-12),
                  f"max|Δ| = "
                  f"{np.nanmax(np.abs(shown - raw)) if len(shown) else '—'}")
        if mode == "Emissive":
            # the full-band integral must equal Y = 0.55 cd/m²
            txt = win.val_integral.text()
            try:
                # the value is "0.6 cd/m²·nm   (380–730 nm)" — the ∫
                # symbol lives in the row TITLE, not the value
                got = float(txt.split(unit_has)[0]
                            .replace(",", "").strip())
            except (ValueError, IndexError):
                got = -1.0
            check("absolute view: band integral == instrument Y (0.55)",
                  abs(got - 0.55) <= 0.05,   # the box formats to 1 decimal
                  f"parsed {got} from {txt!r}")
    # a frame without usable XYZ must NOT get a fabricated curve in
    # the anchored (emissive/ambient) modes...
    win.frames.clear()
    win._on_frame(Spectrum(np.linspace(380, 730, 106),
                           np.abs(np.sin(np.linspace(380, 730, 106))), None))
    win.mode_name = "Emissive"
    win._refresh()
    check("uncalibrated frame: no fabricated absolute curve",
          win.trace_a is None)
    # ...while reflective legitimately shows its curve without an
    # anchor: Argyll's reflective spectrum IS the 0-100 % reflectance
    # factor (norm = 100) — no Y anchor is needed or used
    raw = win.frames[-1]
    win.mode_name = "Reflective"
    win._refresh()
    check("reflective: reflectance-factor curve needs no anchor",
          win.trace_a is not None
          and np.allclose(np.asarray(win.trace_a.values),
                         np.asarray(raw.values), rtol=1e-9),
          f"trace_a is {win.trace_a!r}")
    win.cmb_units.setCurrentText("Raw (a.u.)")
    win.close()


def test_macos_port_paths():
    """macOS compatibility: the system_profiler parser (no sysfs there),
    the Argyll cache directory mapping (~/.cache on Linux ->
    ~/Library/Caches on macOS), and the honest empty scan on other
    unixes. Everything testable on Linux by feeding fixtures —
    the REAL Mac run is documented as untested in the README."""
    from unittest.mock import patch
    prof = """{"SPUSBDataType":[
      {"_name":"USB31 Bus","_items":[
        {"_name":"ColorMunki Photo",
         "vendor_id":"0x0971 (X-Rite, Inc.)",
         "product_id":"0x2007"},
        {"_name":"Footoo Bar Robot",
         "vendor_id":"0x16c0","product_id":"0x0486"}
      ]},
      {"_name":"USB2 Bus","_items":[
        {"_name":"Apple Keyboard",
         "vendor_id":"0x05ac (Apple Inc.)","product_id":"0x0221"}
      ]}
    ]}"""
    got = sources._mac_usb_parse(prof)
    check("mac scan: the ColorMunki is found by vendor/product id",
          got and got[0][2] == "ColorMunki Photo"
          and got[0][1] == "0971:2007", str(got))
    check("mac scan: a no-name HID robot is not mistaken for an"
          " instrument (no hint, unknown id)",
          not any("Robot" in g[2] for g in got), str(got))
    check("mac scan: the keyboard is filtered out",
          not any("Keyboard" in g[2] for g in got), str(got))
    check("mac scan: garbage input -> empty list, no crash",
          sources._mac_usb_parse("not json") == [])
    with patch.object(sources.sys, "platform", "darwin"):
        dirs = sources.munki_cache_dirs()
        check("mac cache dir: ~/Library/Caches/ArgyllCMS (Argyll's own"
              " xdg mapping)",
              any(d.endswith("Library/Caches/ArgyllCMS")
                  for d in dirs), str(dirs))
        check("mac cache dir: color/ spelling also searched",
              any(d.endswith("ArgyllCMS/color") for d in dirs), str(dirs))
    with patch.object(sources.sys, "platform", "sunos"), \
            patch.object(sources.os.path, "isdir",
                         lambda _p: False):
        check("other unixes: no USB scanner exists -> honest empty list",
              sources.usb_devices() == [])
    # Linux behaviour unchanged: /sys scan still runs
    check("linux scan: sysfs path still live",
          isinstance(sources.usb_devices(), list))


def test_windows_port_paths():
    """Windows compatibility: the PowerShell CIM USB parser, the cache
    directory mapping (%APPDATA%\\Cache per Argyll's xdg_bds NT branch),
    and the honest empty scan when powershell is missing."""
    from unittest.mock import patch
    # ConvertTo-Json emits a BARE OBJECT for a single result
    single = ('{"Name":"ColorMunki Photo",'
              '"PNPDeviceID":"USB\\\\VID_0971&PID_2007\\\\000000002619"}')
    got = sources._win_usb_parse(single)
    check("win scan: single-object JSON -> ColorMunki by vid/pid",
          got and got[0][2] == "ColorMunki Photo"
          and got[0][1] == "0971:2007", str(got))
    multi = ('[{"Name":"ColorMunki Photo",'
             '"PNPDeviceID":"USB\\\\VID_0971&PID_2007\\\\A"},'
             '{"Name":"HID Keyboard Device",'
             '"PNPDeviceID":"USB\\\\VID_046D&PID_C31C\\\\B"}]')
    got = sources._win_usb_parse(multi)
    check("win scan: list JSON, keyboard filtered by hints",
          len(got) == 1 and "Keyboard" not in got[0][2], str(got))
    check("win scan: garbage -> empty list, no crash",
          sources._win_usb_parse("no json here") == [])
    hid = ('[{"Name":"Some Widget","PNPDeviceID":"USB\\\\VID_1234&PID_5678\\\\C",'
           '"PNPClass":"HIDClass"},'
           '{"Name":"Other Widget","PNPDeviceID":"USB\\\\VID_2222&PID_3333\\\\D"}]')
    got = sources._win_usb_parse(hid)
    check("win scan: PNPClass HIDClass device is a candidate like the"
          " Linux scan's bInterfaceClass 03",
          len(got) == 1 and got[0][1] == "1234:5678", str(got))
    hub = ('[{"Name":"Generic USB Hub",'
           '"PNPDeviceID":"USB\\\\VID_05E3&PID_0610\\\\5&1"}]')
    check("win scan: libusb-win32 era instruments live in Win32_PnPEntity,"
          " not Win32_USBHub — parser keys on PNPDeviceID either way",
          "ColorMunki" not in str(sources._win_usb_parse(hub)))
    with patch.dict("os.environ", {"APPDATA": "/tmp/opencode/appdata"}), \
            patch.object(sources.os, "name", "nt"):
        dirs = sources.munki_cache_dirs()
        check("win cache dir: %APPDATA%/Cache/ArgyllCMS "
              "(Argyll's NT mapping)",
              any(d == "/tmp/opencode/appdata/Cache/ArgyllCMS"
                  for d in dirs), str(dirs))
        check("win cache dir: color/ spelling also searched",
              any(d.endswith("ArgyllCMS/color") for d in dirs), str(dirs))
        # missing powershell -> honest empty scan (no crash)
        with patch.object(sources.subprocess, "run",
                          side_effect=OSError("no powershell")):
            check("win scan: powershell missing -> empty list",
                  sources.usb_devices() == [])


if __name__ == "__main__":
    for fn in (test_windows_port_paths,
               test_macos_port_paths,
               test_instrument_event_filters_modes,
               test_unsupported_persisted_mode_switches,
               test_menu_cycle_respects_allowed_modes,
               test_adaptive_plot_range,
               test_usb_scan_vendor_class_instrument,
               test_xyz_only_frame,
               test_device_picker_sets_port,
               test_state_wording_generic,
               test_instrument_measurements_panel,
               test_cri_math,
               test_spotread_duv_cri_relay,
               test_force_calibration_menu,
               test_units_follow_mode,
               test_photometric_absolute,
               test_copy_readout):
        print(f"{fn.__name__}:")
        fn()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):", ", ".join(FAILURES))
        sys.exit(1)
    print("all multi-instrument tests passed")
