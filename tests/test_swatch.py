"""GUI-level sRGB swatch tests (headless): the mouse-visible result of
the _on_frame -> _refresh -> swatch pipeline for the new reflective
"brightness = reflectance" behaviour.

Run:  QT_QPA_PLATFORM=offscreen .venv/bin/python tests/test_swatch.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threading                                    # noqa: E402
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


def make_window(mode):
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
    win.mode_name = mode
    win.show()
    return win


def feed(win, xyz):
    wl = np.linspace(380, 730, 106)
    win.frames.clear()
    win._on_frame(Spectrum(wl, np.abs(np.sin(wl)) * 0.3, xyz))
    QApplication.processEvents()
    ss = win.swatch.styleSheet()
    bg = None
    for part in ss.split(";"):
        part = part.strip()
        # scoped rules wrap the CSS (#swatch { ... }) — match anywhere
        # in the part, not just at the start
        if "background:" in part and "rgb(" in part:
            bg = tuple(int(v) for v in
                       part.split("rgb(")[1].split(")")[0].split(","))
    txt = win.lbl_srgb_vals.text().strip("()").split(", ")
    vals = tuple(int(v) for v in txt)
    red = "color: " + Q.HOT in win.lbl_srgb_vals.styleSheet()
    return bg, vals, red


def srgbvals(win):
    txt = win.lbl_srgb_vals.text().strip("()").split(", ")
    try:
        return tuple(int(v) for v in txt)
    except ValueError:
        return None


FAILURES = []


def check(name, cond, extra=""):
    s = "ok  " if cond else "FAIL"
    print(f"  [{s}] {name}{('  ' + extra) if extra and not cond else ''}")
    if not cond:
        FAILURES.append(name)


def main():
    app = QApplication.instance() or QApplication(["t"])
    app.setStyle(Q.QSS)
    app.processEvents()

    # --- Reflective: brightness == reflectance ------------------------
    win = make_window("Reflective")
    bg, vals, red = feed(win, (0.9642, 1.0, 0.8251))   # 100 % white tile
    check("reflective 100 % white tile -> rgb(255,255,255)",
          bg == (255, 255, 255), str(bg))
    check("label shows the 8-bit values in parentheses",
          vals == (255, 255, 255), str(vals))
    check("label not red inside sRGB", not red)
    bg, vals, red = feed(win, tuple(v * 0.5 for v in (0.9642, 1.0, 0.8251)))
    check("reflective 50 % grey -> half-light codes (~188)",
          bg is not None and 185 <= bg[0] <= 191
          and bg[0] == bg[1] == bg[2], str(bg))
    check("label follows grey (188,188,188)",
          vals == bg, f"{vals} vs {bg}")
    bg, vals, red = feed(win, (0.041626, 0.035747, 0.011519))  # the live wall
    check("reflective dark wall (Y=3.6%) -> honest rgb(73,47,27)",
          bg == (73, 47, 27), str(bg))
    check("label shows the wall values", vals == (73, 47, 27), str(vals))
    bg, vals, red = feed(win, (2.4, 1.5, 0.3))          # would exceed 255
    check("reflective out-of-range -> boosted, max channel 255",
          bg is not None and max(bg) == 255
          and bg.count(255) == 1, str(bg))
    check("in-gamut chromaticity: label NOT red (boost is luminance only)",
          not red)
    # genuinely outside the sRGB gamut: laser-red chromaticity
    bg, vals, red = feed(win, (0.73, 0.27, 0.10))
    check("out-of-sRGB chromaticity: label drawn in red", red,
          f"bg={bg}")
    win.worker.stop()

    # --- Emissive/Ambient: still max brightness ------------------------
    win = make_window("Emissive")
    bg, vals, red = feed(win, (0.041626, 0.035747, 0.011519))
    check("emissive dark light -> chromaticity at full brightness "
          "rgb(255,173,113)", bg == (255, 173, 113), str(bg))
    check("emissive label matches swatch", vals == bg, f"{vals} vs {bg}")
    bg, vals, red = feed(win, (2.4, 1.5, 0.3))
    check("emissive bright -> boosted, max channel 255",
          bg is not None and max(bg) == 255, str(bg))
    check("emissive bright-but-in-gamut: label NOT red", not red)
    bg, vals, red = feed(win, (0.73, 0.27, 0.10))
    check("emissive out-of-gamut label red", red, f"bg={bg}")
    win.worker.stop()

    win = make_window("Ambient")
    bg, vals, red = feed(win, (0.041626, 0.035747, 0.011519))
    check("ambient dim light -> full-brightness rgb(255,173,113)",
          bg == (255, 173, 113), str(bg))
    # too dark to evaluate: clean placeholder, no misleading colour
    bg, vals, red = feed(win, (0.0, 0.0, 0.0))
    check("too-dark reading: label resets to placeholder",
          win.lbl_srgb_vals.text() == "—" or bg == (0, 0, 0),
          win.lbl_srgb_vals.text())
    win.worker.stop()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):", ", ".join(FAILURES))
        sys.exit(1)
    print("all swatch tests passed")


if __name__ == "__main__":
    main()
