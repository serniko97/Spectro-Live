"""Window-width regression test (headless).

The window used to grow wider by itself when a long dynamic text
(caption, device string, rate) arrived. Reproduce + guard.

Run:  QT_QPA_PLATFORM=offscreen .venv/bin/python tests/test_layout.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication      # noqa: E402

import spectrov.qtapp as Q                     # noqa: E402
import spectrov.sources as sources            # noqa: E402


class FakeWorker:
    """Stands in for SpotreadWorker: NO real device, NO logs."""
    continuous = None
    wants_read = None

    def __init__(self):
        import threading
        self.continuous = threading.Event()
        self.wants_read = threading.Event()

    def start(self): pass
    def stop(self): pass
    def is_alive(self): return True
    def pause(self): pass
    def resume(self): pass
    def request_read(self): pass
    def start_continuous(self): pass
    def arm_probes(self): pass


LONG_TEXT = ("This is an extremely long caption text that would "
             "otherwise push the window much wider — " + "x" * 160)

FAILURES = []


def check(name, cond, extra=""):
    s = "ok  " if cond else "FAIL"
    print(f"  [{s}] {name}{('  ' + extra) if extra and not cond else ''}")
    if not cond:
        FAILURES.append(name)


def main():
    # HERMETIC: never touch the real instrument, the user's logs, or
    # the live app. Fake worker + throwaway HOME for the whole test.
    old_mk = Q.MainWindow._make_worker
    old_home = os.environ.get("HOME")
    Q.MainWindow._make_worker = lambda self: FakeWorker()
    os.environ["HOME"] = "/tmp/opencode/hermetic-home"
    os.makedirs("/tmp/opencode/hermetic-home", exist_ok=True)
    app = QApplication.instance() or QApplication(["t"])
    app.setStyle("Fusion")
    app.setStyleSheet(Q.QSS)
    win = Q.MainWindow()
    win.resize(1280, 820)
    win.show()
    app.processEvents()

    base = win.minimumSizeHint().width()

    # toolbar button texts change on their own during a session
    for txt in ("▶  START", "⏸  PAUSE", "…  READING", "⏺  SINGLE",
                "Zero (A)", "Load B…", "STOP", "SINGLE"):
        win.btn_start.setText(txt)
        win.btn_single.setText(txt)
        app.processEvents()
    grown_tl = win.minimumSizeHint().width()

    # every dynamic text pushed to a worst-case length (captions now go
    # to the status bar only)
    win._set_caption(LONG_TEXT)
    win.status(LONG_TEXT)
    win.lbl_device.setText("Instrument: " + LONG_TEXT)
    win.lbl_rate.setText("1234 readings · average of N=16 · " + LONG_TEXT)
    win.val_colour.setText(LONG_TEXT)
    app.processEvents()

    grown = win.minimumSizeHint().width()

    check(f"minimum width stable ({base} -> {grown} / toolbar {grown_tl})",
          grown <= base + 8 and grown_tl <= base + 8,
          f"grew by {grown - base}px")
    check("window size unchanged", win.width() <= 1280 + 8)
    win.worker.stop()

    # long log-dialog text must not affect the main window either
    win.status("long log line: " + "y" * 300)
    app.processEvents()
    check("long status message harmless",
          win.minimumSizeHint().width() <= base + 8,
          str(win.minimumSizeHint().width()))

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):", ", ".join(FAILURES))
        sys.exit(1)
    print("all layout tests passed")

    # restore the real environment for anything launched afterwards
    Q.MainWindow._make_worker = old_mk
    if old_home is not None:
        os.environ["HOME"] = old_home


if __name__ == "__main__":
    main()
