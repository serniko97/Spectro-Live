#!/usr/bin/env python3
"""HERMETIC integration test: pause() must stop the driver sending
reading triggers to spotread. Real SpotreadWorker thread against a
fake spotread child over a real PTY.

Reproduces the live complaint: 'paused after 3 readings and the
count kept going up' — if the driver keeps sending '\\n' after
pause(), the fake's key log proves it.
"""
import os
import queue
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import spectrov.sources as sources  # noqa: E402

KEYLOG = "/tmp/opencode/fake_spotread_keys.log"
FAILURES = []


def _wait_exit(w, timeout=5.0):
    """Wait for the worker thread to end — WITHOUT Thread.join(): on
    Python 3.9 (what a Mac commonly ships) join() touches the
    Thread._stop machinery that our worker's OWN self._stop Event
    attribute shadows, raising TypeError. A poll loop is versionproof
    and behaves identically."""
    t0 = time.monotonic()
    while w.is_alive() and time.monotonic() - t0 < timeout:
        time.sleep(0.05)


def check(name, ok, detail=""):
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}"
          + ("" if ok else f"  {detail}"))
    if not ok:
        FAILURES.append(name)


def wait_keys(n, timeout):
    """Wait until the fake received at least n keys."""
    events = []
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with open(KEYLOG) as f:
                events = [ln.split() for ln in f if "key:" in ln]
            if len(events) >= n:
                return events
        except OSError:
            pass
        time.sleep(0.2)
    return events


def main():
    if os.path.exists(KEYLOG):
        os.unlink(KEYLOG)

    w = sources.SpotreadWorker(queue.Queue())
    w._command = lambda mode=None, force=False: \
        [sys.executable, "tests/fake_spotread.py", KEYLOG]
    real_present = sources.instrument_present
    real_path = sources.spotread_path
    sources.instrument_present = lambda: "ColorMunki"
    sources.spotread_path = lambda: "/usr/bin/python"

    # drain worker events quietly
    def drain():
        while True:
            try:
                w.out.get(timeout=0.3)
            except queue.Empty:
                return

    try:
        w.start()
        time.sleep(2.0)          # session up, banner printed
        # recording ON -> driver sends \n within ~1-2 s
        w.start_continuous()
        evs = wait_keys(1, 5)
        check("recording ON: driver triggered a reading",
              len(evs) >= 1 and evs[0][1] == "key:NEWLINE", str(evs))
        # continuous -> a second reading after the menu returns
        evs2 = wait_keys(2, 8)
        check("continuous: second reading follows the menu",
              len(evs2) >= 2, str(evs2))

        # PAUSE — the moment from the live complaint
        w.pause()
        pause_at = time.time()
        time.sleep(12.0)         # long past the 5 s heartbeat mark
        with open(KEYLOG) as f:
            evs3 = [ln.split() for ln in f if "key:" in ln]
        after = [e for e in evs3 if float(e[0]) > pause_at + 1.0]
        check("PAUSED: NO further triggers reach spotread",
              after == [], str(after))
        check("PAUSED: flags are all clear",
              not (w.wants_read.is_set() or w.continuous.is_set()
                   or w._armed), "flags set")
        # SINGLE while paused must still work exactly once
        w.request_read()
        evs4 = wait_keys(len(evs3) + 1, 6)
        after = [e for e in evs4 if float(e[0]) > pause_at + 1.0]
        check("SINGLE while paused: exactly one trigger",
              len(after) == 1 and after[0][1] == "key:NEWLINE", str(after))
        t_single = float(after[-1][0])
        time.sleep(6.0)
        with open(KEYLOG) as f:
            evs5 = [ln.split() for ln in f if "key:" in ln]
        after2 = [e for e in evs5 if float(e[0]) > t_single + 2.0]
        check("after the single: silent again", after2 == [], str(after2))

        # RESUME -> continuous readings come back
        w.start_continuous()
        evs6 = wait_keys(len(evs5) + 1, 6)
        check("RESUME: readings continue", len(evs6) > len(evs5),
              str(evs6))
    finally:
        sources.instrument_present = real_present
        sources.spotread_path = real_path
        w.stop()
        _wait_exit(w)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):", ", ".join(FAILURES))
        sys.exit(1)
    print("pause-behaviour integration test passed")


def test_hardware_switch_tagging():
    """The fake emulates the ColorMunki's switch trigger: readings begin
    ~1.5 s after each menu with NO key from us. The driver must (a) send
    nothing while paused and (b) tag every such frame hw_triggered=True
    so the GUI can tell a button press from a held/stuck switch."""
    os.environ["FAKE_SELTRIG"] = "1.5"
    if os.path.exists(KEYLOG):
        os.unlink(KEYLOG)
    w = sources.SpotreadWorker(queue.Queue())
    w._command = lambda mode=None, force=False: \
        [sys.executable, "tests/fake_spotread.py", KEYLOG]
    real_present = sources.instrument_present
    real_path = sources.spotread_path
    sources.instrument_present = lambda: "ColorMunki"
    sources.spotread_path = lambda: "/usr/bin/python"

    def frames(timeout, min_n=1):
        """Collect delivered frames until min_n arrived or timeout."""
        out, t0 = [], time.time()
        while time.time() - t0 < timeout:
            try:
                ev = w.out.get(timeout=0.3)
            except queue.Empty:
                continue
            if ev[0] == "frame":
                out.append(ev[1])
                if len(out) >= min_n:
                    return out
        return out

    try:
        w.start()
        time.sleep(1.0)
        w.start_continuous()
        fr = frames(10, min_n=2)
        check("hw-emulation: app-requested frames delivered",
              len(fr) >= 2, str(len(fr)))
        check("app-requested frames are NOT tagged hw",
              all(not getattr(f, "hw_triggered", False) for f in fr[:2]),
              str([getattr(f, "hw_triggered", None) for f in fr]))
        # PAUSE: the emulated switch keeps triggering on its own
        w.pause()
        keys_before = _count_keys()
        hw = frames(25, min_n=4)
        check("hw-emulation: switch-triggered frames delivered",
              len(hw) >= 4, str(len(hw)))
        # the FIRST frame after pause may still be the one in flight
        # (triggered by us a moment before the pause) — everything the
        # emulated switch started afterwards must be tagged hw
        check("switch-triggered frames ARE tagged hw",
              all(getattr(f, "hw_triggered", False) for f in hw[1:]),
              str([getattr(f, "hw_triggered", None) for f in hw]))
        check("driver sent NO keys while paused",
              _count_keys() == keys_before,
              f"{keys_before} -> {_count_keys()}")
    finally:
        del os.environ["FAKE_SELTRIG"]
        sources.instrument_present = real_present
        sources.spotread_path = real_path
        w.stop()
        _wait_exit(w)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):", ", ".join(FAILURES))
        sys.exit(1)
    print("hardware-switch tagging test passed")


def _count_keys():
    try:
        with open(KEYLOG) as f:
            return sum(1 for ln in f if "key:" in ln)
    except OSError:
        return 0


if __name__ == "__main__":
    main()
    test_hardware_switch_tagging()
