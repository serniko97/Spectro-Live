"""Annunciator-lamp state machine tests (headless, QT offscreen).

Run:  QT_QPA_PLATFORM=offscreen .venv/bin/python tests/test_lamps.py

The lamp rules under test (all lamps are SOLID when active — no
flashing anywhere, by design):
  CALIBRATION      lit = calibration needed or running, off otherwise;
                    green border while spotread's calibration is valid.
  SURFACE/AMBIENT  lit = the current wheel slot has NOT yet been
  POSITION              confirmed by a successful reading.
                    Never lit while the CAL lamp is active.
"""
import os
import sys
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                      # noqa: E402
trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy 1.x on macOS has no trapezoid()
from PySide6.QtWidgets import QApplication  # noqa: E402

import spectrov.qtapp as Q              # noqa: E402
import spectrov.sources as sources      # noqa: E402
from spectrov.processing import Spectrum  # noqa: E402


class FakeWorker:
    """Stands in for SpotreadWorker: no device, no subprocess."""

    def __init__(self):
        self.continuous = threading.Event()
        self.wants_read = threading.Event()
        self.recal_requested = False

    def start(self): pass
    def stop(self): pass
    def is_alive(self): return True
    def pause(self): self.continuous.clear()
    def resume(self): pass
    def request_read(self): self.wants_read.set()
    def start_continuous(self): self.continuous.set()
    def arm_probes(self): pass
    def request_calibration(self): self.recal_requested = True


CAL_BLINK1 = "Turn the wheel to position 2 (CALIBRATION)…"
CAL_RUN = "Calibrating… keep the wheel at position 2."
CAL_DONE_R = ("Calibrated ✓ — turn the wheel to the SURFACE position and "
              "hold the aperture flat against the surface…")
CAL_DONE_E = ("Calibrated ✓ — move the wheel to position 3 (MEASURING) "
              "and point at the target…")
WP_GUIDE = "Move the wheel: sensor should be in the surface position…"
MISREAD = "Reading failed (misread (Reading is inconsistent)) — retrying…"


def make_window(mode="Reflective"):
    old_mk = Q.MainWindow._make_worker
    old_present = sources.instrument_present
    old_home = os.environ.get("HOME")
    Q.MainWindow._make_worker = lambda self: FakeWorker()
    sources.instrument_present = lambda: None
    # hermetic: the test's diag/CSV logs go to a scratch HOME
    os.environ["HOME"] = "/tmp/opencode/hermetic-home"
    os.makedirs("/tmp/opencode/hermetic-home", exist_ok=True)
    try:
        app = QApplication.instance() or QApplication(["t"])
        app.setStyle("Fusion")
        win = Q.MainWindow()
    finally:
        Q.MainWindow._make_worker = old_mk
        sources.instrument_present = old_present
        if old_home is not None:
            os.environ["HOME"] = old_home
    win.mode_name = mode
    # the mode changed after launch auto-recording: resync lamps/caption
    win._set_lamps()
    win._update_position_lamp()
    win.show()
    return win


def frame(win):
    wl = np.linspace(380, 730, 106)
    win._on_frame(Spectrum(wl, np.abs(np.sin(wl)) * 0.3, (0.2, 0.15, 0.05)))


def pause(win):
    win.running = True
    win.on_start_stop()          # toggles to PAUSE
    # debounce guard in on_start_stop: force the toggle to be honoured
    if win.running:
        win._last_toggle = 0.0
        win.on_start_stop()


FAILURES = []


def check(name, cond, extra=""):
    status = "ok  " if cond else "FAIL"
    print(f"  [{status}] {name}{('  ' + extra) if extra and not cond else ''}")
    if not cond:
        FAILURES.append(name)


def test_launch_in_reflective():
    win = make_window("Reflective")     # launch == continuous recording
    check("launch: continuous pending, unconfirmed -> SURFACE blinks",
          win._update_position_lamp() == "solid",
          str(win._lamp_state))
    # calibration takes over: CAL blinks, SURFACE must not show
    win._on_state("calibrating", CAL_BLINK1)
    check("calibration prompt: CAL blinks",
          win._lamp_state["cal"] == "solid")
    check("calibration prompt: SURFACE suppressed while CAL active",
          win._lamp_state["reflective"] == "off")
    win._on_state("calibrating", CAL_RUN)
    check("calibration running: CAL solid",
          win._lamp_state["cal"] == "solid")
    win._on_state("measuring", CAL_DONE_R)
    check("after calibration: SURFACE blinks again (measurement pending)",
          win._lamp_state["reflective"] == "solid")
    frame(win)
    check("first successful reading: lamps all off",
          all(v == "off" for v in win._lamp_state.values()),
          str(win._lamp_state))


def test_reconnect_keeps_confirmed_slot():
    """User report: while stopped in single mode with the wheel already
    in the right slot, a USB drop flashed the SURFACE lamp as if the
    wheel had moved. A USB blip cannot turn the wheel — the confirmed
    slot must stay confirmed; only a real wrong-position failure may
    re-arm the prompt."""
    win = make_window("Emissive")
    win._on_state("measuring", CAL_DONE_E)
    frame(win)                          # slot confirmed by a reading
    check("confirmed slot: lamp off",
          win._lamp_state["surface"] == "off", str(win._lamp_state))
    pause(win)                          # single mode, app stopped
    # USB drop -> restart WITHOUT any calibration dance
    win._on_state("restart", "Instrument stopped responding — restarting")
    win._on_state("starting", "Session with ColorMunki Photo")
    win._on_state("measuring", "")
    win._update_position_lamp()
    check("USB drop while stopped: NO prompt (slot still confirmed)",
          win._lamp_state["surface"] == "off", str(win._lamp_state))
    # user really did move the wheel while unplugged -> next reading says so
    win._on_state("guide", "Move the wheel: sensor should be in the "
                   "measuring position…")
    check("genuine wrong position: lamp prompts again",
          win._lamp_state["surface"] == "solid", str(win._lamp_state))
    frame(win)
    check("confirmed again after a reading",
          win._lamp_state["surface"] == "off", str(win._lamp_state))
    win.close()


def test_reconnect_while_stopped():
    """Reconnect calibration while stopped: nothing may auto-measure, and
    the lamp must PROMPT for the measuring slot again (the wheel sits at
    the calibration slot after the dance; recordings and single/button
    disciplines alike need it back at the measuring slot)."""
    # launch == continuous recording: unconfirmed slot blinks until the
    # first reading confirms it
    win = make_window("Reflective")
    check("paused at launch, slot unconfirmed: SURFACE blinks (prompt)",
          win._lamp_state["reflective"] == "solid", str(win._lamp_state))
    pause(win)                          # user stopped the recording
    check("paused: no CALIBRATION lamp",
          win._lamp_state["cal"] == "off", str(win._lamp_state))
    # USB blip -> session restart -> re-calibration
    win._on_state("restart", "Instrument stopped responding — restarting")
    win._on_state("starting", "Session with ColorMunki Photo")
    win._on_state("calibrating", CAL_BLINK1)
    win._on_state("calibrating", CAL_RUN)
    win._on_state("measuring", CAL_DONE_R)   # stage-3 message arrives...
    check("reconnect while stopped: CALIBRATION lamp off",
          win._lamp_state["cal"] == "off")
    check("reconnect while stopped: SURFACE prompts again",
          win._lamp_state["reflective"] == "solid",
          str(win._lamp_state))
    check("status bar offers next step",
          "surface" in win.lbl_message.text().lower())
    # the session must NOT auto-measure while stopped
    check("stopped: no reading requested",
          not win.worker.wants_read.is_set()
          and not win.worker.continuous.is_set())


def test_start_after_confirmed_position():
    """Pause -> resume at the same wheel slot must not re-flash."""
    win = make_window("Reflective")
    win._on_state("measuring", CAL_DONE_R)
    frame(win)                          # position confirmed
    check("confirmed by reading: lamp off",
          win._lamp_state["reflective"] == "off")
    pause(win)
    win.running = False
    # resume: START pressed while position already confirmed
    # (on_start_stop sets running=True before _begin_recording)
    win.running = True
    win._begin_recording()
    check("resume with confirmed slot: lamp stays off",
          win._lamp_state["reflective"] == "off", str(win._lamp_state))
    # ...but if the wheel then becomes wrong, a guide re-lights it
    win._on_state("guide", WP_GUIDE)
    check("wrong-position guide while recording: lamp blinks",
          win._lamp_state["reflective"] == "solid")
    frame(win)
    check("recovery: lamp off again",
          win._lamp_state["reflective"] == "off")


def test_mode_change_prompts_new_slot():
    """Only a PHYSICAL slot change voids the wheel confirmation:
    Emissive and Reflective both read through the front aperture, so
    switching between them must NOT re-prompt (user report: USB drop
    during Emissive -> Reflective kept flashing the lamp though the
    wheel never moved). Any switch involving the ambient diffuser does
    prompt for the new slot."""
    win = make_window("Reflective")
    frame(win)                          # reflective slot confirmed
    check("confirmed slot: lamp off",
          win._lamp_state["reflective"] == "off", str(win._lamp_state))

    # Reflective -> Emissive: same aperture, confirmation carries over
    win._on_mode_changed("Emissive")
    check("Reflective -> Emissive: same slot, NO prompt",
          win._lamp_state["reflective"] == "off", str(win._lamp_state))
    # a USB drop during the switch keeps the confirmation too
    win._on_state("restart", "Instrument stopped responding — restarting")
    win._on_state("starting", "Session with ColorMunki Photo")
    check("USB drop during mode switch: still no prompt",
          win._lamp_state["reflective"] == "off", str(win._lamp_state))

    win._on_mode_changed("Emissive")   # no-op (guard branch)
    win._on_mode_changed("Reflective")  # and back — still the same slot
    check("Emissive -> Reflective: same slot, NO prompt",
          win._lamp_state["surface"] == "off"
          and win._lamp_state["reflective"] == "off", str(win._lamp_state))

    # any switch involving the ambient diffuser is a different slot
    win._on_mode_changed("Ambient")
    check("switch to Ambient: AMBIENT lamp prompts the diffuser slot",
          win._lamp_state["ambient"] == "solid", str(win._lamp_state))
    check("switch to Ambient: SURFACE lamp silent",
          win._lamp_state["reflective"] == "off", str(win._lamp_state))
    frame(win)                          # reading in the ambient slot
    check("ambient reading confirms the new slot",
          win._lamp_state["ambient"] == "off", str(win._lamp_state))
    win._on_mode_changed("Emissive")     # back to the front aperture
    check("Ambient -> Emissive: SURFACE lamp prompts (different slot)",
          win._lamp_state["surface"] == "solid"
          and win._lamp_state["reflective"] == "off"
          and win._lamp_state["ambient"] == "off", str(win._lamp_state))
    win.close()


def test_single_reading():
    win = make_window("Reflective")
    pause(win)                          # continuous off: SINGLE only
    win._on_state("calibrating", CAL_RUN)
    win._on_state("measuring", CAL_DONE_R)
    # even with NO measurement in flight, the wheel sits at the
    # calibration slot after the dance: the lamp must prompt for the
    # measuring slot so a single click (or instrument-button press)
    # doesn't silently read the wrong slot
    check("idle at menu after cal: position lamp prompts",
          win._lamp_state["reflective"] == "solid", str(win._lamp_state))
    win.worker.request_read = win.worker.request_read
    win.on_single()                      # single read in flight
    check("SINGLE in flight, position unconfirmed -> blink",
          win._lamp_state["reflective"] == "solid")
    frame(win)
    check("single delivered: lamp off",
          win._lamp_state["reflective"] == "off")


def test_misread_keeps_lamp_semantics():
    """Misread is not a position fault: lamp must not turn on if position
    was confirmed, must stay on if it wasn't."""
    win = make_window("Reflective")
    win._on_state("measuring", CAL_DONE_R)
    frame(win)
    win._on_state("guide", MISREAD)
    check("misread after confirmed reading: lamp off",
          win._lamp_state["reflective"] == "off")
    # unconfirmed + pending + misread: keep blinking (still probing)
    win._on_state("calibrating", CAL_RUN)     # position unknown again
    win._on_state("measuring", CAL_DONE_R)
    win.running = True
    win._update_position_lamp()
    win._on_state("guide", MISREAD)
    check("misread while unconfirmed+pending: lamp keeps blinking",
          win._lamp_state["reflective"] == "solid", str(win._lamp_state))


def test_ambient_mode():
    win = make_window("Ambient")
    win.running = True
    check("ambient: AMBIENT lamp blinks (unconfirmed)",
          win._lamp_state["ambient"] == "solid", str(win._lamp_state))
    check("ambient: SURFACE lamp stays off",
          win._lamp_state["surface"] == "off")
    frame(win)
    check("ambient confirmed: off", win._lamp_state["ambient"] == "off")


def test_paused_button_frames_captured():
    """When the user clicks STOP, the plot is authoritative: a frame
    arriving within the short grace window (the in-flight reading of
    the run just stopped) is kept, anything later — a straggler, a
    physical button press, a held/defective switch re-triggering
    spotread — is DROPPED so the frozen trace never moves. A ⏺ SINGLE
    in flight while stopped is never dropped (deliberate measurement)."""
    win = make_window("Reflective")
    frame(win)
    n = win.frame_count
    pause(win)
    check("stopped: running is off", win.running is False)

    def hwframe(tag):
        spec = Spectrum(np.linspace(380, 730, 106),
                        np.abs(np.sin(np.linspace(380, 730, 106))) * 0.3,
                        (0.2, 0.15, 0.05))
        spec.hw_triggered = True
        win._on_frame(spec)
        return tag

    # in-flight reading of the just-stopped run (within the grace
    # window) -> kept
    frame(win)
    check("stopped: in-flight frame right after STOP kept",
          win.frame_count == n + 1, f"{win.frame_count}")
    # later stragglers -> dropped, whether button presses or streams
    win._stop_since = time.monotonic() - 10.0
    hwframe("press-late")
    hwframe("held2")
    hwframe("held3")
    check("stopped: late button frames dropped (isolated or held)",
          win.frame_count == n + 1, f"{win.frame_count}")
    # in-flight SINGLE: the one deliberate measurement -> accepted
    win.btn_single.setEnabled(False)
    frame(win)
    check("stopped: in-flight SINGLE frame accepted",
          win.frame_count == n + 2, f"{win.frame_count}")
    win.btn_single.setEnabled(True)
    # and once recording resumes, frames flow again
    win.running = True
    frame(win)
    check("resumed: frames accepted again",
          win.frame_count == n + 3, f"{win.frame_count}")
    win.close()


def test_startup_without_calibration_lamps():
    """With -N (reusing Argyll's saved calibration) a session may start
    with NO calibration dance at all: the lamps must go straight from
    idle to the normal pending-measurement state — no stale CALIBRATION
    blink, no stuck lamps. Covers startup, USB reconnect and mode
    switch, which all follow the same path."""
    win = make_window("Reflective")
    # no calibrating events whatsoever — straight to the reading guide
    win._on_state("starting", "Session with ColorMunki Photo")
    win._on_state("guide", "Place instrument on spot to be measured")
    check("no-cal startup: CALIBRATION lamp never lit",
          win._lamp_state["cal"] == "off", str(win._lamp_state))
    # a pending measurement with the slot unconfirmed -> position blink
    win.running = True
    win._pos_confirmed = False
    win._on_state("measuring", "Reading taken")
    win._update_position_lamp()
    check("no-cal startup: SURFACE blinks once a reading is pending",
          win._lamp_state["reflective"] == "solid", str(win._lamp_state))
    # first reading confirms the slot -> lamp goes out on its own
    win._pos_confirmed = True
    win._update_position_lamp()
    check("no-cal startup: lamp out after first confirmed reading",
          win._lamp_state["reflective"] == "off", str(win._lamp_state))
    # and a late genuine calibration demand still flashes CAL normally
    win._on_state("calibrating",
                  "Calibrated ✓ — turn the wheel to position 2")
    check("late cal demand: CALIBRATION blinks again",
          win._lamp_state["cal"] == "solid", str(win._lamp_state))
    win.close()


def test_emissive_blinks_surface_lamp():
    """Emissive reads through the SAME front aperture (wheel position 3)
    as reflective — so after calibration, with the slot unconfirmed, the
    SURFACE POSITION lamp must be lit there too (user-observed: it used
    to stay dark in emissive only)."""
    win = make_window("Emissive")
    win.running = True
    win._on_state("measuring",
                  "Calibrated ✓ — move the wheel to position 3 (MEASURING) "
                  "and point at the target…")
    check("emissive: SURFACE POSITION lamp lit while unconfirmed",
          win._lamp_state["surface"] == "solid"
          and win._lamp_state["ambient"] == "off",
          str(win._lamp_state))
    frame(win)                          # successful reading: confirmed
    check("emissive: confirmed by reading, lamp off",
          win._lamp_state["surface"] == "off",
          str(win._lamp_state))


def test_cal_lamp_click_requests_recalibration():
    """The CALIBRATION lamp is clickable: a click asks the worker to
    recalibrate NOW (one-shot -N omission), invalidates the green
    border immediately, and reports the wheel-at-2 instruction. With
    no live session it just tells the user there is nothing to do."""
    win = make_window("Reflective")
    win._mark_cal_valid()
    check("pre: green cal validity window armed",
          win._cal_ok_until > time.monotonic())
    win._on_cal_lamp_clicked()
    w = win.worker
    check("click: worker asked to recalibrate",
          w.recal_requested is True)
    check("click: green border invalidated immediately",
          win._cal_ok_until == 0.0, str(win._cal_ok_until))
    # no live session -> friendly refusal, no crash
    w.recal_requested = False
    w.is_alive = lambda: False
    win._on_cal_lamp_clicked()
    check("no live session: nothing requested",
          w.recal_requested is False)
    win.close()


def test_cal_lamp_progress_bar():
    """During the all-modes calibration pass the CALIBRATION lamp doubles
    as a progress bar: amber fill 25 % at stage 1/3, 50 % at 2/3, 75 % at
    3/3 — and when the pass confirms validity the fill is gone, the lamp
    dark, and the green border ("100 % ready") is armed. A plain
    single-mode dance shows no fill."""
def test_cal_lamp_progress_bar():
    """During the all-modes calibration pass the CALIBRATION lamp fills
    a quarter per completed-and-started mode — directly from the stage
    headers (1/3 -> 25 %, 2/3 -> 50 %, 3/3 -> 75 %, no solid-yellow
    blink between stages: the worker auto-answers each wheel prompt
    seconds after the header, so a blink would be noise). When the
    pass confirms validity the fill is gone, the lamp dark and the
    green border ("100 % ready") armed. A plain single-mode dance
    shows a solid lamp throughout."""
    win = make_window("Reflective")
    for i, frac in ((1, 0.25), (2, 0.5), (3, 0.75)):
        # the pass header starts the fill right away
        win._on_state("calibrating",
                      f"Calibrating every reading mode — Emissive "
                      f"({i}/3): keep the wheel at the CALIBRATION "
                      "position…")
        check(f"stage {i}/3 header: quarter fill {int(frac*100)} % starts",
              win._cal_progress == frac, str(win._cal_progress))
        ss = win.lamps["cal"].styleSheet()
        check(f"stage {i}/3 header: gradient fill painted",
              "qlineargradient" in ss and f"stop:{frac:.4f}" in ss,
              ss[:90])
        # ...spotread confirming the running dance keeps the same fill
        win._on_state("calibrating",
                      "Calibrating… keep the wheel at position 2.")
        check(f"stage {i}/3 running: fill steady (no blink back)",
              win._cal_progress == frac, str(win._cal_progress))
    win._on_state("measuring",
                  "Calibrated ✓ — turn the wheel to the SURFACE position "
                  "and hold the aperture flat against the surface…")
    check("pass done: progress cleared, lamp back to normal",
          win._cal_progress is None)
    check("pass done: green border armed (100 % ready)",
          win._cal_ok_until > time.monotonic())
    win._on_state("calibrating", "Calibrating… keep the wheel at position 2.")
    check("single-mode dance: no progress-bar fill",
          win._cal_progress is None)
    ss = win.lamps["cal"].styleSheet()
    check("single-mode dance: plain solid lamp",
          "qlineargradient" not in ss, ss[:90])
    win.close()


def test_mode_lamps_select_mode():
    """The SURFACE / REFLECTIVE / AMBIENT lamps ARE the mode selector:
    click selects the mode; the selected one gets the blue border and
    lights amber only for itself while its position is unconfirmed;
    every lamp has a hover highlight (they are buttons). Unsupported
    modes are refused with a status message."""
    win = make_window("Reflective")
    ss = {k: win.lamps[k].styleSheet() for k in win._LAMP_MODES}
    check("pre: selected REFLECTIVE lamp has the blue border",
          Q.ACCENT in ss["reflective"] and Q.ACCENT not in ss["surface"]
          and Q.ACCENT not in ss["ambient"])
    check("pre: every mode lamp shows a hover highlight (button-like)",
          all("LampLabel:hover" in s for s in ss.values()))
    win.lamps["ambient"].clicked.emit()      # click the AMBIENT lamp
    check("click AMBIENT lamp -> mode is Ambient",
          win.mode_name == "Ambient", win.mode_name)
    check("selected moved: blue border follows Ambient",
          Q.ACCENT in win.lamps["ambient"].styleSheet()
          and Q.ACCENT not in win.lamps["reflective"].styleSheet())
    check("unconfirmed slot: ONLY the selected lamp is amber",
          win._lamp_state["ambient"] == "solid"
          and win._lamp_state["surface"] == "off"
          and win._lamp_state["reflective"] == "off",
          str(win._lamp_state))
    frame(win)                                # confirm the diffuser slot
    check("reading confirms the slot: amber goes away",
          win._lamp_state["ambient"] == "off")
    # a mode the instrument cannot do is refused
    win.allowed_modes = ("Emissive",)
    win.lamps["reflective"].clicked.emit()
    check("unsupported mode refused: selection unchanged",
          win.mode_name == "Ambient", win.mode_name)
    win.close()


if __name__ == "__main__":
    for fn in (test_launch_in_reflective,
               test_reconnect_keeps_confirmed_slot,
               test_reconnect_while_stopped,
               test_start_after_confirmed_position,
               test_single_reading,
               test_mode_change_prompts_new_slot,
               test_misread_keeps_lamp_semantics,
               test_paused_button_frames_captured,
               test_startup_without_calibration_lamps,
               test_ambient_mode,
               test_emissive_blinks_surface_lamp,
               test_cal_lamp_click_requests_recalibration,
               test_cal_lamp_progress_bar,
               test_mode_lamps_select_mode):
        print(f"{fn.__name__}:")
        fn()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):", ", ".join(FAILURES))
        sys.exit(1)
    print("all lamp-state tests passed")
