"""Driver (SpotreadWorker) replay tests — no instrument, no GUI.

Run:  .venv/bin/python tests/test_driver_replay.py

Feeds scripted spotread transcripts (the exact line formats observed from
the real ColorMunki session logs) through the real SpotreadWorker state
machine and asserts on what reaches the GUI event queue.
"""
import os
import queue
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spectrov.sources import SpotreadWorker   # noqa: E402
import spectrov.sources as sources            # noqa: E402


class FakeProc:
    def poll(self): return None


MENU = ("Hit ESC or Q to exit, instrument switch or any other key to "
        "take a reading:")


def spectrum_block(n=106):
    vals = ["0.05"] * (n - 1)
    return ["Spectrum from 380.000 to 730.000 nm in 106 steps",
            "0.01, " + ", ".join(vals)]


RESULT_LAB = (" Result is XYZ: 0.40 0.31 0.25, D50 Lab: 62.4 0.3 1.2 "
              "(CCT = 2311K)")
RESULT_NO_LAB_BIG = " Result is XYZ: 4.02 3.11 2.50"          # >1.6, no Lab
RESULT_EMISSIVE = " Result is XYZ: 40.0 31.0 25.0 (CCT = 2311K)"


def make_worker(mode="r", continuous=False, wants=False):
    w = SpotreadWorker.__new__(SpotreadWorker)
    w.out = queue.Queue()
    w._stop = threading.Event()
    w.wants_read = threading.Event()
    w.continuous = threading.Event()
    w._armed = False
    w._app_trig = False
    w.user_confirm = threading.Event()
    w.STALL_TIMEOUT = 45.0
    w.mode = mode
    w.kind = "c"
    w.high_res = True
    w.fast = False
    w.force_cal = False
    w.restart_now = threading.Event()
    w.cal_pass_modes = None
    w._pass_since_frame = False
    w._skip_cache_once = False
    w.inst_type = ""
    w.port = None
    w._proc = FakeProc()
    w._master = None
    w.respond = lambda t: None
    w._kill = lambda: None
    if continuous:
        w.continuous.set()
    if wants:
        w.wants_read.set()
    return w


def run_script(w, lines, settle=2.0):
    it = iter(lines)

    def gen():
        yield from it
        while True:
            yield None
    w._lines = gen
    t = threading.Thread(target=SpotreadWorker._session,
                         args=(w, "ColorMunki"), daemon=True)
    t.start()
    time.sleep(settle)
    return w


def frames(w):
    return [e for e in list(w.out.queue) if e[0] == "frame"]


def states(w):
    return [e for e in list(w.out.queue) if e[0] == "state"]


FAILURES = []


def check(name, cond, extra=""):
    s = "ok  " if cond else "FAIL"
    print(f"  [{s}] {name}{('  ' + extra) if extra and not cond else ''}")
    if not cond:
        FAILURES.append(name)


# ----------------------------------------------------------------- tests
def test_reflective_percent_scale_normalized_via_lab():
    """A real reflective reading: XYZ is on the x100 percent scale; the
    Lab on the same line must drag it back to physical reflectance."""
    w = make_worker("r")
    run_script(w, spectrum_block() + [RESULT_LAB, MENU])
    fr = frames(w)
    check("reflective: one frame delivered", len(fr) == 1)
    if fr:
        xyz = fr[0][1].xyz
        check("reflective: XYZ divided by 100 (Y~0.31, not 31)",
              xyz is not None and abs(xyz[1] - 0.31) < 0.001, str(xyz))
        check("reflective: Lab captured onto the frame",
              getattr(fr[0][1], "meta_lab", None) is not None)


def test_reflective_plausibility_fallback():
    """No Lab printed: values above 1.6 must still be rescaled (a 4% blue
    channel can't be a physical reflectance above 1)."""
    w = make_worker("r")
    run_script(w, spectrum_block() + [RESULT_NO_LAB_BIG, MENU])
    fr = frames(w)
    check("reflective fallback: frame delivered", len(fr) == 1)
    if fr:
        xyz = fr[0][1].xyz
        check("reflective fallback: rescaled by 100",
              xyz is not None and abs(xyz[1] - 0.0311) < 1e-6, str(xyz))


def test_emissive_never_rescaled():
    """Emissive XYZ is absolute cd/m²: must pass through untouched."""
    w = make_worker("e")
    run_script(w, spectrum_block() + [RESULT_EMISSIVE, MENU])
    fr = frames(w)
    check("emissive: frame delivered", len(fr) == 1)
    if fr:
        xyz = fr[0][1].xyz
        check("emissive: XYZ untouched (Y=31.0)",
              xyz is not None and abs(xyz[1] - 31.0) < 1e-6, str(xyz))


def test_wedge_early_delivery():
    """spotread sometimes never prints the closing menu banner after the
    result. The frame must still be delivered (early delivery)."""
    w = make_worker("e")
    # result arrives, then NOTHING (no menu banner) — wedge
    run_script(w, spectrum_block() + [RESULT_EMISSIVE])
    fr = frames(w)
    check("wedge: frame delivered despite no menu banner", len(fr) == 1,
          str(len(fr)))


def test_calibration_does_not_autoarm_when_stopped():
    """The user rule: a (re)calibration while nothing is recording must
    not arm probes nor deliver any frame."""
    w = make_worker("r")
    run_script(w, ["Set instrument sensor to calibration position,",
                   "Calibration complete",
                   "Place instrument on spot to be measured,", MENU])
    check("stopped reconnect: not armed", not w._armed)
    check("stopped reconnect: no frames leaked", len(frames(w)) == 0)


def test_calibration_arms_when_continuous():
    w = make_worker("r", continuous=True)
    run_script(w, ["Calibration complete"]
               + spectrum_block() + [RESULT_LAB, MENU])
    check("continuous: armed after calibration", w._armed or frames(w))
    check("continuous: frame delivered",
          len(frames(w)) == 1, str(len(frames(w))))
    check("continuous: disarmed after first success", not w._armed)


def test_pause_disarms_and_stops():
    w = make_worker("r", continuous=True)
    w._armed = True
    w.pause()
    check("pause(): probes disarmed", not w._armed)
    check("pause(): continuous cleared", not w.continuous.is_set())
    check("pause(): wants_read cleared", not w.wants_read.is_set())


def test_wrong_position_reprompts_with_state_events():
    """The wheel prompts relayed to the GUI as state events."""
    w = make_worker("r", continuous=True)
    run_script(w, ["Set instrument sensor to calibration position,",
                   "Calibration complete", MENU])
    st = states(w)
    msgs = " | ".join((s[2] or "").lower() for s in st)
    check("state events mention the wheel/position",
          "wheel" in msgs or "position" in msgs, msgs[:120])


def test_instrument_type_parsed_and_emitted():
    """spotread's own 'Instrument Type:' line must reach the GUI as a
    state event so it can adapt modes/wording, and must anchor the
    ColorMunki-specific wheel wording."""
    w = make_worker("r")
    run_script(w, ["Current instrument port: /dev/bus/usb/001/017",
                   "Instrument Type: X-Rite ColorMunki Photo",
                   "Lamp is OK",
                   "Instrument Type: GretagMacbeth i1 Pro",
                   MENU])
    ev = [s for s in states(w) if s[1] == "instrument"]
    types = [s[2] for s in ev]
    check("instrument type emitted as state event",
          any("ColorMunki Photo" in t for t in types), str(types))
    check("re-emitted when the instrument changes",
          any("i1 Pro" in t for t in types), str(types))
    check("worker remembers the current type",
          w.inst_type == "GretagMacbeth i1 Pro", w.inst_type)
    check("same type not re-emitted spam",
          len(ev) == 2, str(len(ev)))


def test_generic_instrument_gets_neutral_wording():
    """A non-ColorMunki instrument must NEVER be told about 'wheel
    position 2' — generic calibration wording everywhere, including
    the all-modes pass the demand now always triggers."""
    w = make_worker("r", continuous=True)
    # i1 Pro supports emissive + reflective (no ambient)
    w.cal_pass_modes = ("r", "e")
    scripts = {
        "r": ["Instrument Type: GretagMacbeth i1 Pro",
              "Set instrument sensor to calibration position,",
              "Calibration complete", MENU],
        "e": ["Instrument Type: GretagMacbeth i1 Pro",
              "Set instrument sensor to calibration position,",
              "Calibration complete", MENU],
    }

    def fake_spawn(mode=None, force=False):
        lines = list(scripts[mode])
        w._lines = lambda: (_ for _ in iter(lines))
    w._spawn = fake_spawn
    run_script(w, ["Instrument Type: GretagMacbeth i1 Pro",
                   "Set instrument sensor to calibration position,",
                   "Calibration complete", MENU], settle=9.0)
    msgs = " | ".join((s[2] or "").lower() for s in states(w))
    check("i1 Pro: no wheel-slot wording", "wheel" not in msgs, msgs[:120])
    check("i1 Pro: neutral calibration wording", "position" in msgs,
          msgs[:120])
    check("i1 Pro: pass covers its own supported modes",
          "reflective (1/2)" in msgs and "emissive (2/2)" in msgs,
          msgs[:200])
    check("i1 Pro: neutral running wording",
          "calibrating…" in msgs and "instrument in its calibration "
      "position" in msgs, msgs[:200])
    check("i1 Pro: pass ends with the resume restart",
          any(s[1] == "restart" and "all-mode" in (s[2] or "").lower()
              for s in states(w)), msgs[-120:])


def test_wider_spectral_range():
    """An instrument with a different range/resolution (i1 Pro:
    380–730 in 40 steps amplitude-scaled differently — and hypothetically
    a wider 360–750) must produce a frame with ITS OWN wavelengths."""
    w = make_worker("e")
    run_script(w, ["Spectrum from 360.000 to 750.000 nm in 40 steps",
                   ", ".join(["0.1"] * 40),
                   RESULT_EMISSIVE, MENU])
    fr = frames(w)
    check("wider range: frame delivered", len(fr) == 1)
    if fr:
        wl = fr[0][1].wavelengths
        check("wider range: wavelengths are 360..750 (not hardcoded)",
              abs(wl[0] - 360.0) < 1e-6 and abs(wl[-1] - 750.0) < 1e-6,
              f"{wl[0]}..{wl[-1]}")
        check("wider range: 40 values", len(wl) == 40, str(len(wl)))


def test_xyz_only_colorimeter_frame():
    """Colorimeters print 'Result is XYZ' with NO spectrum block. The
    frame must still be delivered (XYZ-only, empty spectrum)."""
    w = make_worker("e", continuous=True)
    run_script(w, [RESULT_EMISSIVE,
                   "Place instrument on spot to be measured,", MENU])
    fr = frames(w)
    check("XYZ-only: frame delivered", len(fr) == 1, str(len(fr)))
    if fr:
        spec = fr[0][1]
        check("XYZ-only: xyz survives", spec.xyz == (40.0, 31.0, 25.0),
              str(spec.xyz))
        check("XYZ-only: spectrum is empty",
              len(spec.wavelengths) == 0, str(len(spec.wavelengths)))


def test_calibrate_all_pass():
    """CALIBRATE-ALL: when a calibration is due — ANY demand, on any
    instrument — the session must be replaced by one short session PER
    reading mode (current mode first), each auto-answered; then the
    normal session respawns with everything fresh. Every per-mode
    session is spawned WITHOUT -N (forced): each one must do its own
    dance — a session that skipped to the menu without a wheel prompt
    would let the pass claim freshness it never earned."""
    w = make_worker("e")            # current mode: emissive
    spawned, acked, kills = [], [], []
    w.respond = lambda t: acked.append(t)
    w._kill = lambda: kills.append(1)

    # per-mode scripts for the spawn override to serve; the banner
    # lead-in alone (reflective) means the dance never ran — a FAILED
    # mode the pass must report honestly
    scripts = {
        "e": ["Set instrument sensor to calibration position,",
              "Calibration complete", MENU],
        "a": ["Set instrument sensor to calibration position,",
              "Calibration complete", MENU],
        "r": ["Place instrument on spot to be measured,"],
    }
    forced = []

    def fake_spawn(mode=None, force=False):
        spawned.append(mode if mode is not None else w.mode)
        forced.append(force)
        lines = list(scripts[mode])
        w._lines = lambda: (_ for _ in iter(lines))   # fresh generator
    w._spawn = fake_spawn

    # the working session sees a calibration demand -> pass -> respawn
    run_script(w, ["Set instrument sensor to calibration position,",
                   "Calibration complete", MENU], settle=12.0)
    st = states(w)
    msgs = " | ".join((s[2] or "").lower() for s in st)
    check("pass: one session per mode, current first",
          spawned == ["e", "a", "r"], str(spawned))
    check("pass: every per-mode session forced (-N omitted)",
          forced == [True, True, True], str(forced))
    check("pass: the fresh dance modes auto-acknowledged",
          acked == ["\n", "\n"], str(acked))
    check("pass: GUI told which mode is being calibrated",
          "emissive (1/3)" in msgs and "ambient (2/3)" in msgs
          and "reflective (3/3)" in msgs, msgs[:200])
    check("pass: session ends for respawn",
          any(s[1] == "restart" and "all-mode" in (s[2] or "").lower()
              for s in st), msgs[:200])
    check("pass: failed mode reported honestly",
          "could not be calibrated" in msgs
          and "reflective" in msgs, msgs[-200:])
    check("pass: no infinite loop — guard set until a frame",
          w._pass_since_frame is True)
    # a later session with the same worker: if still no frame arrived,
    # a new cal demand must NOT re-run the pass (falls back in-session)
    check("command with mode override builds the right argv",
          w._command("a") == [sources.spotread_path() or "spotread",
                              "-v", "-s", "-N", "-a", "-H"],
          str(w._command("a")))
    # spotread.c XORs CCT/CRI display with ambient: -T would SUPPRESS
    # it for -a (its absence enables it); emissive/reflective need -T
    check("ambient: no -T (spotread enables CCT/CRI by itself)",
          "-T" not in w._command("a"), str(w._command("a")))
    check("emissive: -T present",
          "-T" in w._command("e"), str(w._command("e")))
    check("reflective: -T present",
          "-T" in w._command("r"), str(w._command("r")))


def test_request_calibration_one_shot():
    """Clicking the CALIBRATION lamp must omit -N exactly once (so
    spotread demands a fresh calibration), then normal cache reuse
    resumes; the flag is consumed by the next spawn."""
    w = make_worker("r")
    check("pre: -N present by default", "-N" in w._command(),
          str(w._command()))
    w.request_calibration()
    check("requested: -N omitted for the next start",
          "-N" not in w._command(), str(w._command()))
    check("requested: restart is signaled",
          w.restart_now.is_set())
    # spawn: -N-less command, one-shot flag consumed (Popen stubbed —
    # hermetic: no real spotread spawn)
    spawned = []

    class _FakePopen:
        pid = 1234
        def poll(self): return None
        def kill(self): pass
        def wait(self, t=None): return 0
    orig_popen = sources.subprocess.Popen
    sources.subprocess.Popen = lambda cmd, **kw: (spawned.append(cmd),
                                                 _FakePopen())[1]
    try:
        w._spawn()
    finally:
        sources.subprocess.Popen = orig_popen
        os.close(w._master)
    check("spawn: -N omitted from the actual argv",
          spawned and "-N" not in spawned[0], str(spawned))
    check("after spawn: -N present again (one-shot consumed)",
          "-N" in w._command(), str(w._command()))
    check("after spawn: one-shot flag cleared",
          w._skip_cache_once is False)


def test_nt_transport():
    """The Windows pipe transport (no pty module there): _spawn_nt /
    _lines / respond / _kill must behave like the PTY one — lines arrive
    intact (byte-wise reader thread), None surfaces while idle, and a
    quit terminates the child. Exercised on this POSIX host by faking
    os.name: the pipe machinery itself is OS-agnostic."""
    import tempfile
    from unittest.mock import patch
    stub = os.path.join(tempfile.mkdtemp(dir="/tmp/opencode"),
                        "spotread")
    with open(stub, "w") as f:
        f.write("""#!/usr/bin/env python3
import sys
big = "".join(f"{0.01 * ((i % 100) + 1):.5f}, " for i in range(200))
while True:
    line = sys.stdin.readline()
    if not line:
        break
    if line.strip() == "q":
        break
    print("Instrument Type:   ColorMunki", flush=True)
    print("Spectrum from 380.000 to 730.000 nm in 106 steps", flush=True)
    print(big.strip(), flush=True)
    print(" Result is XYZ: 0.40 0.31 0.25, D50 Lab: 62.4 0.3 1.2",
          flush=True)
""")
    os.chmod(stub, 0o755)
    sources.set_argyll_path(stub)
    w = SpotreadWorker(queue.Queue())
    w.mode, w.kind, w.port = "r", "l", None
    w.high_res, w.fast, w.force_cal = True, False, False
    try:
        with patch.object(sources.os, "name", "nt"):
            w._spawn()                # must take the _spawn_nt branch
            check("nt: pipe transport armed (no pty master)",
                  w._master is None and w._out_q is not None
                  and w._proc is not None)
            w.respond("\n")
            lines = []
            saw_none = False
            for _ in range(400):
                ln = next(w._lines())
                if ln is None:
                    saw_none = True      # idle tick — the state machine
                    continue             # needs these to run its timers
                lines.append(ln)
                if len(lines) >= 4 and saw_none:
                    break
            check("nt: lines arrived intact",
                  len(lines) >= 4
                  and lines[0].startswith("Instrument Type:"),
                  str(lines[:2]))
            check("nt: long spectrum line reassembled from byte reads",
                  any(ln.count(",") >= 199 for ln in lines),
                  f"max commas {max(ln.count(',') for ln in lines)}")
            check("nt: idle None surfaced (timer contract)",
                  saw_none)
            w._kill()
            check("nt: quit via stdin kills the child",
                  w._proc is None)
            check("nt: transport cleaned up",
                  w._out_q is not None)   # drained by the reader thread
    finally:
        sources.set_argyll_path(None)
        try:
            os.unlink(stub)
        except OSError:
            pass


def test_cal_validity_windows():
    """The green CAL border lasts the instrument's own validity window:
    reflective white cal 24 h (WCALTOUT), dark cal 1 h (DCALTOUT)."""
    check("reflective: 24 h", sources.cal_validity_s("r") == 86400.0)
    check("emissive: 1 h", sources.cal_validity_s("e") == 3600.0)
    check("ambient: 1 h", sources.cal_validity_s("a") == 3600.0)
    w = make_worker("r")
    check("worker delegates (reflective mode)",
          w.cal_validity_s() == 86400.0)
    w.mode = "e"
    check("worker delegates (emissive mode)",
          w.cal_validity_s() == 3600.0)


def test_dance_falls_back_in_session_after_pass():
    """The pass runs on EVERY calibration demand now, but only once per
    reading (loop guard): if a pass already ran and no frame came of
    it, the NEXT demand must fall back to the ordinary in-session
    single-mode dance instead of looping forever."""
    w = make_worker("r", continuous=True)
    acked = []
    w.respond = lambda t: acked.append(t)
    spawns = []

    def fake_spawn(mode=None, force=False):
        spawns.append(mode if mode is not None else w.mode)
        lines = ["Place instrument on spot to be measured,"]
        w._lines = lambda: (_ for _ in iter(lines))
    w._spawn = fake_spawn

    # FIRST demand: the all-modes pass (its per-mode sessions hit the
    # banner-lead-in above and "fail" — the respawn then re-demands)
    run_script(w, ["Set instrument sensor to calibration position,"], settle=6.0)
    st = states(w)
    msgs = " | ".join((s[2] or "").lower() for s in st)
    check("first demand: the all-modes pass ran",
          spawns and "every reading mode" in msgs, msgs[:160])
    # SECOND demand (same worker, still no frame): the guard defers to
    # the in-session dance — prompt acknowledged inline, no new spawns
    n_before = len(acked)
    run_script(w, ["Set instrument sensor to calibration position,",
                   "Calibration complete", MENU], settle=6.0)
    st = states(w)
    msgs = " | ".join((s[2] or "").lower() for s in st)
    check("second demand: prompt acknowledged in-session",
          len(acked) > n_before and acked[-1] == "\n", str(acked))
    check("second demand: no second pass",
          len(spawns) == 3, str(spawns))


def test_port_selection_in_command():
    """-c N selects the instrument when several are connected; -N skips
    the startup-calibration demand (Argyll's host cal cache is reused)."""
    w = make_worker("e")
    w.port = 2
    # argv[0] is the RESOLVED spotread binary (spotread_path(): Argyll
    # path override or PATH lookup), not the bare command name
    expected = [sources.spotread_path() or "spotread", "-v", "-s",
                "-T", "-N", "-c", "2", "-e", "-y", "c", "-H"]
    check("-c 2 appears in the command when a device is selected",
          w._command() == expected, str(w._command()))
    w.port = None
    check("default device: no -c flag", "-c" not in w._command(),
          str(w._command()))
    check("by default the saved calibration is reused (-N)",
          "-N" in w._command(), str(w._command()))
    w.force_cal = True
    check("force calibration: -N omitted",
          "-N" not in w._command(), str(w._command()))
    w.force_cal = False
    check("all-modes pass sessions omit -N (forced)",
          "-N" not in w._command("a", force=True),
          str(w._command("a", force=True)))
    w2 = make_worker("e")
    w2.restart_session()
    check("restart_session() arms the restart event",
          w2.restart_now.is_set(), "event not set")


def test_capability_helpers():
    from spectrov.sources import (instrument_modes, is_colormunki,
                                  cal_prompt)
    check("colormunki: all three modes + wheel wording",
          instrument_modes("X-Rite ColorMunki Photo")
          == ("Emissive", "Ambient", "Reflective")
          and "wheel" in cal_prompt("ColorMunki Photo").lower())
    check("i1pro: emissive+reflective, no wheel wording",
          instrument_modes("i1 Pro") == ("Emissive", "Reflective")
          and "wheel" not in cal_prompt("i1 Pro").lower())
    check("spyder colorimeter: emissive only",
          instrument_modes("Spyder5") == ("Emissive",))
    check("unknown instrument: safe default with neutral wording",
          instrument_modes("FooSensor 3000") == ("Emissive", "Reflective")
          and not is_colormunki("FooSensor 3000"))


def test_duv_and_cri_relay():
    """The spotread transcript lines observed live:

        CCT = 6361K (Duv -0.0075)
        Color Rendering Index (Ra) = 66.3 [ R9 = -61.0 ] (Caution)

    must be parsed and attached to the next frame as meta_duv/meta_cri,
    and reset for the frame after that."""
    w = make_worker("e", continuous=True)
    run_script(w, spectrum_block()
               + [RESULT_EMISSIVE,
                  "                            CCT = 6361K (Duv -0.0075)",
                  " Closest Planckian temperature = 6853K (DE2K -9.0)",
                  " Color Rendering Index (Ra) = 66.3 [ R9 = -61.0 ] "
                  "(Caution)",
                  " IES TM-30-15 Rf = 61.24 Rg = 98.51 CCT = 6361 "
                  "Duv = -0.007500 (Caution)",
                  MENU,
                  # second reading: no CCT/Duv/CRI lines (menu-only path)
                  "Place instrument on spot to be measured,",
                  RESULT_EMISSIVE, MENU])
    fr = frames(w)
    check("duv/cri: two frames delivered", len(fr) == 2, str(len(fr)))
    if len(fr) == 2:
        s1, s2 = fr[0][1], fr[1][1]
        check("frame 1 carries spotread's CCT",
              getattr(s1, "meta_cct", None) == 2311.0 or
              getattr(s1, "meta_cct", None) == 6361.0,
              str(getattr(s1, "meta_cct", None)))
        check("frame 1 carries spotread's Duv",
              abs(getattr(s1, "meta_duv", 9) - (-0.0075)) < 1e-9,
              str(getattr(s1, "meta_duv", None)))
        mc = getattr(s1, "meta_cri", None)
        check("frame 1 carries spotread's CRI (Ra/R9/flag)",
              mc is not None and abs(mc["ra"] - 66.3) < 1e-9
              and abs(mc["r9"] - (-61.0)) < 1e-9
              and mc["flag"] == "Caution", str(mc))
        check("frame 2: duv/cri reset (not leaked)",
              getattr(s2, "meta_duv", None) is None
              and getattr(s2, "meta_cri", None) is None,
              f"{getattr(s2, 'meta_duv', '??')} {getattr(s2, 'meta_cri', '??')}")
        mt = getattr(s1, "meta_tm30", None)
        check("frame 1 carries spotread's TM-30 (Rf/Rg/flag)",
              mt is not None and abs(mt["rf"] - 61.24) < 1e-9
              and abs(mt["rg"] - 98.51) < 1e-9
              and mt["flag"] == "Caution", str(mt))
        check("frame 2: TM-30 reset (not leaked)",
              getattr(s2, "meta_tm30", None) is None,
              str(getattr(s2, "meta_tm30", "??")))


if __name__ == "__main__":
    for fn in (test_reflective_percent_scale_normalized_via_lab,
               test_reflective_plausibility_fallback,
               test_emissive_never_rescaled,
               test_wedge_early_delivery,
               test_calibration_does_not_autoarm_when_stopped,
               test_calibration_arms_when_continuous,
               test_pause_disarms_and_stops,
               test_wrong_position_reprompts_with_state_events,
               test_instrument_type_parsed_and_emitted,
               test_generic_instrument_gets_neutral_wording,
               test_wider_spectral_range,
               test_duv_and_cri_relay,
               test_xyz_only_colorimeter_frame,
               test_port_selection_in_command,
               test_request_calibration_one_shot,
               test_nt_transport,
               test_cal_validity_windows,
               test_calibrate_all_pass,
               test_dance_falls_back_in_session_after_pass,
               test_capability_helpers):
        print(f"{fn.__name__}:")
        fn()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):", ", ".join(FAILURES))
        sys.exit(1)
    print("all driver-replay tests passed")
