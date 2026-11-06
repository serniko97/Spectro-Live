"""Measurement sources: ArgyllCMS spotread (any supported instrument), a
built-in simulator and file loading/saving (CSV / CGATS / Argyll .sp).

Everything is thread-based with a callback queue: sources push
("state", ...), ("frame", Spectrum) and ("log", text) messages, never
touching the GUI directly. spotread is run through a PTY because its
keypress prompts need a real terminal.
"""
from __future__ import annotations

import glob
import os
import queue
import random
import re
import select
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime

import numpy as np

from .processing import (Spectrum, approx_xyz_from_spectrum, cct_from_xyz,
                         find_peaks, gamut_report, peak_info, srgb_preview,
                         xy_from_xyz)


# ---------------------------------------------------------------------------
# Device presence (sysfs scan, no dependencies)
# ---------------------------------------------------------------------------
# Best-effort nicer labels; the REAL compatibility test is spotread itself:
# if Argyll can drive it, the session starts — our scan must not be the
# gatekeeper for devices we don't know by name.
KNOWN_INSTRUMENTS = {("0971", "2007"): "ColorMunki Photo",
                     ("0971", "3007"): "ColorMunki Photo",
                     ("0765", "0852"): "ColorMunki Design",
                     ("0765", "085a"): "ColorMunki Display"}

_INSTRUMENT_HINTS = ("munki", "i1", "spyder", "spectro", "k10",
                     "dtp", "huey", "colibri", "color", "x-rite",
                     "argyll", "monster", "hueypro")

# HID devices that are definitely NOT measurement instruments: the
# widened scan (any HID device) must not present the user's keyboard as
# the session instrument just because it sorts first on the bus.
_NON_INSTRUMENTS = ("keyboard", "mouse", "touchpad", "trackball",
                    "digitizer", "receiver", "gaming", "joystick",
                    "webcam", "headset", "audio", "led", "controller")


def is_candidate(v: str, p: str, product: str, hid: bool) -> bool:
    """Does this USB device qualify as a possible instrument?
    HID-class interface, recognised instrument ids, or an
    instrument-like product name (some instruments — e.g. the
    ColorMunki — enumerate with a vendor-specific interface class
    instead of HID; spotread -v tells us what it really is once its
    session starts)."""
    if hid:
        return True
    if (v, p) in KNOWN_INSTRUMENTS:
        return True
    label = KNOWN_INSTRUMENTS.get((v, p), product or f"USB {v}:{p}")
    return any(h in (label + " " + v + ":" + p).lower()
               for h in _INSTRUMENT_HINTS)


def usb_devices() -> list[tuple[str, str, str]]:
    """USB devices that could be Argyll instruments: HID-class devices,
    plus anything with recognised instrument ids or a product name that
    sounds like one (some instruments — e.g. the ColorMunki — enumerate
    with a vendor-specific interface class instead of HID). Labelled by
    the device's own product name (or our known-name map). Sorted by
    bus path so the order roughly matches spotread's -c enumeration.

    macOS port: there is no sysfs, so the scan runs
    `system_profiler SPUSBDataType -json` instead (expensive, a few
    seconds — cached for a TTL) and matches by vendor/product ids and
    product names. The enumeration ORDER is best-effort: spotread walks
    IOUSB itself, so with multiple instruments the -c numbers may not
    match our list — pick 'Auto' and let spotread open the first, or
    try each entry (spotread remains the final arbiter, as always)."""
    if sys.platform == "darwin":
        return _mac_usb_scan()
    if os.name == "nt":
        return _win_usb_scan()
    if not os.path.isdir("/sys/bus/usb"):
        # some other non-Linux unix: no scanner exists — report nothing
        # honestly rather than pretending the bus is empty
        return []
    out = []
    for d in glob.glob("/sys/bus/usb/devices/*/"):
        base = os.path.basename(d.rstrip("/"))
        if ":" in base:            # interface entry, not a device
            continue
        try:
            with open(os.path.join(d, "idVendor")) as f:
                v = f.read().strip()
            with open(os.path.join(d, "idProduct")) as f:
                p = f.read().strip()
            with open(os.path.join(d, "busnum")) as f:
                bus = int(f.read().strip())
            with open(os.path.join(d, "devnum")) as f:
                dev = int(f.read().strip())
        except OSError:
            continue
        # instrument candidates: HID interface or recognised instrument
        # (checked below by is_candidate)
        hid = False
        for iface in glob.glob(os.path.join(d, base + ":*/bInterfaceClass")):
            try:
                with open(iface) as f:
                    if f.read().strip() == "03":
                        hid = True
                        break
            except OSError:
                continue
        product = ""
        try:
            with open(os.path.join(d, "product")) as f:
                product = f.read().strip()
        except OSError:
            pass
        label = KNOWN_INSTRUMENTS.get((v, p), product or f"USB {v}:{p}")
        if not is_candidate(v, p, product, hid):
            continue
        out.append((f"{bus:03d}-{dev:03d}", f"{v}:{p}", label))
    return sorted(out, key=lambda t: t[0])


# ---- macOS USB scan (no sysfs; system_profiler is the only clean
# source of vendor/product ids without extra dependencies) -----------
_MAC_SCAN: list[tuple[str, str, str]] | None = None
_MAC_SCAN_AT = 0.0
_MAC_SCAN_TTL = 15.0       # seconds; system_profiler takes seconds


def _mac_usb_parse(json_out: str) -> list[tuple[str, str, str]]:
    """Parse `system_profiler SPUSBDataType -json` output into the same
    (bus-key, 'vid:pid', label) triples usb_devices() returns on Linux.
    Pure function (feed it the text) so it is testable anywhere."""
    import json
    try:
        data = json.loads(json_out)
    except (ValueError, TypeError):
        return []
    items = data.get("SPUSBDataType", []) if isinstance(data, dict) else []
    out: list[tuple[str, str, str]] = []

    def walk(nodes):
        for it in nodes or []:
            if not isinstance(it, dict):
                continue
            walk(it.get("_items"))
            name = str(it.get("_name") or "").strip()
            vid_raw = str(it.get("vendor_id") or "").split()
            pid_raw = str(it.get("product_id") or "").split()
            if not vid_raw or not pid_raw:
                continue
            vid = vid_raw[0].lower().removeprefix("0x")
            pid = pid_raw[0].lower().removeprefix("0x")
            label = KNOWN_INSTRUMENTS.get((vid, pid),
                                          name or f"USB {vid}:{pid}")
            # system_profiler does not tell us the USB interface class,
            # so the HID branch of is_candidate cannot apply — known
            # ids and name hints are the filter (spotread remains the
            # final arbiter when its session opens)
            if not is_candidate(vid, pid, label, hid=False):
                continue
            out.append((f"{vid}:{pid}", f"{vid}:{pid}", label))

    walk(items)
    return out


def _mac_usb_scan() -> list[tuple[str, str, str]]:
    """usb_devices() body on macOS — with a TTL cache so the respawn
    loop's ~2 s presence probes don't spawn system_profiler (which
    takes seconds) every round."""
    global _MAC_SCAN, _MAC_SCAN_AT
    now = time.monotonic()
    if _MAC_SCAN is not None and now - _MAC_SCAN_AT < _MAC_SCAN_TTL:
        return _MAC_SCAN
    try:
        out = subprocess.run(
            ["system_profiler", "SPUSBDataType", "-json"],
            capture_output=True, text=True, timeout=30).stdout
        _MAC_SCAN = _mac_usb_parse(out)
    except (OSError, subprocess.SubprocessError):
        _MAC_SCAN = []        # honest: no scanner, no devices reported
    _MAC_SCAN_AT = now
    return _MAC_SCAN


# ---- Windows USB scan (no sysfs, no system_profiler; PowerShell CIM --
# is the pragmatic source of vendor/product ids). Also cached: the
# respawn loop probes every ~2 s and powershell takes ~1 s per run.
_WIN_SCAN: list[tuple[str, str, str]] | None = None
_WIN_SCAN_AT = 0.0
_WIN_SCAN_TTL = 15.0


def _win_usb_parse(json_out: str) -> list[tuple[str, str, str]]:
    """Parse `Get-CimInstance Win32_USBHub | ConvertTo-Json` output
    into the same (key, 'vid:pid', label) triples usb_devices() returns
    elsewhere. PNPDeviceID looks like 'USB\\VID_0971&PID_2007\\<serial>'.
    Pure function (feed it the text) so it is testable anywhere."""
    import json
    try:
        data = json.loads(json_out)
    except (ValueError, TypeError):
        return []
    if isinstance(data, dict):       # ConvertTo-Json emits a single
        data = [data]                # object (not a list) for one item
    out: list[tuple[str, str, str]] = []
    for it in data if isinstance(data, list) else []:
        if not isinstance(it, dict):
            continue
        pnp = str(it.get("PNPDeviceID") or "")
        m = re.search(r"VID_([0-9A-Fa-f]{4})&PID_([0-9A-Fa-f]{4})",
                      pnp)
        if not m:
            continue
        vid, pid = m.group(1).lower(), m.group(2).lower()
        name = str(it.get("Name") or "").strip()
        label = KNOWN_INSTRUMENTS.get((vid, pid),
                                      name or f"USB {vid}:{pid}")
        # CIM gives no USB interface class: filter by known ids and
        # name hints only — spotread remains the final arbiter once
        # its session opens
        if not is_candidate(vid, pid, label, hid=False):
            continue
        out.append((f"{vid}:{pid}", f"{vid}:{pid}", label))
    return out


def _win_usb_scan() -> list[tuple[str, str, str]]:
    """usb_devices() body on Windows — TTL-cached like the macOS scan."""
    global _WIN_SCAN, _WIN_SCAN_AT
    now = time.monotonic()
    if _WIN_SCAN is not None and now - _WIN_SCAN_AT < _WIN_SCAN_TTL:
        return _WIN_SCAN
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command",
             "Get-CimInstance Win32_USBHub | Select-Object Name,"
             "PNPDeviceID | ConvertTo-Json -Compress"],
            capture_output=True, text=True, timeout=30).stdout
        _WIN_SCAN = _win_usb_parse(out)
    except (OSError, subprocess.SubprocessError):
        _WIN_SCAN = []        # honest: no scanner, no devices reported
    _WIN_SCAN_AT = now
    return _WIN_SCAN


def instrument_present() -> str | None:
    """Label of the most plausible instrument, or None.

    Preference order: an instrument-like USB device (our hints or its
    own product string), then an unnamed/unknown HID device that is at
    least not obviously a keyboard/mouse. spotread remains the real
    arbiter — the moment its session starts it reports the instrument
    type it actually opened."""
    devs = usb_devices()
    hinted = [d for d in devs
              if any(h in (d[2] + " " + d[1]).lower()
                     for h in _INSTRUMENT_HINTS)]
    if hinted:
        return hinted[0][2]
    plausible = [d for d in devs
                 if not any(n in d[2].lower() for n in _NON_INSTRUMENTS)]
    return plausible[0][2] if plausible else None


def set_argyll_path(path: str | None) -> None:
    """User-specified spotread location (Instrument → Argyll path…):
    an absolute path to a spotread binary, or None/"" to fall back to
    whatever is on PATH."""
    global _ARGYLL_OVERRIDE
    _ARGYLL_OVERRIDE = path or None


_ARGYLL_OVERRIDE: str | None = None


def spotread_path() -> str | None:
    if _ARGYLL_OVERRIDE:
        # an explicit user choice: accept only a real executable file
        # (both the no-argyll state report and every spawn check it)
        try:
            if os.path.isfile(_ARGYLL_OVERRIDE) \
                    and os.access(_ARGYLL_OVERRIDE, os.X_OK):
                return _ARGYLL_OVERRIDE
        except OSError:
            pass
        return None
    return shutil.which("spotread")


# ---------------------------------------------------------------------------
# Worker base
# ---------------------------------------------------------------------------
class SourceWorker(threading.Thread):
    """Base class: pushes events to a Queue:
       ("state", key, message)   idle|no-argyll|no-device|reconnecting|starting|
                                 calibrating|calibrate-confirm|measuring|error
       ("frame", Spectrum)
       ("log", text)
    """
    def __init__(self, out: queue.Queue):
        super().__init__(daemon=True)
        self.out = out
        self._stop = threading.Event()
        self.wants_read = threading.Event()     # one-shot reading request
        self.continuous = threading.Event()     # keep reading forever
        self.user_confirm = threading.Event()   # (reserved for GUI prompts)

    def stop(self):
        self._stop.set()
        self.wants_read.set()
        self.user_confirm.set()

    def request_read(self):
        """Request one more reading (one-shot)."""
        self.wants_read.set()

    def start_continuous(self):
        """Keep requesting readings until pause()/stop()."""
        self.continuous.set()

    def pause(self):
        """Stop the reading loop but keep the instrument session (and its
        calibration) alive. No measurement pending -> no position probes."""
        self.wants_read.clear()
        self.continuous.clear()
        self._armed = False

    def arm_probes(self):
        """A new measurement phase just started (START, SINGLE, resume):
        until the wheel/target is in the right place, failed reads are
        retried silently every few seconds instead of filling the GUI
        with errors. Disarmed by pause(), stop() (via new session) and
        by the first successful reading."""
        self._armed = True

    def resume(self):
        """Resume continuous measurements (or one shot if paused)."""
        if self.continuous.is_set():
            self.wants_read.set()
        else:
            self.wants_read.set()

    def respond(self, text: str):
        """Send a key press to the instrument program (e.g. '\n')."""
        raise NotImplementedError

    def emit_state(self, key, msg=""):
        self.out.put(("state", key, msg))

    def emit_frame(self, spec: Spectrum):
        self.out.put(("frame", spec))

    def emit_log(self, text):
        self.out.put(("log", text))

    def _finish(self):
        try:
            self.emit_state("idle")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# ArgyllCMS spotread driver (PTY based)
# ---------------------------------------------------------------------------
# observed output (spotread 3.4.1, ColorMunki Photo):
#   "Spectrum from 380.000 to 730.000 nm in 106 steps"
#   "  -5.4407e-05, ..., 0.00284167"        <- ONE long comma-separated line
#   "Peak value 0.026259 at (aprox.) 456.7 nm"
#   " Result is XYZ: 1.423468 1.425269 1.050596, D50 Lab: ..."
#   "CCT = 4375K (Duv -0.0005)"
# errors: "Spot read failed due to misread (Sensor is saturated)"
#         "wrong position (Sensor should be in surface position)"
_RE_SPECTRUM_HDR = re.compile(r"Spectrum from\s+([0-9.]+)\s+to\s+([0-9.]+)\s+nm\s+in\s+(\d+)\s+steps")
_RE_RESULT = re.compile(r"Result is\s+(?:XYZ:\s*)?([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)")
_RE_LAB = re.compile(r"D50\s+Lab:\s*(-?[0-9.eE+-]+)\s+(-?[0-9.eE+-]+)\s+(-?[0-9.eE+-]+)")
_RE_CCT = re.compile(r"CCT\s*=?\s*([0-9.eE+-]+)")
# "CCT = 6361K (Duv -0.0075)" — spotread reports Duv whenever the CCT
# is meaningful; |Duv| > 0.006 means the CCT is far off the Planckian
# locus and only vaguely "correlated".
_RE_DUV = re.compile(r"\(Duv\s*([-+0-9.]+)\)")
# "Color Rendering Index (Ra) = 66.3 [ R9 = -61.0 ] (Caution)"
# — Argyll's own validity flag warns of a truncated spectral range etc.
_RE_CRI = re.compile(r"Color Rendering Index \(Ra\) = (-?[0-9.]+)"
                     r"(?:\s*\[\s*R9 = (-?[0-9.]+)\s*\])?"
                     r"(?:\s*\(\s*(Caution|Invalid)\s*\))?",
                     re.I)
# " IES TM-30-15 Rf = 85.34 Rg = 99.81 CCT = 6102 Duv = 0.001200 (Caution)"
# — IES TM-30-15: Rf = colour FIDELITY over 99 test samples (CIE Ra's
# modern successor; 100 = renders colours exactly like daylight/Planck
# of the same CCT), Rg = relative GAMUT AREA (Rg > 100 = the light
# over-saturates colours, < 100 = washes them out). spotread prints the
# same Caution flag as for CRI (truncated spectral range etc.); on a
# failed calculation it prints no numbers at all — treated as absent.
# The flag, when present, sits at the END of the line, behind TM-30's
# own "CCT = … Duv = …" summary text — hence the .* before it.
_RE_TM30 = re.compile(r"IES TM-30-15 Rf = (-?[0-9.]+)\s+Rg = (-?[0-9.]+)"
                      r"(?:.*\(\s*(Caution|Invalid)\s*\))?",
                      re.I)
_RE_FAIL = re.compile(r"Spot read failed due to (.+)")
_RE_WRONGPOS = re.compile(r"wrong position", re.I)
_RE_WHEEL = re.compile(r"Set instrument sensor to (?:the )?calibration position", re.I)
_RE_MENU = re.compile(r"any other key to take a reading", re.I)
# spotread's menu BANNER lead-in — printed the instant the session is
# at its menu; the closing "Hit ESC or Q …" line can lag by minutes
# (observed on the ColorMunki Photo), so never wait for it alone.
_RE_MENU_LEAD = re.compile(r"Place instrument on spot to be measured", re.I)
_RE_INSTTYPE = re.compile(r"Instrument Type:\s*(.+)")

# What a given instrument supports. Keyed by spotread's own
# "Instrument Type:" string (lower-cased substring match); unknown
# instruments get the conservative default (emissive only is too
# restrictive — spotread's default IS a reflective reading — so unknown
# spectrometers are trusted to do the two safe modes).
_INSTRUMENT_MODES = (
    ("colormunki", ("Emissive", "Ambient", "Reflective")),
    ("i1pro", ("Emissive", "Reflective")),          # ambient needs the
    ("i1 pro", ("Emissive", "Reflective")),         # clip-on adapter
    ("i1pro2", ("Emissive", "Reflective")),
    ("i1pro3", ("Emissive", "Reflective")),
    ("spyder", ("Emissive",)),                      # colorimeter, no LED
    ("dtp94", ("Emissive",)),
    ("huey", ("Emissive",)),
    ("k10", ("Emissive",)),
)


def instrument_modes(inst_type: str) -> tuple[str, ...]:
    """Reading modes (subset of Emissive/Ambient/Reflective) an
    instrument supports, judged from Argyll's own type string."""
    low = (inst_type or "").lower()
    for key, modes in _INSTRUMENT_MODES:
        if key in low:
            return modes
    # unknown instrument: spotread's default mode (reflective spot)
    # plus emissive are the safe bets; ambient is the rare capability
    return ("Emissive", "Reflective")


def is_colormunki(inst_type: str) -> bool:
    return "munki" in (inst_type or "").lower()


# Device-aware wording. Only ColorMunki-style instruments have the
# rotating wheel with numbered slots and the top diffuser — every other
# instrument gets wording that stays true regardless of its hardware
# (spotread's own position hints are relayed verbatim anyway).
# --- Argyll host calibration cache ---------------------------------------
# Argyll judges calibration validity against the ACTUAL last
# calibration dates, which munki_restore_calibration() reads back from
# the host cache file (~/.cache/ArgyllCMS/.mk_<serno>.cal): per reading
# mode, a binary layout written by munki_save_calibration() —
# header: int argyllversion, int structs-size, char[17] serial,
#         int nraw, int nwav1, int nwav2
# then 11 mode blocks, each: 8 int flags (emiss, trans, reflective,
# scan, flash, ambient, projector, adaptive), int gainmode,
# double inttime, int dark_valid, time_t ddate, double dark_int_time,
# (nraw+1) doubles dark_data, [dark_int_time2 + (nraw+1) doubles] ×2,
# int dark_gain_mode, [if !emiss: int cal_valid, time_t cfdate,
# nwav1 + nwav2 + (nraw+1) + 1 + 2·(nraw+1) doubles], int idark_valid,
# time_t iddate, 4·double idark_int_time, 4·(nraw+1) doubles
# idark_data … then the trailing checksum int. These functions read
# the dates and apply exactly Argyll's own invalidation rules
# (munki_imp_get_n_a_cals): dark cal DCALTOUT = 1 h, adaptive dark
# uses iddate, white cal WCALTOUT = 24 h, reflective needs BOTH.

MUNKI_CACHE_SUBDIR = "ArgyllCMS"


def munki_cache_dirs() -> list[str]:
    """Candidate directories for Argyll's host calibration cache
    (mirrors Argyll's xdg_bds cache search)."""
    dirs: list[str] = []
    if os.environ.get("XDG_CACHE_HOME"):
        dirs.append(os.path.join(os.environ["XDG_CACHE_HOME"],
                                 MUNKI_CACHE_SUBDIR))
    home = os.path.expanduser("~")
    if sys.platform == "darwin":
        # Argyll's own xdg_bds maps the XDG cache dir to
        # ~/Library/Caches on macOS (xdg_bds.c, UNIX_APPLE)
        dirs.append(os.path.join(home, "Library", "Caches",
                                 MUNKI_CACHE_SUBDIR))
    elif os.name == "nt":
        # Windows: Argyll's xdg_bds maps the cache dir to
        # %APPDATA%\Cache (xdg_bds.c NT branch); HOME-style installs
        # (login_HOME set) use ~/.cache like Unix
        appdata = os.environ.get("APPDATA")
        if appdata:
            dirs.append(os.path.join(appdata, "Cache",
                                     MUNKI_CACHE_SUBDIR))
        dirs.append(os.path.join(home, ".cache", MUNKI_CACHE_SUBDIR))
    else:
        dirs.append(os.path.join(home, ".cache", MUNKI_CACHE_SUBDIR))
    if sys.platform != "win32":
        # Argyll's fallback spelling: <cache>/ArgyllCMS/color/
        for d in list(dirs):
            dirs.append(os.path.join(d, "color"))
    return dirs


def munki_cache_file(serno: str | None = None) -> str | None:
    """The newest Argyll .mk_<serial>.cal cache file; if `serno` is
    given, only a file whose embedded serial matches is accepted."""
    pats = []
    for d in munki_cache_dirs():
        pats.append(os.path.join(d, ".mk_*.cal"))
    best: tuple[float, str] | None = None
    for pat in pats:
        for path in glob.glob(pat):
            if serno is not None:
                try:
                    with open(path, "rb") as f:
                        if f.read(25)[8:].split(b"\0")[0].decode(
                                "ascii", "replace").strip() != str(serno):
                            continue
                except OSError:
                    continue
            m = os.path.getmtime(path)
            if best is None or m > best[0]:
                best = (m, path)
    return best[1] if best else None


def munki_cache_cal_dates(path: str) -> list[dict] | None:
    """Parse the per-mode calibration dates out of Argyll's ColorMunki
    host cache. Returns None when the file does not verify (wrong
    layout/version sanity checks) — callers fall back gracefully."""
    import struct
    import time as _time
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return None
    if len(data) < 64:
        return None
    try:
        # header: argyllversion, structs-size, char[17] serial,
        # nraw, nwav1, nwav2
        nraw, nwav1, nwav2 = struct.unpack_from("<iii", data, 25)
        if not (8 <= nraw <= 128 and 10 <= nwav1 <= 128
                and 10 <= nwav2 <= 256):
            return None
        off = 37
        modes: list[dict] = []
        for _ in range(11):
            (emiss, trans, reflective, _scan, _flash, ambient,
             _proj, adaptive) = struct.unpack_from("<8i", data, off)
            off += 32
            off += 4 + 8 + 4            # gainmode, inttime, dark_valid
            ddate = struct.unpack_from("<q", data, off)[0]
            off += 8                    # time_t ddate
            off += 8 + 8 * (nraw + 1)   # dark_int_time + dark_data
            off += 8 + 8 * (nraw + 1)   # dark_int_time2 + dark_data2
            off += 8 + 8 * (nraw + 1)   # dark_int_time3 + dark_data3
            off += 4                    # dark_gain_mode
            cfdate = None
            if not emiss:
                off += 4                # cal_valid
                cfdate = struct.unpack_from("<q", data, off)[0]
                off += 8                # time_t cfdate
                off += (8 * nwav1 + 8 * nwav2 + 8 * (nraw + 1)
                        + 8 + 2 * 8 * (nraw + 1))
            off += 4                    # idark_valid
            iddate = struct.unpack_from("<q", data, off)[0]
            off += 8                    # time_t iddate
            off += 8 * 4 + 4 * 8 * (nraw + 1)
            modes.append({"emiss": emiss, "trans": trans,
                          "reflective": reflective, "ambient": ambient,
                          "adaptive": adaptive,
                          "ddate": ddate, "cfdate": cfdate,
                          "iddate": iddate})
        # the walk must land exactly on the trailing checksum
        if off + 4 != len(data):
            return None
        # a real calibration dates sanity: epoch stamps within 30 days
        now = int(_time.time())
        for m in modes:
            for k in ("ddate", "iddate"):
                d = m[k]
                if d not in (0,) and not (0 < d <= now + 3600):
                    return None
        return modes
    except (struct.error, IndexError, ValueError):
        return None


def argyll_cal_remaining(mode: str, fast: bool = False) -> float | None:
    """Seconds of calibration validity Argyll itself still sees for a
    reading mode ('e','a','r' or full name), read from its host cache —
    None when nothing verifiable is on disk. Mirrors the invalidation
    rules of munki_imp_get_n_a_cals exactly:
      * every mode's dark cal expires DCALTOUT = 1 h after ddate
        (adaptive modes restore the interpolated dark cal, dated iddate)
      * reflective additionally needs the white cal (cfdate,
        WCALTOUT = 24 h) — and any non-emissive mode's white cal too"""
    import time as _time
    path = munki_cache_file()
    if path is None:
        return None
    modes = munki_cache_cal_dates(path)
    if not modes:
        return None
    m = str(mode).lower()[:1]
    # ColorMunki mode blocks, in file order (munki_imp.h mk_mode):
    # 0 reflective spot, 2 emissive non-adaptive, 4 emissive adaptive,
    # 7 ambient adaptive, 9 transmissive adaptive
    if m == "r":
        blk = modes[0]
    elif m == "a":
        blk = modes[7]
    else:
        blk = modes[2] if fast else modes[4]
    if blk is None:
        return None
    now = _time.time()
    windows: list[float] = []
    if blk["emiss"] or blk["trans"] or blk["reflective"]:
        dark = blk["iddate"] if blk["adaptive"] else blk["ddate"]
        if dark:
            windows.append(dark + 3600.0 - now)
    if blk["cfdate"]:
        if not blk["emiss"]:
            windows.append(blk["cfdate"] + 24 * 3600.0 - now)
    if not windows:
        return None
    left = min(windows)
    return left if left > 0.0 else 0.0


def cal_validity_s(mode: str) -> float:
    """Seconds a freshly completed calibration stays valid for a given
    reading mode — Argyll's own instrument rules (munki_imp.c / inst.h):
    the dark calibration the emission modes depend on is valid
    DCALTOUT = 1 h, the reflective white calibration WCALTOUT = 24 h.
    `mode` is 'r'/'a'/'e' or the full mode name."""
    return 24 * 3600.0 if str(mode).lower().startswith("r") else 3600.0


def cal_prompt(inst_type: str) -> str:
    if is_colormunki(inst_type):
        return ("Turn the wheel to position 2 (CALIBRATION)…")
    return ("Set the instrument to its CALIBRATION position")


def cal_running(inst_type: str) -> str:
    if is_colormunki(inst_type):
        return ("Calibrating… keep the wheel at position 2.")
    return ("Calibrating… keep the instrument in its calibration position.")


def cal_done(mode: str, inst_type: str) -> str:
    if is_colormunki(inst_type):
        if mode == "r":
            return ("Calibrated ✓ — turn the wheel to the SURFACE "
                    "position and hold the aperture flat against the "
                    "surface…")
        if mode == "a":
            return ("Calibrated ✓ — turn the wheel to the outer "
                    "diffuser slot (past OFF), diffuser facing the "
                    "room…")
        return ("Calibrated ✓ — move the wheel to position 3 "
                "(MEASURING) and point at the target…")
    if mode == "r":
        return ("Calibrated ✓ — place the sensor flat against the "
                "surface to measure…")
    if mode == "a":
        return ("Calibrated ✓ — hold the sensor up to measure the "
                "ambient light…")
    return ("Calibrated ✓ — aim the sensor at the target to measure…")


def wrong_cal_position(inst_type: str) -> str:
    if is_colormunki(inst_type):
        return ("Move the wheel to position 2 (CALIBRATION)…")
    return ("Instrument is not in its calibration position…")


class _Pending:
    """Spectral values being collected for the frame in progress."""
    def __init__(self, lo: float, hi: float, n: int):
        self.lo, self.hi, self.n = lo, hi, n
        self.wl = np.linspace(lo, hi, n)
        self.vals: list[float] = []
        self.xyz: tuple[float, float, float] | None = None
        self.lab: tuple[float, float, float] | None = None

    def ready(self) -> bool:
        """Frame ready: full spectrum collected, or an XYZ-only result
        from a device without spectral output (colorimeters)."""
        return (len(self.vals) >= self.n
                or (self.xyz is not None and not self.vals))


class SpotreadWorker(SourceWorker):
    """Drives `spotread` in a persistent subprocess through a PTY.

    Survives USB unplug/replug: if the instrument disappears the run is
    *not* stopped, the GUI reports "device disconnected" in its status
    bar (and any error text from spotread verbatim), and
    the worker keeps polling for the hardware, resuming automatically.
    """

    STALL_TIMEOUT = 45.0    # s without output mid-reading -> restart session

    def __init__(self, out: queue.Queue, mode: str = "e", kind: str = "c",
                 high_res: bool = True, fast: bool = False,
                 port: int | None = None, force_cal: bool = False):
        super().__init__(out)
        self.mode = mode       # "e" emissive | "a" ambient | "r" reflective
        self.kind = kind       # c CRT | l LCD (only advisory for emissive)
        self.port = port       # None = spotread's default (first found)
        self.high_res = high_res
        self.fast = fast       # -Y A: non-adaptive (short) integration time
        self.force_cal = force_cal   # skip Argyll's host calibration cache
        self.restart_now = threading.Event()  # "Force calibration" request
        # CALIBRATE-ALL: Argyll keeps a separate calibration state per
        # reading mode, and stock spotread only refreshes the mode it is
        # currently running — so switching modes costs another dance.
        # the all-modes calibration pass replaced the old per-demand
        # single-mode dance: every calibration trigger refreshes ALL
        # reading modes (one wheel-at-2 effort), keeping mode switches
        # prompt-free for the whole validity window afterwards.
        self.cal_pass_modes = None    # modes the instrument supports
        self._pass_since_frame = False  # pass loop guard (fall back to
        #                              the ordinary in-session dance if
        #                              a pass just ran without a frame)
        self._skip_cache_once = False  # one-shot: next start omits -N
        #                              (click on the CALIBRATION lamp)
        self._armed = False     # silent position probing after calibration
        self._app_trig = False  # the CURRENT reading was requested by us
        self._master = None
        self._out_q = None     # Windows pipe transport (reader thread)
        self._proc = None
        self.inst_type = ""     # set from spotread's "Instrument Type:"

    # -- worker API -----------------------------------------------------
    def respond(self, text: str):
        if os.name == "nt":
            p = self._proc
            if p is not None and p.stdin is not None:
                try:
                    p.stdin.write(text.encode())
                    p.stdin.flush()
                except (OSError, ValueError):
                    pass
            return
        m = self._master
        if m is not None:
            try:
                os.write(m, text.encode())
            except OSError:
                pass

    def restart_session(self):
        """End the current spotread run; the session respawns with the
        current settings (used by the 'Force calibration' menu item —
        Argyll caches calibration on the host, so a restart WITH
        force_cal is the only way to demand a fresh calibration)."""
        self.restart_now.set()

    # -- calibrate-all pass --------------------------------------------
    def cal_supported_modes(self) -> tuple[str, ...]:
        """Reading modes to cover in the all-modes calibration pass."""
        modes = tuple(self.cal_pass_modes) if self.cal_pass_modes \
            else ("e", "a", "r")
        # current mode first: return to readings as soon as possible
        return (self.mode,) + tuple(m for m in modes if m != self.mode)

    def _run_cal_pass(self):
        """One short spotread session per reading mode, each answering
        spotread's own calibration prompts: a single wheel-at-2 effort
        ends with every mode's calibration fresh in Argyll's host cache
        (~/.cache/ArgyllCMS), so mode switches afterwards resume with
        NO calibration at all (-N) within the validity windows."""
        names = {"e": "Emissive", "a": "Ambient", "r": "Reflective"}
        # position wording follows the instrument: only the ColorMunki
        # has a wheel; generic instruments just need the sensor at
        # their calibration position
        hint = ("keep the wheel at the CALIBRATION position"
                if is_colormunki(self.inst_type) else
                "keep the sensor at the instrument's calibration position")
        todo = self.cal_supported_modes()
        ok, failed = [], []
        for i, md in enumerate(todo):
            if self._stop.is_set():
                return
            self.emit_state(
                "calibrating",
                f"Calibrating every reading mode — {names.get(md, md)} "
                f"({i + 1}/{len(todo)}): {hint}…")
            if self._cal_one_mode(md):
                ok.append(md)
            else:
                failed.append(md)
            self._kill()
        self.emit_log("[cal-pass] fresh calibrations: "
                      + (", ".join(names.get(m, m) for m in ok) or "none")
                      + (f"; failed: "
                         + ", ".join(names.get(m, m) for m in failed)
                         if failed else ""))
        if failed:
            self.emit_state(
                "calibrating",
                "Some modes could not be calibrated ("
                + ", ".join(names.get(m, m) for m in failed)
                + ") — the current mode calibrates normally instead.")

    def _cal_one_mode(self, mode: str) -> bool:
        """Run one session in `mode` until its calibration is done (or
        turns out to be still valid and spotread goes straight to its
        menu). Auto-answers only calibration prompts; never readings."""
        try:
            self._spawn(mode, force=True)   # no -N: this session MUST
        except Exception as e:              # dance, or the pass lies
            self.emit_log(f"[cal-pass] spotread for mode '{mode}' failed "
                          f"to start: {e}")
            return False
        t0 = time.monotonic()
        cal_saw = False
        for line in self._lines():
            if line is None:
                if self._proc is None or self._proc.poll() is not None:
                    break          # instrument/spawn refused this mode
                if time.monotonic() - t0 > 180.0:
                    self.emit_log(f"[cal-pass] mode '{mode}' timed out — "
                                  "was the wheel at the calibration slot?")
                    return False
                continue
            if not line.strip():
                continue
            self.emit_log(line)
            s = line.strip()
            low = s.lower()
            if _RE_WHEEL.search(s):
                cal_saw = True
                self._stop.wait(4.0)    # let the user finish the wheel turn
                if self._stop.is_set():
                    return False
                self.respond("\n")      # acknowledge: do the dance
                # tell the GUI the dance is RUNNING (it drives the
                # CALIBRATION lamp's quarter-fill loading stage — the
                # header this mode started under carried only the
                # (k/N) stage fraction)
                self.emit_state("calibrating",
                                cal_running(self.inst_type or ""))
            elif cal_saw and "wrong position" in low:
                self._stop.wait(4.0)    # spotread re-prompted: try again
                if self._stop.is_set():
                    return False
                self.respond("\n")
            elif "Calibration complete" in s:
                return True
            elif _RE_MENU_LEAD.search(s) or _RE_MENU.search(s):
                # the menu arrived (the banner lead-in is instantaneous)
                # without any calibration demand. This is a FORCED
                # session (no -N): spotread must demand a dance — a
                # menu with no demand means the calibration never ran,
                # and the pass may not claim freshness it never earned.
                return False
            elif "hit any key" in low:
                self.respond("\n")      # generic acknowledgment prompt
        # session ended without a completed/valid calibration
        return False

    def cal_validity_s(self, mode: str | None = None) -> float:
        """Seconds the current (or given) reading mode's calibration
        stays valid after a completed calibration (delegate to the
        module function; instruments could override this per type)."""
        return cal_validity_s(mode if mode is not None else self.mode)

    def request_calibration(self):
        """Recalibrate NOW (the CALIBRATION lamp is clickable): end the
        session and restart it once WITHOUT Argyll's host cache (-N
        omitted one shot), so spotread demands a fresh calibration —
        which the driver then runs for every reading mode in one
        wheel-at-2 effort. The setting is one-shot: the next restarts
        resume normal cache reuse."""
        self._skip_cache_once = True
        self.restart_now.set()

    def _command(self, mode: str | None = None,
                force: bool = False) -> list[str]:
        md = mode if mode is not None else self.mode
        cmd = [spotread_path() or "spotread", "-v", "-s"]
        if md == "a":
            # Ambient: spotread XORs CCT/CRI display with the mode —
            # "-T" SUPPRESSES it there and its ABSENCE enables it
            # (spotread.c, "Xor ambient and doCCT"). Emissive and
            # reflective need -T to show it. Never pass -T for -a.
            pass
        else:
            cmd.append("-T")
        if not force and not self.force_cal and not self._skip_cache_once:
            # -N: don't demand a calibration at startup just because it's
            # a new session — reuse the calibration Argyll caches on the
            # host (~/.cache/ArgyllCMS/.mk_<serial>.cal) while it is
            # still valid (dark: 1 h, white: 24 h). spotread still checks
            # before EVERY reading and demands calibration when the cache
            # is genuinely stale — and instruments that don't support
            # this option simply ignore it (a warning, default behavior).
            cmd.append("-N")
        if self.port:
            cmd += ["-c", str(self.port)]   # select instrument N
        if md == "r":
            pass  # reflective (default)
        elif md == "a":
            cmd += ["-a"]
        else:
            cmd += ["-e"]
            if self.kind in ("c", "l"):
                cmd += ["-y", self.kind]
        if self.high_res:
            cmd.append("-H")
        if self.fast:
            cmd.append("-Y")   # non-adaptive integration: fastest readings
            cmd.append("A")
        return cmd

    def run(self):
        while not self._stop.is_set():
            if not spotread_path():
                self.emit_state("no-argyll", "ArgyllCMS spotread not found on this system")
                if self._stop.wait(3.0):
                    break
                continue
            # Device presence: the USB scan (sysfs) gates the spawn on
            # Linux, where it is trustworthy. On macOS system_profiler
            # does NOT reliably list Argyll instruments (the ColorMunki
            # enumerates with a vendor-specific interface class), so
            # an empty scan would mean nothing — there spotread itself
            # is the arbiter: spawn, and its launch/exit tells us.
            if sys.platform == "darwin":
                label = instrument_present() or "spotread"
            else:
                label = instrument_present()
                if not label:
                    self.emit_state("no-device",
                                    "Instrument not connected (USB)")
                    if self._stop.wait(2.0):
                        break
                    continue
            try:
                self._spawn()
            except Exception as e:
                self.emit_log(f"spotread failed to start: {e}")
                self.emit_state("reconnecting", str(e))
                if self._stop.wait(2.0):
                    break
                continue
            self.emit_state("starting", f"Session with {label}")
            try:
                self._session(label)
            finally:
                self._kill()
            if not self._stop.is_set():
                self.emit_state("reconnecting", "Instrument lost – retrying…")
                self._stop.wait(1.0)
        self._finish()

    def _spawn(self, mode: str | None = None, force: bool = False):
        if os.name == "nt":
            self._spawn_nt(mode, force)
            return
        import pty
        mfd, sfd = pty.openpty()
        p = subprocess.Popen(self._command(mode, force), stdin=sfd,
                             stdout=sfd, stderr=sfd, close_fds=True,
                             preexec_fn=lambda: os.setsid())
        os.close(sfd)
        # the one-shot "-N omitted" request (CALIBRATION lamp click, or
        # a forced calibration) is consumed by THIS spawn — unless the
        # caller demanded force (the all-modes pass: every per-mode
        # session must omit -N, or spotread would skip the calibration
        # dance and the pass would claim freshness it never earned)
        if not force:
            self._skip_cache_once = False
        self._master, self._proc = mfd, p

    def _spawn_nt(self, mode: str | None, force: bool):
        """Windows transport: there is no pty module, so spotread runs
        on plain pipes. Windows pipes are synchronous and give partial
        reads ONLY one byte at a time, and cannot be select()ed — so a
        daemon reader thread drains stdout into a queue that _lines()
        polls with a timeout (the same 'yield None every 0.2 s when
        idle' contract the PTY loop gives the session state machine)."""
        p = subprocess.Popen(
            self._command(mode, force),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,
                                  "CREATE_NEW_PROCESS_GROUP", 0))
        out_q: queue.Queue = queue.Queue()

        def _reader(fh, q):
            try:
                while True:
                    b = fh.read(1)          # byte-wise: never blocks
                    if not b:               # EOF — process ended
                        break
                    q.put(b)
            except (OSError, ValueError):
                pass
            finally:
                q.put(None)                 # EOF sentinel

        threading.Thread(target=_reader, args=(p.stdout, out_q),
                         daemon=True).start()
        if not force:
            self._skip_cache_once = False
        self._out_q, self._proc = out_q, p

    def _kill(self):
        p, m = self._proc, self._master
        if os.name == "nt" and p is not None:
            try:
                if p.stdin is not None:
                    p.stdin.write(b"q\n")
                    p.stdin.flush()
            except (OSError, ValueError):
                pass
            try:
                p.wait(timeout=2)
            except subprocess.TimeoutExpired:
                p.kill()
            self._proc = None
            return
        if p is not None and p.poll() is None:
            try:
                os.write(m, b"q\n")
            except OSError:
                try:
                    p.terminate()
                except Exception:
                    pass
            try:
                p.wait(timeout=2)
            except subprocess.TimeoutExpired:
                p.kill()
        if m is not None:
            try:
                os.close(m)
            except OSError:
                pass
        self._proc = None
        self._master = None

    # -- PTY / pipe line reader -------------------------------------------
    def _lines(self):
        """Yield lines from the transport until the process dies or we
        stop (None every ~0.2 s while idle, so the session state
        machine can run its timers)."""
        if os.name == "nt" and self._proc is not None \
                and getattr(self, "_out_q", None) is not None:
            q = self._out_q
            buf = b""
            while not self._stop.is_set() and self._proc is not None \
                    and self._proc.poll() is None:
                try:
                    data = q.get(timeout=0.2)
                except queue.Empty:
                    yield None
                    continue
                if data is None:        # EOF sentinel from the reader
                    break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    yield line.decode(errors="replace").rstrip("\r ")
            if buf:
                yield buf.decode(errors="replace").rstrip("\r ")
            return
        buf = b""
        while not self._stop.is_set() and self._proc and self._proc.poll() is None:
            r, _, _ = select.select([self._master], [], [], 0.2)
            if not r:
                yield None
                continue
            try:
                data = os.read(self._master, 4096)
            except OSError:
                break
            if not data:
                break
            buf += data
            # do not block mid-escape-sequence: only split on \r/\n
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                yield line.decode(errors="replace").rstrip("\r ")
        if buf:
            yield buf.decode(errors="replace").rstrip("\r ")

    # -- main session loop ------------------------------------------------
    def _session(self, label: str):
        pending: _Pending | None = None
        cct: float | None = None
        duv: float | None = None      # spotread's own Duv (CCT validity)
        cri: dict | None = None        # spotread's own Ra/R9 + validity flag
        tm30: dict | None = None       # spotread's own TM-30 Rf/Rg + flag
        last_spec: Spectrum | None = None   # last delivered frame (late meta)
        in_spectrum = False
        last_prompt = 0.0
        t_frame = time.monotonic()
        last_out = time.monotonic()      # last line received from spotread
        gone_since = None                # device-absent since (for fast kill)
        pos_hint = None                  # spotread's own wheel hint, if any
        in_cal = False                   # currently in the calibration dance
        menu_wait = None                 # result sent, closing banner pending
        last_probe_hint = None           # spotlight hint already relayed
        first_menu = True                # banner not seen yet this session
        cal_danced = False               # spotread asked for a calibration?

        def emit_frame():
            nonlocal pending, cct, duv, cri, tm30, last_spec, t_frame
            if pending is not None and (
                    pending.ready() or (
                        pending.xyz is not None and not pending.vals)):
                # normal spectral frame — or an XYZ-only frame from an
                # instrument with no spectral output (colorimeters)
                if pending.vals:
                    vals = np.array(pending.vals[:pending.n])
                    spec = Spectrum(pending.wl.copy(), vals, pending.xyz)
                else:
                    spec = Spectrum(np.array([]), np.array([]),
                                    pending.xyz)
                spec.timestamp = datetime.now().strftime("%H:%M:%S")
                spec.source = label
                if cct is not None:
                    spec.meta_cct = cct        # type: ignore[attr-defined]
                if duv is not None:
                    spec.meta_duv = duv       # type: ignore[attr-defined]
                if cri is not None:
                    spec.meta_cri = cri       # type: ignore[attr-defined]
                if tm30 is not None:
                    spec.meta_tm30 = tm30     # type: ignore[attr-defined]
                if pending.lab is not None:
                    spec.meta_lab = pending.lab  # type: ignore[attr-defined]
                # was this reading requested by us, or by spotread's
                # hardware trigger (the instrument's physical switch)?
                spec.hw_triggered = getattr(pending, "hw_triggered", False)
                dt = time.monotonic() - t_frame
                t_frame = time.monotonic()
                self._armed = False     # position found: probes are over
                # a frame proves the session (and any all-modes pass that
                # led to it) succeeded: re-arm the pass trigger
                self._pass_since_frame = False
                self.emit_frame(spec)
                self.emit_state("measuring",
                                f"Last reading took {dt:.1f} s" if dt > 0.4 else "")
                last_spec = spec      # for the late CCT/Duv/CRI patch below
            pending, cct, duv, cri, tm30 = None, None, None, None, None
            in_spectrum = False

        def note_meta_late():
            # spotread prints its "CCT = … (Duv …)", "Color Rendering
            # Index" and "IES TM-30-15" lines AFTER the result block —
            # i.e. possibly after the frame was already
            # (early-)delivered. They belong to that same reading, so
            # patch the emitted Spectrum object in place; the GUI reads
            # these via getattr on each refresh.
            # Guard: only patch while the closing banner for this reading
            # is still pending (menu_wait), never across readings.
            nonlocal last_spec
            if menu_wait is not None and last_spec is not None:
                if cct is not None:
                    last_spec.meta_cct = cct  # type: ignore[attr-defined]
                if duv is not None:
                    last_spec.meta_duv = duv  # type: ignore[attr-defined]
                if cri is not None:
                    last_spec.meta_cri = cri  # type: ignore[attr-defined]
                if tm30 is not None:
                    last_spec.meta_tm30 = tm30  # type: ignore[attr-defined]

        for line in self._lines():
            # "Force calibration": the user asked for a fresh calibration,
            # so end this spotread run — the respawn loop restarts it
            # with the current (force_cal) settings.
            if self.restart_now.is_set():
                self.restart_now.clear()
                self.emit_state("restart",
                                "Restarting session for a forced calibration…")
                self._kill()
                break
            if line is None:
                # nothing arrived: keep an eye on the USB device
                if self._proc is None or self._proc.poll() is not None:
                    break
                stall = time.monotonic() - last_out
                # -- fast path 1: USB device gone during a transaction ------
                # spotread usually exits on its own, but a hung child keeps
                # the session dead; kill it quickly so the respawn loop can
                # reconnect (small grace so connector jiggles don't cost
                # the calibration). macOS keeps OUT of this entirely: the
                # scan cannot see the vendor-class instrument (above), so
                # "scan empty" carries no information — only spotread's
                # own exit says the device is gone.
                if sys.platform != "darwin" \
                        and instrument_present() is None:
                    if gone_since is None:
                        gone_since = time.monotonic()
                        self.emit_state("no-device",
                                        "Instrument not connected (USB)")
                    if pending is not None and \
                            time.monotonic() - gone_since > 2.0:
                        self.emit_log(f"[driver] USB gone while reading — "
                                      "killing wedged spotread")
                        self.emit_state("restart",
                                        "Instrument connection lost — "
                                        "restarting (needs new calibration)")
                        self._kill()
                        break
                    continue
                gone_since = None
                # -- fast path 2b: reading delivered but spotread never
                # returned to the menu (wedged USB read) -> restart now
                if menu_wait is not None and \
                        time.monotonic() - menu_wait > 8.0:
                    self.emit_log("[driver] reading delivered but spotread "
                                  "never returned to the menu — restarting "
                                  "session")
                    self.emit_state("restart",
                                    "Instrument stopped responding — "
                                    "restarting (needs new calibration)")
                    self._kill()
                    break
                # -- fast path 2: silence mid-reading -----------------------
                # spotread prints nothing during a long dark reading (up to
                # ~31 s observed), so 45 s of silence means it is wedged:
                # kill the child; run() respawns the session (which
                # re-calibrates automatically).
                if pending is not None and stall > self.STALL_TIMEOUT:
                    self.emit_log(f"[driver] spotread stalled {stall:.0f}s "
                                  "mid-reading with no output — restarting "
                                  "session")
                    self.emit_state("restart",
                                    "Instrument stopped responding — "
                                    "restarting (needs new calibration)")
                    self._kill()
                    break
                idle = time.monotonic() - last_prompt
                if (self.wants_read.is_set() or self.continuous.is_set()) \
                        and pending is None:
                    if idle > 1.0:
                        # spotread waits for a key: take the reading now
                        self.wants_read.clear()
                        self._app_trig = True
                        self.respond("\n")
                        last_prompt = time.monotonic()
                elif idle > 5.0 and pending is None and not in_cal:
                    # periodic heartbeat so the log shows why we are idle
                    self.emit_log(f"[driver] idle at menu; wants_read="
                                  f"{int(self.wants_read.is_set())} "
                                  f"continuous={int(self.continuous.is_set())} "
                                  f"last_menu={idle:.1f}s ago")
                    last_prompt = time.monotonic() - 1.0
                continue
            last_out = time.monotonic()
            if not line.strip():
                continue
            self.emit_log(line)
            s = line.strip()

            # ---- instrument identity (spotread's own type string) ----------
            m = _RE_INSTTYPE.search(s)
            if m:
                kind = m.group(1).strip()
                if kind and kind != self.inst_type:
                    self.inst_type = kind
                    self.emit_state("instrument", kind)
                continue

            # ---- spectrum header (first line of a reading) ------------------
            m = _RE_SPECTRUM_HDR.search(s)
            if m:
                last_spec = None            # new reading: stop patching
                pending = _Pending(float(m.group(1)), float(m.group(2)), int(m.group(3)))
                # spotread reached the menu and we sent no key, yet a
                # reading began: this can only be spotread's hardware
                # trigger (the instrument's physical switch). Tag the
                # frame so the GUI can treat button presses (isolated)
                # differently from a held/defective switch (a stream).
                pending.hw_triggered = not self._app_trig  # type: ignore[attr-defined]
                self._app_trig = False
                in_spectrum = True
                continue
            if in_spectrum and pending is not None:
                # the values arrive as one long comma-separated line (possibly
                # wrapped by the terminal into several lines)
                try:
                    pending.vals.extend(float(x) for x in s.split(",")
                                        if x.strip() not in ("", ".."))
                except ValueError:
                    in_spectrum = False
                    continue
                if pending.ready():
                    in_spectrum = False
                continue
            # ---- result block ------------------------------------------------
            m = _RE_RESULT.search(s)
            if m:
                if pending is None:
                    pending = _Pending(380.0, 730.0, 106)   # unseen header fallback
                xyz = tuple(float(g) for g in m.groups())
                # the same line can also carry the D50 Lab and the CCT
                m2 = _RE_LAB.search(s)
                if m2:
                    pending.lab = tuple(float(g) for g in m2.groups())
                # Argyll prints reflective XYZ on a 0..100 percent scale
                # (its own Lab corroborates: L* matches Y/100), while
                # emissive/ambient are absolute cd/m². Decide with the
                # instrument's own Lab when available (a dark surface's
                # percent values can be < 1.6), else fall back to the
                # physical plausibility of a reflectance factor.
                if self.mode == "r":
                    if pending.lab is not None:
                        L = pending.lab[0]
                        f = ((L + 16.0) / 116.0) ** 3   # Lab-implied Y
                        if abs(xyz[1] / 100.0 - f) < abs(xyz[1] - f):
                            xyz = tuple(v / 100.0 for v in xyz)
                    elif max(xyz) > 1.6:
                        xyz = tuple(v / 100.0 for v in xyz)
                pending.xyz = xyz
                # EARLY DELIVERY: spotread sometimes wedges right after
                # the result block and never prints the closing menu
                # banner. Hand the reading over the moment it is complete
                # instead of losing it (the GUI recomputes the CCT from
                # XYZ when spotread's is missing).
                if pending is not None and pending.ready():
                    emit_frame()
                    menu_wait = time.monotonic()
                continue
            m = _RE_LAB.search(s)
            if m:
                if pending is not None:
                    pending.lab = tuple(float(g) for g in m.groups())
                continue
            # TM-30 FIRST: its line prints its own "CCT = …" summary,
            # which the loose CCT regex below would otherwise swallow —
            # and the CCT/Duv worth showing with the reading are the
            # ones spotread computed for the READING, not TM-30's
            # internal reference comparison.
            m = _RE_TM30.search(s)
            if m:
                try:
                    tm30 = {"rf": float(m.group(1)),
                            "rg": float(m.group(2)),
                            "flag": m.group(3)}
                except ValueError:
                    pass
                else:
                    note_meta_late()
                continue
            m = _RE_CCT.search(s)
            if m:
                try:
                    cct = float(m.group(1))
                except ValueError:
                    pass
                md = _RE_DUV.search(s)
                if md:
                    try:
                        duv = float(md.group(1))
                    except ValueError:
                        pass
                note_meta_late()
                continue
            m = _RE_CRI.search(s)
            if m:
                cri = {"ra": float(m.group(1)),
                       "r9": (float(m.group(2))
                              if m.group(2) is not None else None),
                       "flag": m.group(3)}   # None | "Caution" | "Invalid"
                note_meta_late()
                continue
            # menu banner lead-in: the instant signal that the session
            # reached its menu; "calibration still valid" is decided
            # here (the closing "Hit ESC…" line can lag behind by
            # minutes). The reading flow itself still waits for the
            # closing line, as it always has.
            if _RE_MENU_LEAD.search(s):
                if first_menu and not cal_danced:
                    self.emit_state("measuring",
                                    "Calibration still valid — ready to "
                                    "read (no new calibration needed).")
                    first_menu = False
                continue
            # ---- reading complete: the menu banner is printed again ---------
            if _RE_MENU.search(s):
                last_prompt = time.monotonic()
                menu_wait = None
                emit_frame()
                # Session start without any calibration demand: the menu
                # arrived on Argyll's host cache alone — tell the GUI so
                # it can mark the calibration as valid (green border).
                if first_menu and not cal_danced:
                    self.emit_state("measuring",
                                    "Calibration still valid — ready to "
                                    "read (no new calibration needed).")
                first_menu = False
                # continuous mode (or a pending single request) -> next reading
                read_next = self.continuous.is_set() or self.wants_read.is_set()
                self.wants_read.clear()
                if read_next:
                    self._app_trig = True
                    self.respond("\n")
                continue
            # ---- calibration prompt: handled automatically --------------------
            if _RE_WHEEL.search(s):
                # CALIBRATE-ALL: a calibration is due — refresh EVERY
                # reading mode's calibration in one wheel-at-2 effort
                # instead of just this one (always: calibrating all
                # modes keeps mode switches prompt-free for the whole
                # validity window; one pass per reading: if a pass just
                # ran and no frame came of it, fall through to the
                # ordinary in-session dance — no infinite loop).
                if not self._pass_since_frame:
                    self._pass_since_frame = True
                    self.emit_log("[driver] calibration due — calibrating "
                                  "ALL reading modes before resuming")
                    self._kill()           # release the device for the pass
                    self._run_cal_pass()
                    if self._stop.is_set():
                        return
                    self.emit_state("restart",
                                    "All-mode calibration done — resuming…")
                    return               # run() respawns; caches are fresh
                # STAGE 1: spotread wants the wheel at the calibration
                # slot. If it is not there yet, this is the message the
                # user needs; after we acknowledge, slot-2 feedback comes
                # from the "calibrating" state below.
                in_cal = True
                cal_danced = True
                menu_wait = None
                self.wants_read.clear()
                self.emit_state("calibrating", cal_prompt(self.inst_type))
                self._stop.wait(4.0)          # give the user time to turn it
                if self._stop.is_set():
                    return
                self.respond("\n")           # acknowledge the prompt
                # STAGE 2: wheel is (or should be) at the calibration slot
                # now — if it isn't, spotread re-issues the prompt above.
                self.emit_state("calibrating",
                                cal_running(self.inst_type))
                continue
            if "Calibration complete" in s:
                in_cal = False
                # ARMED PROBING: only when a recording is actually wanted
                # (START active or a SINGLE pending) — a mere calibration
                # after a reconnect must NOT auto-measure: the trace the
                # user was looking at stays untouched until they ask.
                # While armed the driver silently retries a reading every
                # few seconds until the wheel/target is in the right
                # place; failed probes are logged but never shown, and
                # the first success delivers the reading and disarms.
                if self.continuous.is_set() or self.wants_read.is_set():
                    self._armed = True
                    self.wants_read.set()
                # STAGE 3: calibration done — guide the user to the
                # position the current reading mode needs.
                self.emit_state("measuring",
                                cal_done(self.mode, self.inst_type))
                # NOTE: no auto-reading here — calibration is automatic,
                # but the first reading waits for START/SINGLE.
                continue
            # spotread tells us which wheel slot it wants — remember it
            mh = re.search(r"\(Sensor should be in (.+?) position\)", s)
            if mh:
                pos_hint = mh.group(1).strip()
                continue
            # ---- failed reading: guide the user and retry patiently ----------
            m = _RE_FAIL.search(s)
            if m:
                reason = m.group(1).strip()
                pending, cct, in_spectrum = None, None, False
                menu_wait = None
                if self._armed and (
                        _RE_WRONGPOS.search(reason)
                        or "misread" in reason.lower()
                        or "saturated" in reason.lower()
                        or "dark" in reason.lower()
                        or "black" in reason.lower()):
                    # armed probe failed: not in position yet — stay
                    # quiet in the GUI and try again in a few seconds,
                    # but do relay (once) spotread's own slot hint
                    if pos_hint and pos_hint != last_probe_hint:
                        last_probe_hint = pos_hint
                        self.emit_state("guide",
                                        f"Wheel not in the {pos_hint} "
                                        "position yet — probing every few "
                                        "seconds…")
                    self.emit_log("!! [armed probe] " + reason)
                    self._stop.wait(4.0)
                    if self._stop.is_set():
                        return
                    self._app_trig = True
                    self.respond("\n")
                    continue
                if "calibration position" in reason.lower():
                    msg = wrong_cal_position(self.inst_type)
                elif _RE_WRONGPOS.search(reason):
                    # reflective/ambient modes need a different wheel slot
                    # than emissive — relay spotread's own hint verbatim
                    msg = (f"Move the wheel: sensor should be in the "
                           f"{pos_hint or 'measuring'} position…")
                elif "saturated" in reason.lower():
                    msg = "Sensor saturated — move the instrument back " \
                          "from the light source…"
                elif "dark" in reason.lower() or "black" in reason.lower():
                    msg = "Reading too dark — try a brighter target…"
                else:
                    msg = f"Reading failed ({reason}) — retrying…"
                self.emit_log("!! " + msg)
                self.emit_state("guide", msg)
                self._stop.wait(4.0)           # patient auto-retry
                if self._stop.is_set():
                    return
                self._app_trig = True
                self.respond("\n")
                continue
            if "retry" in s.lower() and ("hit" in s.lower() or "esc" in s.lower()):
                # "Hit Esc or Q to give up, any other key to retry:"
                continue
            if "Place instrument on spot" in s:
                last_prompt = time.monotonic()
                continue
            low = s.lower()
            if "hit esc or q" in low or "esc or q to exit" in low:
                # key-press hints printed with every prompt: harmless
                continue
            # ---- fatal errors: let the outer loop restart --------------------
            if "claiming usb port" in low and "failed" in low:
                self.emit_state("reconnecting",
                                "USB port busy – a stale spotread process may "
                                "still hold the instrument")
                return
            if ("communications failure" in low) or ("failed with" in low) or \
                    ("failed to init" in low):
                self.emit_state("reconnecting", s)
                return


# ---------------------------------------------------------------------------
# Simulator (fluorescent lamp with Hg lines + noise + slow drift)
# ---------------------------------------------------------------------------
SimWL = np.arange(380.0, 731.0, 2.5)


class SimulatorWorker(SourceWorker):
    """Generates plausible spectra so the whole application can be exercised
    without hardware."""

    PROFILES = ("fluorescent", "LED", "incandescent", "dark")

    def __init__(self, out: queue.Queue):
        super().__init__(out)
        self.profile = "fluorescent"
        self.dwell = 1.5

    def _spectrum(self, t: float) -> Spectrum:
        wl = SimWL
        rng = np.random.default_rng()
        drift = 1.0 + 0.03 * np.sin(t / 7.0) + rng.normal(0, 0.01)
        if self.profile == "fluorescent":
            tri = np.clip(1.0 - np.abs(wl - 590.0) / 200.0, 0, 1) ** 2.5
            v = 0.15 + 0.3 * tri + 0.06 * np.exp(-((wl - 450) / 70) ** 2)
            for lw, amp in ((404.7, 0.95), (435.8, 1.35), (546.1, 1.6),
                            (577.0, 1.15), (579.1, 1.05), (615.0, 0.7),
                            (650.0, 0.4)):
                v = v + amp * np.exp(-((wl - lw) / 1.8) ** 2)
        elif self.profile == "LED":
            v = 0.9 * np.exp(-((wl - 452.0) / 16.0) ** 2) + \
                0.75 * np.exp(-((wl - 555.0) / 95.0) ** 2)
        elif self.profile == "incandescent":
            v = np.clip((950.0 - wl) / 500.0, 0.02, None) ** 1.6
        else:  # dark
            v = np.zeros_like(wl) + rng.normal(0.003, 0.004, len(wl))
        v = np.clip(v * drift + rng.normal(0, 0.004, len(wl)), 0.0, None)
        spec = Spectrum(wl.copy(), v)
        spec.timestamp = datetime.now().strftime("%H:%M:%S")
        spec.source = f"Simulator ({self.profile})"
        spec.xyz = approx_xyz_from_spectrum(spec)
        return spec

    def respond(self, text):
        pass

    def run(self):
        while not self._stop.is_set():
            if not (self.wants_read.is_set() or self.continuous.is_set()):
                self.wants_read.wait(0.2)
                continue
            self.wants_read.clear()
            self.emit_state("measuring", "Reading…")
            if self._stop.wait(self.dwell):
                break
            try:
                self.emit_frame(self._spectrum(time.monotonic()))
            except Exception as e:
                self.emit_log(f"simulator error: {e}")
        self._finish()


# ---------------------------------------------------------------------------
# File formats
# ---------------------------------------------------------------------------
def save_csv(path: str, spec: Spectrum, zeroed: bool = False):
    import csv
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["# spectro-live spectrum", spec.timestamp or "",
                     spec.source or ""])
        if zeroed:
            # explicit flag: the values are ZERO-CORRECTED (a stored
            # zero/background was subtracted) and XYZ recomputed from
            # the corrected curve, not the raw instrument reading
            wr.writerow(["# zero_corrected", "1"])
        if spec.xyz is not None:
            X, Y, Z = spec.xyz
            x, y = xy_from_xyz(spec.xyz)
            cct = getattr(spec, "meta_cct", None) or cct_from_xyz(spec.xyz)
            lab = getattr(spec, "meta_lab", None)
            rgb, oog, over, black = srgb_preview(spec.xyz)
            g = gamut_report(spec.xyz)
            where = ("within sRGB" if g["sRGB"] else
                     "within Adobe RGB" if g["Adobe RGB"] else
                     "within Display P3" if g["Display P3"] else
                     "within Rec. 2020" if g["Rec. 2020"] else
                     "outside Rec. 2020")
            wr.writerow(["# XYZ", f"{X:.6f}", f"{Y:.6f}", f"{Z:.6f}"])
            wr.writerow(["# xy_chromaticity", f"{x:.4f}", f"{y:.4f}"])
            wr.writerow(["# CCT_K", f"{cct:.0f}"])
            if lab is not None:
                wr.writerow(["# Lab_D50", f"{lab[0]:.3f}", f"{lab[1]:.3f}",
                             f"{lab[2]:.3f}"])
            wr.writerow(["# sRGB_preview", f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"])
            wr.writerow(["# sRGB_gamut", where])
        wr.writerow(["wavelength_nm", "intensity"])
        for w, v in zip(spec.wavelengths, spec.values):
            wr.writerow([f"{w:.3f}", f"{v:.6f}"])
        # peak list (strongest first) — written into every trace-A CSV
        # (used to be a separate "Write peaks to log" button)
        peaks = find_peaks(spec, prominence=0.05)
        wr.writerow(["# peaks_prominence", "5%", f"count={len(peaks)}"])
        for i, (w, v) in enumerate(peaks, 1):
            info = peak_info(spec, w - 12.0, w + 12.0)
            wr.writerow([f"# peak_{i}", f"{w:.1f}", f"E={v:.4f}",
                         f"FWHM_nm={info['fwhm']:.1f}",
                         f"centroid_nm={info['centroid']:.1f}"])


def _numeric_pairs(lines):
    wl, val = [], []
    for line in lines:
        parts = line.replace(",", " ").replace(";", " ").split()
        nums = []
        try:
            for p in parts[:2]:
                nums.append(float(p))
        except ValueError:
            continue
        if len(nums) == 2 and 200.0 <= nums[0] <= 2000.0:
            wl.append(nums[0])
            val.append(nums[1])
    return np.array(wl), np.array(val)


def load_csv(path: str) -> Spectrum:
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    wl, val = _numeric_pairs(lines)
    if len(wl) < 10:
        raise ValueError("no spectral data found in file")
    order = np.argsort(wl)
    spec = Spectrum(wl[order], val[order],
                    label=f"File {os.path.basename(path)}")
    # ---- header block written by save_csv --------------------------------
    # The '#' comment lines carry the ABSOLUTE anchor of the reading:
    # the instrument-measured XYZ (and the meta values derived from
    # it). Restoring them keeps every absolute metric — photometric Y,
    # radiometric power, CCT, chromaticity, Lab — identical to what
    # the live reading showed instead of recomputing them from the
    # arbitrary-unit curve, which would silently change their meaning.
    xyz = None
    meta_lab = meta_cct = None
    for line in lines:
        if not line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        # parts[0] == '#'
        try:
            if len(parts) >= 5 and parts[1] == "XYZ":
                xyz = tuple(float(v) for v in parts[2:5])
            elif len(parts) >= 3 and parts[1] == "CCT_K":
                meta_cct = float(parts[2])
            elif len(parts) >= 5 and parts[1] == "Lab_D50":
                meta_lab = (float(parts[2]), float(parts[3]),
                            float(parts[4]))
        except ValueError:
            pass
    spec.xyz = xyz
    if meta_cct is not None:
        spec.meta_cct = meta_cct
    if meta_lab is not None:
        spec.meta_lab = meta_lab
    return spec


def load_argyll(path: str) -> Spectrum:
    """Load an Argyll .sp (or CGATS with SPECTRAL_NM) file."""
    text = open(path, encoding="utf-8", errors="replace").read()
    m = re.search(r"SPECTRAL_NM\s+((?:[0-9.]+\s*)+)", text)
    if m:
        wl = np.array([float(x) for x in m.group(1).split()])
        rows = re.search(r"SPECTRAL_[A-Z_]+\s+((?:[0-9.eE+-]+\s+){%d})" % len(wl),
                         text[m.end():])
        if rows:
            vals = np.array([float(x) for x in rows.group(1).split()][:len(wl)])
            return Spectrum(wl, vals, label=f"File {os.path.basename(path)}")
    return load_csv(path)
