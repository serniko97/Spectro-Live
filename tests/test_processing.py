"""Processing / colourimetry / persistence tests (headless, no Qt).

Run:  .venv/bin/python tests/test_processing.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                     # noqa: E402
trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy 1.x on macOS has no trapezoid()
from spectrov import sources, processing as P           # noqa: E402
from spectrov.processing import Spectrum                # noqa: E402

FAILURES = []


def check(name, cond, extra=""):
    s = "ok  " if cond else "FAIL"
    print(f"  [{s}] {name}{('  ' + extra) if extra and not cond else ''}")
    if not cond:
        FAILURES.append(name)


def test_srgb_preview_max_brightness():
    # dark warm surface: preview must be bright, hue preserved vs white
    rgb, oog, over, black = P.srgb_preview((0.011675, 0.009596, 0.003217))
    check("dark surface: brightest channel 255",
          max(rgb) == 255, str(rgb))
    check("dark surface: warm hue (r>g>b)", rgb[0] > rgb[1] > rgb[2])
    x1, y1 = P.xy_from_xyz((0.011675, 0.009596, 0.003217))
    # same chromaticity scaled up: identical preview colour
    rgb2, *_ = P.srgb_preview((0.11675, 0.09596, 0.03217))
    check("chromaticity-only: 10x scaling changes nothing",
          rgb == rgb2, f"{rgb} vs {rgb2}")
    # white stays white
    rgbw, *_ = P.srgb_preview((0.9642, 1.0, 0.8251))
    check("white paper: stays white", rgbw == (255, 255, 255), str(rgbw))
    # pure black: flagged, not amplified
    rgbz, *_z2, blackflag = P.srgb_preview((0.0, 0.0, 0.0))
    check("pure black: no boost", max(rgbz) <= 10 or blackflag,
          str(rgbz))


def test_gamut_containment_ordering():
    # Display-P3 red primary: outside sRGB, inside P3 and Rec. 2020
    g = P.gamut_report((0.68, 0.32, 0.15))
    check("P3 red outside sRGB, inside P3/2020",
          (not g["sRGB"]) and g["Display P3"] and g["Rec. 2020"], str(g))
    w = P.gamut_report((0.3333, 0.3333, 0.3334))   # flat grey
    check("grey inside sRGB", all([w["sRGB"], w["Adobe RGB"],
                                   w["Display P3"], w["Rec. 2020"]]))
    # containment must be monotone: sRGB -> Adobe -> P3 -> 2020
    any_c = P.gamut_report((0.64, 0.33, 0.30))
    inside = [k for k in ("sRGB", "Adobe RGB", "Display P3",
                          "Rec. 2020") if any_c[k]]
    check("gamut set is a contiguous chain",
          ("sRGB" in inside) <= ("Adobe RGB" in inside) <=
          ("Display P3" in inside) <= ("Rec. 2020" in inside), str(any_c))
    # luminance independence: same chromaticity, 100x brighter reports same
    check("gamut independent of luminance",
          P.gamut_report((0.73, 0.27, 0.10)) ==
          P.gamut_report((7.3, 2.7, 1.0)) ==
          P.gamut_report((0.073, 0.027, 0.010)))


def test_cct_and_xy():
    # D65 approx
    cct = P.cct_from_xyz((0.9505, 1.0, 1.089))
    check("D65-ish XYZ yields ~6500K", 6000 < cct < 7000, f"{cct:.0f}")
    x, y = P.xy_from_xyz((0.9505, 1.0, 1.089))
    check("D65 chromaticity", abs(x - 0.3127) < 0.005 and
          abs(y - 0.3290) < 0.005, f"{x:.4f},{y:.4f}")
    warm = P.cct_from_xyz((0.44751, 0.40742, 0.14497))   # ~3000K-ish norm
    check("warm light yields lower CCT than D65", warm < cct,
          f"warm {warm:.0f}")


def test_lab_implied_y_arbiter_math():
    """The percent-scale decision the driver makes: Lab L* implies Y;
    the closer of Y and Y/100 to the printed value wins."""
    cases = [
        # (printed X Y Z — Argyll's x100 percent scale, Lab L*, expected)
        ((0.40, 30.87, 0.25), 62.4, True),        # L*62.4 ~ Y=0.3087 real
        ((0.9016, 100.0, 0.324), 100.0, True),   # white tile: printed Y%=100
        ((0.0117, 0.9596, 0.0032), 8.65, True),  # dark wall: printed Y%<1
        ((0.0002, 0.0002, 0.0002), 0.2, False),  # tiny values only
    ]
    for xyz, L, expect in cases:
        f = ((L + 16.0) / 116.0) ** 3
        divide = abs(xyz[1] / 100.0 - f) < abs(xyz[1] - f)
        check(f"arbiter L*={L}: divide-by-100=={expect}",
              divide == expect,
              f"printed Y={xyz[1]}, implied Y={f:.4f}")


def test_csv_roundtrip_with_metadata():
    spec = Spectrum(np.linspace(380, 730, 106),
                    np.abs(np.sin(np.linspace(380, 730, 106))) * 0.3,
                    (0.011675, 0.009596, 0.003217))
    spec.meta_lab = (8.65, 8.54, 8.87)
    spec.timestamp = "01:30:00"
    spec.source = "ColorMunki Photo"
    p = "/tmp/opencode/testspec.csv"
    sources.save_csv(p, spec)
    txt = open(p).read()
    check("CSV header present", txt.startswith("# spectro-live spectrum"))
    for key in ("# XYZ", "# xy_chromaticity", "# CCT_K", "# Lab_D50",
                "# sRGB_preview", "# sRGB_gamut", "# peaks_prominence"):
        check(f"CSV row {key} written", key in txt)
    check("CSV peak rows written with the trace",
          "# peak_1" in txt and "FWHM_nm=" in txt and
          "centroid_nm=" in txt, txt[-400:])
    back = sources.load_argyll(p)
    check("CSV reloads: 106 points", len(back.wavelengths) == 106,
          str(len(back.wavelengths)))
    check("CSV reloads: values intact",
          np.allclose(back.values, spec.values, atol=1e-6))
    os.remove(p)


def test_reflective_honest_preview():
    """Reflective swatch: brightness = reflectance; boost only when a
    channel would exceed 255 (mirrors the qtapp swatch branch)."""
    # white surface, 100 % reflectance
    white = (0.9642, 1.0, 0.8251)
    lin = P.linear_channels(white)
    check("100 % white fits without boost (0.2% tolerance)",
          float(lin.max()) <= 1.002, f"max lin {float(lin.max()):.4f}")
    rgb, oog, *_ = P.srgb_preview(white, normalize=False)
    check("100 % white shows as rgb(255,255,255)", rgb == (255, 255, 255),
          str(rgb))
    # 50 % neutral grey: emits half the photons -> ~code 188 (not 127)
    grey = tuple(v * 0.5 for v in white)
    rgb, oog, *_ = P.srgb_preview(grey, normalize=False)
    check("50 % grey emits half the light (~188, not 127)",
          185 <= rgb[0] <= 191, str(rgb))
    # dark warm wall (the live session colour): honestly dark now
    wall = (0.041626, 0.035747, 0.011519)
    lin = P.linear_channels(wall)
    check("live wall: no boost needed", float(lin.max()) <= 1.002)
    rgb, oog, *_ = P.srgb_preview(wall, normalize=False)
    check("live wall: dark honest preview (kept hue order)",
          rgb[0] > rgb[1] > rgb[2] and rgb[0] < 100, str(rgb))
    # a too-bright chromaticity WOULD exceed 255 -> boost fallback
    hot = (2.4, 1.5, 0.3)
    lin = P.linear_channels(hot)
    check("out-of-range case detected", float(lin.max()) > 1.0)
    rgb, oog, *_ = P.srgb_preview(hot)          # boosted branch
    check("boost: brightest channel at 255", max(rgb) == 255, str(rgb))


def test_settings_roundtrip():
    """Mode/hires/fast choices persist via settings.json."""
    import tempfile
    import spectrov.qtapp as Q
    tmp = tempfile.mkdtemp(dir="/tmp/opencode")
    old_path = Q.MainWindow._CFG_PATH
    path = os.path.join(tmp, "settings.json")
    Q.MainWindow._CFG_PATH = path
    try:
        import types
        fake = types.SimpleNamespace(_CFG_PATH=path, mode_name="Ambient",
                                     high_res=False, fast=True,
                                     force_cal=True, curve_style="steps",
                                     x_mode="wl", clip_unreliable=True)
        # call the real save/load pair with a stub `self`
        Q.MainWindow._save_settings(fake)
        check("settings: file written", os.path.exists(path))
        back = Q.MainWindow._load_settings(
            types.SimpleNamespace(_CFG_PATH=path))
        check("settings: mode persisted", back.get("mode") == "Ambient",
              str(back))
        check("settings: high_res persisted",
              back.get("high_res") is False, str(back))
        check("settings: fast persisted", back.get("fast") is True,
              str(back))
        check("settings: curve style persisted",
              back.get("curve_style") == "steps", str(back))
    finally:
        Q.MainWindow._CFG_PATH = old_path


def test_argyll_cache_countdown():
    """The calibration countdown must be reconstructed from Argyll's
    OWN host cache (.mk_<serno>.cal): the file stores the real
    last-dance dates per mode, and validity is judged against the
    actual calibration age (munki_imp_get_n_a_cals), not per session.
    A synthetic cache exercises the parser without needing hardware."""
    import struct
    import tempfile
    import time
    from spectrov import sources

    tmp = tempfile.mkdtemp(dir="/tmp/opencode")
    old_xdg = os.environ.get("XDG_CACHE_HOME")
    old_home = os.environ.get("HOME")
    os.environ["XDG_CACHE_HOME"] = tmp
    os.environ["HOME"] = tmp      # munki_cache_dirs fallback
    os.makedirs(os.path.join(tmp, "ArgyllCMS"), exist_ok=True)
    now = int(time.time())
    nraw, nwav1, nwav2 = 36, 36, 101
    out = bytearray()
    out += struct.pack("<ii", 1, 999)          # argyllversion, ss
    out += b"SYNTH0001".ljust(17, b"\0")       # serial
    out += struct.pack("<iii", nraw, nwav1, nwav2)

    def blk(emiss, trans, refl, amb, adaptive, ddate, cfdate, iddate):
        b = bytearray()
        b += struct.pack("<8i", emiss, trans, refl, 0, 0, amb, 0,
                         adaptive)
        b += struct.pack("<i", 0)              # gainmode
        b += struct.pack("<d", 1.0)            # inttime
        b += struct.pack("<i", 1)              # dark_valid
        b += struct.pack("<q", ddate)
        b += struct.pack("<d", 0.7)
        b += struct.pack("<" + "d" * (nraw + 1), *([0.0] * (nraw + 1)))
        b += struct.pack("<d", 0.3)
        b += struct.pack("<" + "d" * (nraw + 1), *([0.0] * (nraw + 1)))
        b += struct.pack("<d", 0.1)
        b += struct.pack("<" + "d" * (nraw + 1), *([0.0] * (nraw + 1)))
        b += struct.pack("<i", 1)              # dark_gain_mode
        if not emiss:
            b += struct.pack("<i", 1)          # cal_valid
            b += struct.pack("<q", cfdate)
            b += struct.pack("<" + "d" * nwav1, *([1.0] * nwav1))
            b += struct.pack("<" + "d" * nwav2, *([1.0] * nwav2))
            b += struct.pack("<" + "d" * (nraw + 1), *([1.0] * (nraw + 1)))
            b += struct.pack("<d", 2850.0)
            b += struct.pack("<" + "d" * (2 * (nraw + 1)),
                             *([1.0] * (2 * (nraw + 1))))
        b += struct.pack("<i", 1)              # idark_valid
        b += struct.pack("<q", iddate)
        b += struct.pack("<" + "d" * 4, *([1.0] * 4))
        b += struct.pack("<" + "d" * (4 * (nraw + 1)),
                         *([1.0] * (4 * (nraw + 1))))
        return bytes(b)

    dark_age = 50 * 60                 # the emissive dance: 50 min old
    white_age = 300                    # reflective white: 5 min old
    for i in range(11):
        if i == 0:      # reflective spot
            out += blk(0, 0, 1, 0, 0, now - dark_age,
                       now - white_age, 0)
        elif i == 2:    # emissive non-adaptive (fast)
            out += blk(1, 0, 0, 0, 0, now - dark_age, 0, 0)
        elif i == 4:    # emissive adaptive
            out += blk(1, 0, 0, 0, 1, 0, 0, now - dark_age)
        elif i == 7:    # ambient adaptive
            out += blk(1, 0, 0, 1, 1, now - dark_age, 0, now - dark_age)
        else:
            out += blk(1, 0, 0, 0, 1, 0, 0, 0)
    out += struct.pack("<i", 0)        # trailing checksum
    with open(os.path.join(tmp, "ArgyllCMS", ".mk_SYNTH0001.cal"),
              "wb") as f:
        f.write(bytes(out))
    try:
        # adaptive emissive: interpolated dark cal dated 50 min ago
        left = sources.argyll_cal_remaining("e", False)
        check("cache: adaptive emissive 10 min left",
              left is not None and 9 * 60 < left < 11 * 60,
              str(left))
        # fast (non-adaptive) emissive reads ddate — same age here
        left = sources.argyll_cal_remaining("e", True)
        check("cache: non-adaptive emissive reads ddate",
              left is not None and 9 * 60 < left < 11 * 60,
              str(left))
        # reflective: white cal (24 h window, fresh) + dark cal
        # (1 h window, 50 min old) — the DARK cal governs
        left = sources.argyll_cal_remaining("r", False)
        check("cache: reflective bounded by its dark cal",
              left is not None and 9 * 60 < left < 11 * 60,
              str(left))
        # a stale cache (dark cal 2 h old) must read as expired (0)
        with open(os.path.join(tmp, "ArgyllCMS", ".mk_SYNTH0001.cal"),
                  "r+b") as f:
            pass
        # fresh-white-only scenario: reflective with an old dark cal
        stale = bytes(out)
        # byte-locate the reflective ddate: header 37 + 32 + 4 + 8 + 4
        off = 37 + 32 + 4 + 8 + 4
        stale = (stale[:off]
                 + struct.pack("<q", now - 2 * 3600)
                 + stale[off + 8:])
        with open(os.path.join(tmp, "ArgyllCMS",
                               ".mk_SYNTH0002.cal"), "wb") as f:
            f.write(stale[:25].replace(b"SYNTH0001", b"SYNTH0002")
                    + stale[25:])
        # SYNTH0002 is newer (just written) → selected by mtime
        left = sources.argyll_cal_remaining("r", False)
        check("cache: expired dark cal reads 0",
              left == 0.0, str(left))
    finally:
        for k, v in (("XDG_CACHE_HOME", old_xdg), ("HOME", old_home)):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


if __name__ == "__main__":
    for fn in (test_srgb_preview_max_brightness,
               test_gamut_containment_ordering,
               test_cct_and_xy,
               test_lab_implied_y_arbiter_math,
               test_reflective_honest_preview,
               test_csv_roundtrip_with_metadata,
               test_settings_roundtrip,
               test_argyll_cache_countdown):
        print(f"{fn.__name__}:")
        fn()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):", ", ".join(FAILURES))
        sys.exit(1)
    print("all processing tests passed")
