# Spectro-Live

Realtime-style spectrum viewer for spot-reading spectrometers and
spectrophotometers — it works with any instrument driven by
**ArgyllCMS `spotread`**. The app was originally built for the
**X-Rite ColorMunki Photo**, and so far it has only been tested on that
one instrument. It has the kind of features you'd expect from an
optical spectrometer console: averaging, smoothing, integration bands,
zero/background subtraction, A/B reference comparison, wavelength markers,
peak analysis and colorimetry.

The UI is a flat GNOME-style dark theme with a matplotlib plot,
matching the look of [Thermal-DMM-Viewer].

![Main window](screenshots/main_window.png)

The chromaticity diagram can also be detached into its own resizable window,
showing the CIE 1931 horseshoe with the sRGB / Adobe RGB / Display P3 /
Rec. 2020 gamuts around D50 and D65:

![Chromaticity window](screenshots/chroma_window.png)

## Quick start

```sh
git clone https://github.com/serniko97/Spectro-Live.git   # then:
cd Spectro-Live
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./spectro-live                 # GUI
./spectro-live --source simulator --start   # demo without hardware
```

## Sources

* **ColorMunki / i1 Pro / … (spotread)** — drives `spotread -v -s -T`
  through a PTY, because it needs a terminal for its key prompts.
  Emissive, ambient and reflective modes (`-e` / `-a`), high-resolution
  spectral output (`-H`), and `-c N` to pick one of several connected
  instruments. In ambient mode the app drops `-T`: spotread enables
  the CCT/CRI/TM-30 output by itself there and `-T` would suppress it
  — spotread XORs the flag with ambient. XYZ-only colorimeters
  supported.
* **Simulator** — synthetic fluorescent / LED / incandescent / dark spectra so
  everything can be exercised without hardware.

## Instrument / workflow notes

* Any instrument ArgyllCMS supports works (ColorMunki Photo/Design/Display,
  i1 Pro family, colorimeters like Spyder5, …); with several connected, pick
  one under **Instrument → Device…**. Reading modes auto-restrict to what
  the hardware supports and the wording adapts (wheel-slot hints are
  ColorMunki-only).
* **The annunciator lamps are the mode selector**: EMISSIVE,
  REFLECTIVE and AMBIENT are clickable — the selected one carries a
  blue border and lights amber only for itself while its wheel position
  is unconfirmed; hovering highlights them like the buttons they are.
  CALIBRATION stays a lamp too (green border while the calibration is
  valid; click it to recalibrate every mode). Mode changes restart the
  instrument session with one automatic re-calibration.
* The ColorMunki wheel has three positions and an outer detent:
  **1 = off, 2 = calibration, 3 = measuring** (front aperture: emissive and
  reflective), **outer slot past OFF = ambient** (top diffuser). Calibration
  is fully automatic — the app answers spotread's prompt itself; just turn
  the wheel to 2 while the CALIBRATION lamp lights up, then to the slot for
  your measurement. Calibration is **cached by ArgyllCMS on this computer**
  (~/.cache/ArgyllCMS; dark calibration valid 1 h, reflective white
  24 h): within those windows the app resumes measuring after a USB blip
  or a mode switch with no calibration dance at all (spotread `-N`).
  Instrument → **Force calibration** demands a fresh calibration at every
  session start instead; spotread itself still asks for one whenever the
  cache is genuinely stale. Argyll keeps a **separate calibration per
  reading mode**, so whenever a calibration is due the app runs one short
  session per mode automatically: keep the wheel at position 2 once and
  emissive, ambient and reflective are all freshly calibrated — mode
  switches then resume with no dance at all within the validity
  windows. Other instruments: follow the caption text.
* The ColorMunki's **physical button is a hardware trigger** fixed by
  spotread: each press takes one reading. While the app is **stopped**,
  late readings are dropped — the trace stays exactly as it was when
  STOP was clicked; use ⏺ SINGLE instead for one deliberate
  measurement while stopped. If the button is **held closed** —
  pressed-and-held, resting against something, or defective —
  spotread re-triggers a reading at every menu return; the app cannot
  prevent those readings, but while stopped they are dropped, so the
  trace never moves.
* The CALIBRATION lamp carries a **green border while the instrument's
  calibration is valid** — it turns green when spotread says
  "Calibration complete" (or when a session starts straight on
  Argyll's host cache), and fades when the instrument's own validity
  window expires (dark cal 1 h, reflective white 24 h) or the moment a
  new calibration demand appears. It never claims more than spotread
  itself said. **Click the CALIBRATION lamp** to demand a fresh
  calibration of every reading mode right now — normal Argyll cache
  reuse resumes afterwards. Every calibration the driver demands (at
  session start, before a reading, or after the lamp click) runs the
  same all-modes pass: one wheel-at-2 effort leaves *all three* mode
  calibrations fresh, so switching modes stays prompt-free for the
  whole validity window. During the pass the lamp fills a quarter for
  each completed mode (1/3 → 25 %, 2/3 → 50 %, 3/3 → 75 %); the all-green
  border means 100 % ready.
* **View → Curve style** switches the trace between straight segments
  (default), overshoot-free monotone smoothing, and
  sample-and-hold steps — presentation only, applied to the current
  trace instantly, without needing a new measurement (the raw data is
  unchanged either way).
* **View → Units** picks the vertical-axis unit: raw counts,
  V(λ)-weighted, photometric absolute, radiometric absolute or
  reflectance %. The **V(λ)-weighted** view is a shape
  view on the arbitrary scale (a.u.) — the eye's perceptual
  weighting, no cd/m² claim. The **photometric absolute** view
  follows the reading mode — luminance **cd/m²·nm⁻¹** (emissive),
  illuminance **lx·nm⁻¹** (ambient, E = π·Y) — scaled so its integral
  equals the instrument's own calibrated Y; in reflective mode it
  falls back to the plain, unweighted reflectance factor, because a
  reflectance is dimensionless and V(λ) weighting has no business
  applying to a surface. The **radiometric absolute** view gives
  the exact radiometric counterpart — radiance
  **W/(sr·m²)·nm⁻¹** (emissive) or irradiance **W/m²·nm⁻¹** (ambient) —
  undoing the V(λ) weighting so deep blue/red photons carry their real
  energy (Xₑ = Y·∫S/(683·∫S·ȳ)); reflective likewise falls back to
  plain reflectance. Both absolute views honestly show nothing when no
  calibrated Y anchor exists.
* **View → Axis: wavelength / frequency (THz) / energy (eV)** replots
  the same data against f = c/λ or the photon energy E = hc/λ
  (λ[nm] · E[eV] = 1239.84; ascending frequency/energy, red on the
  left). Presentation only: markers, the band and every analysis stay
  in nanometres; plot clicks convert back automatically.
* The **saturation bar right of the plot** shows how close the busiest
  sensor channel is to the end of its linear range — log scale, 0 at
  the bottom, 100 % = the sensor's analog knee. Below the knee the
  reading is linear; at or above it Argyll still accepts the reading
  but the brightest channels are no longer quantitative, so the bar
  goes steady red (flashing red well past the knee, or whenever
  spotread itself reports "Sensor is saturated"). The scale is shared
  between adaptive and fast mode (Argyll normalises each reading by
  its integration time). The ceiling model is only claimed where it
  can be substantiated — ColorMunki: the sensor's observed analog
  knee (per Argyll's `munki_imp.c`); DTP20: the fixed 16-bit
  full-scale conversion (`dtp20.c`); everything else, and all
  reflective readings: n/a rather than a guess.
* The left panel's **Chromaticity group** (below Band & markers) shows
  the CIE chromaticity diagram in true colours with the sRGB, Adobe RGB,
  Display P3 and Rec. 2020 gamut triangles, the D65/D50 white points
  and the gray Planckian (black-body) line; a red dot
  marks the current measurement, and **Open in a window** detaches a
  resizable copy. **View → Chromaticity** switches the diagram between
  CIE 1931 (x, y), CIE 1976 (u′, v′) and CIE 1960 (u, v) — the
  measurement dot, gamuts, white points and the black-body line all
  re-project. The sRGB triangle is always coloured with real sRGB
  values at full strength (they are exact inside the gamut), while the
  rest of the horseshoe stays muted.
* Each new spectrum updates the plot as soon as it is complete (a few
  seconds on the ColorMunki Photo). The plot range follows the
  instrument's own spectral range and resolution.
* The status bar carries a **centered calibration countdown**
  (“CAL✓ 47 min left” — Argyll's own validity windows: dark cal 1 h
  for emission modes, reflective white 24 h). It is honest about
  cache reuse: a restart that skips the calibration dance does NOT
  restart the window — the countdown is reconstructed from the
  wall-clock stamp of the last real dance (persisted in the app
  settings), so the minutes keep ticking down across app restarts.
* If the USB device disappears (flaky mini-USB happens),
  **“Device disconnected”** shows in the status bar and the app keeps
  polling; when the device re-appears the session continues automatically
  without losing settings — and a recalibration while stopped never
  restarts measuring on its own, so your on-screen trace is preserved.
* `spotread` must be in PATH (Fedora: `sudo dnf install argyllcms`). A different
  ArgyllCMS build can be selected explicitly with **Instrument → ArgyllCMS
  spotread path…** (persisted; the running session restarts on the new binary),
  and **Instrument → Use default spotread (from PATH)** reverts to automatic
  lookup.
* **The ColorMunki's first spectral bins are clones, not data**: Argyll's
  driver duplicates the 400 nm bin for everything below it (420 nm for
  reflective readings — `munki_imp.c`, "Duplicate below this wl"). The
  app shades that region **bordeaux** on the chart, and **Instrument →
  Ignore unreliable band** removes it from the chart, every
  measurement, the colour data and exports (XYZ is recomputed from
  the trusted band). Other instruments show no highlight and the clip
  is a no-op — the boundary is only claimed where it is known.
* **High spectral resolution / fast readings** toggles take effect
  immediately when a session is running (the instrument session
  restarts with the new flag; Argyll's calibration cache keeps it quick).

### Platforms

**Linux** is the primary platform: udev exposes the ColorMunki at
`/dev/hidraw*` after your distribution's Argyll udev rules, and device
presence is scanned from sysfs.

**macOS** is supported. `system_profiler` may not list Argyll
instruments at all (the ColorMunki enumerates with a vendor-specific
USB interface class), so on macOS `spotread` itself is the arbiter of
whether an instrument is connected; Argyll's calibration cache is
read from `~/Library/Caches/ArgyllCMS` (Argyll's own mapping,
`xdg_bds.c`).

**Windows** is supported too, implemented against Microsoft's
documented interfaces but **not yet exercised on real Windows
hardware** — honest caveat. The instrument session runs `spotread`
on plain pipes (there is no `pty` module), the USB scan uses PowerShell
`Get-CimInstance Win32_USBHub`, and Argyll's calibration cache is read
from `%APPDATA%\Cache\ArgyllCMS` (`xdg_bds.c` NT branch). Caveats: the
device-order → `-c` mapping is best-effort (with multiple instruments
use *Auto (first found)*), and Argyll's Windows USB driver may need
the usual one-time [Zadig](https://zadig.akeo.ie/) / libusb install for
the ColorMunki.

## Features

| Where | What |
| --- | --- |
| Averaging | rolling (last *N*) or cumulative mean over raw frames |
| Smoothing | moving-average with adjustable window |
| Zero | capture the current RAW reading as the new zero — pressing Zero again re-zeros (never undoes; only Clear zero removes it). While active the button is highlighted and everything (integral, peaks, chromaticity, colour coordinates) is computed from the corrected curve, with XYZ recomputed from it; saved CSVs are flagged `# zero_corrected` |
| Reference B | freeze the current processed trace or load from file; views: overlay, difference (A−B), ratio A/B % |
| Integration | trapezoidal ∫ over a band; band can follow the two markers |
| Markers | M1/M2 wavelength cursors (left/right mouse click on the plot), Δλ, intensity at each marker, Δ (intensity difference) |
| Peaks | λpeak, FWHM (linear interpolated half-max crossing), spectral centroid, peak table in the right panel (and written into every saved CSV). Peak **honesty rules**: endpoints count (a curve still rising into 730 nm genuinely peaks there), a bump counts by its height above the saddle between its own flanking valleys (not a global baseline), flat tops yield ONE peak, and a robust noise floor (4×σ from second differences) keeps jitter on a plateau from flooding the list |
| Colorimetry | CIE xy and CCT **with Duv** (instrument-reported, or computed from the spectrum with the CIE 1931 observer); photometric Y per mode (cd/m² / **lx = π·Y** ambient per Argyll's Lambertian-field model / %) **plus the exact radiometric counterpart**: radiance **W/(sr·m²)** pairing with the cd/m² (emissive) and irradiance **W/m²** pairing with the lx (ambient), derived as Xₑ = Xᵥ·∫S dλ/(683·∫S·V dλ) — the absolute scale anchored to the instrument's calibrated Y, the spectrum supplying the shape that undoes the photopic weighting (n/a for surfaces: reflectance is dimensionless); CRI Ra + R9 (spotread's own value and validity flag preferred) **and IES TM-30-15 Rf/Rg** (the modern fidelity/gamut pair, printed by spotread `-T` whenever your Argyll version provides it — instrument-reported with Argyll's own Caution flag, honest "absent" when not printed); sRGB preview with gamut report (sRGB, Adobe RGB, Display P3, Rec. 2020) |
| Logging | all **opt-in, off by default**: auto-log every reading to `~/spectro-live-logs/session-*.csv`, and the full spotread transcript to `~/spectro-live-logs/instrument-log-*.txt` via **Instrument → Write instrument log** (can be turned on mid-session; keeps what it wrote when turned off) |
| Export | traces and references as CSV, plot as PNG; load spectra from CSV / Argyll `.sp` / CGATS |

## Processing chain

```
raw frame → subtract zero (per frame) → average (rolling/cumulative)
         → smooth → trace A
                       ├── overlay / A−B / A÷B vs reference B
                       └── stats: integral, peak/FWHM/centroid, xy/CCT
```

[Thermal-DMM-Viewer]: https://github.com/serniko97/thermal-dmm-viewer

## Acknowledgements

This project could not exist without **[ArgyllCMS](https://www.argyllcms.com/)**
(Graziano Plegagnolu): every measurement flows through its `spotread` driver,
and its source was the reference for the instrument's behaviour that this UI
wraps honestly — calibration cache validity windows, the ambient diffuser's
Lambertian-field model (lx = π·Y), the integration-time ladder behind the
saturation estimates, and the prompt/flag etiquette the session automation
follows. The colorimetry itself uses the CIE 1931 standard observer, McCamy's
CCT approximation, and the CIE CRI / IES TM-30 methods as printed by Argyll.

## License

Copyright © 2026 Nicola Serra

Spectro-Live is free software: you can redistribute it and/or modify it
under the terms of the **GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version**. See [LICENSE](LICENSE) for the full text.
It depends on — but does not include — [ArgyllCMS](https://www.argyllcms.com/)
(AGPL), which you install separately.
