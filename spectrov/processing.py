"""Spectrum data type and processing math: averaging, zero subtraction,
integration, peak / FWHM / centroid, smoothing and simple colorimetry."""
from __future__ import annotations

from dataclasses import dataclass, field

import math

import numpy as np


@dataclass
class Spectrum:
    wavelengths: np.ndarray          # nm, ascending
    values: np.ndarray               # arbitrary intensity units
    xyz: tuple[float, float, float] | None = None
    label: str = ""
    timestamp: str = ""
    source: str = ""
    # True when spotread started this reading without any key from us —
    # i.e. its hardware trigger (the instrument's physical switch) fired
    hw_triggered: bool = False

    def copy(self) -> "Spectrum":
        return Spectrum(self.wavelengths.copy(), self.values.copy(),
                        self.xyz, self.label, self.timestamp, self.source)

    # -- interpolation helpers -----------------------------------------
    def at(self, wavelength: float) -> float:
        return float(np.interp(wavelength, self.wavelengths, self.values))

    def band(self, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
        m = (self.wavelengths >= min(lo, hi)) & (self.wavelengths <= max(lo, hi))
        return self.wavelengths[m], self.values[m]


def align(a: Spectrum, b: Spectrum) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bring two spectra onto a common wavelength grid (union, ascending)."""
    wl = np.union1d(a.wavelengths, b.wavelengths)
    return wl, np.interp(wl, a.wavelengths, a.values), np.interp(wl, b.wavelengths, b.values)


def average(spectra: list[Spectrum]) -> Spectrum | None:
    """Arithmetic mean over a common grid."""
    if not spectra:
        return None
    wl = spectra[0].wavelengths
    for s in spectra:
        wl = np.union1d(wl, s.wavelengths)
    vals = np.zeros_like(wl, dtype=float)
    for s in spectra:
        vals += np.interp(wl, s.wavelengths, s.values)
    out = Spectrum(wl, vals / len(spectra), label="average")
    xyz = [s.xyz for s in spectra if s.xyz is not None]
    if xyz:
        out.xyz = tuple(sum(c[i] for c in xyz) / len(xyz) for i in range(3))
    return out


def subtract(a: Spectrum, b: Spectrum) -> Spectrum:
    """a - b (zero/background subtraction, or difference view)."""
    wl, va, vb = align(a, b)
    return Spectrum(wl, va - vb, label="difference")


def ratio_pct(a: Spectrum, b: Spectrum) -> Spectrum:
    """a / b as percent, b=0 handled as 100."""
    wl, va, vb = align(a, b)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(np.abs(vb) > 1e-12, va / np.where(np.abs(vb) > 1e-12, vb, 1.0) * 100.0, 100.0)
    return Spectrum(wl, r, label="ratio %")


def smooth(s: Spectrum, window: int) -> Spectrum:
    """Moving-average smoothing; window is clipped to odd, min 3."""
    if window < 3:
        return s
    if window % 2 == 0:
        window += 1
    if window >= len(s.values):
        return s
    ker = np.ones(window) / window
    v = np.convolve(s.values, ker, mode="valid")
    pad = window // 2
    v = np.concatenate([np.full(pad, s.values[:pad].mean()),
                        v,
                        np.full(pad, s.values[-pad:].mean())])
    return Spectrum(s.wavelengths, v, s.xyz, s.label, s.timestamp, s.source)


# numpy 1.x (still common on macOS installs, where another pinned
# package can force it) has no trapezoid(): trapz() is the same
# function, renamed in numpy 2.0
_trapz = getattr(np, "trapezoid", None) or np.trapz


def integrate(s: Spectrum, lo: float, hi: float) -> float:
    """Trapezoidal integral of the spectrum clipped to band lo..hi (a.u.*nm)."""
    wl, v = s.band(lo, hi)
    if len(wl) < 2:
        # single point inside the band: estimate with neighbours
        wl, v = s.wavelengths, s.values
    return float(_trapz(np.clip(v, 0.0, None), wl))


def photopic_efficacy(wl) -> np.ndarray:
    """K(λ) = K_m·V(λ) — the CIE photopic SPECTRAL LUMINOUS EFFICACY
    in lm/W (K_m = 683 lm/W, the maximum at 555 nm). Multiplying an
    absolute spectral radiance by it gives spectral luminance
    (cd/m² per nm); on arbitrary-unit spectra it is the V(λ) eye
    weighting that turns radiometry into photometry."""
    return 683.0 * np.interp(np.asarray(wl, dtype=float),
                             _CMF_WL, _CMF_Y)


def xyz_from_values(s: Spectrum) -> tuple[float, float, float] | None:
    """Tristimulus XYZ computed straight from the spectral values —
    the CMF-weighted integral (clip-negative: noise below the stored
    zero has no colour). Used when the displayed curve is NOT the raw
    instrument reading (zero/background corrected): colorimetry then
    treats the corrected curve as the new reference for every
    computation."""
    if s is None or len(s.wavelengths) < 2:
        return None
    wl = np.asarray(s.wavelengths, dtype=float)
    v = np.clip(np.asarray(s.values, dtype=float), 0.0, None)
    xb = np.interp(np.clip(wl, 380.0, 780.0), _CMF_WL, _CMF_X)
    yb = np.interp(np.clip(wl, 380.0, 780.0), _CMF_WL, _CMF_Y)
    zb = np.interp(np.clip(wl, 380.0, 780.0), _CMF_WL, _CMF_Z)
    X = float(_trapz(v * xb, wl))
    Y = float(_trapz(v * yb, wl))
    Z = float(_trapz(v * zb, wl))
    if X + Y + Z <= 1e-12:
        return None
    return X, Y, Z


def photometric_absolute(s: Spectrum, y: float | None) -> Spectrum | None:
    """Spectral luminance / spectral illuminance ANCHORED to the
    instrument's own calibrated Y, in the display scale chosen by the
    caller:

        c(λ) = Y · [v(λ)·ȳ(λ)] / ∫ v·ȳ dλ       so that ∫ c dλ = Y exactly.

    The V(λ) weighting embodies the whole radiometric->photometric
    difference: radiometric quantities carry energy per se (W, W·m⁻²,
    W·sr⁻¹·m⁻²), photometric ones weight it by human perception
    (683 lm/W at 555 nm, falling to ~0 in the deep blue and red). The
    instrument's spectrum is in arbitrary units, so the ABSOLUTE scale
    is taken from its calibrated tristimulus Y — the curve shows the
    spectral *distribution* of that measured quantity. The luminous
    *shape* v(λ)·ȳ(λ) assumes the reported spectrum is proportional to
    the source's SPD (true for spotread's high-res output).

    Returns None when there is nothing to anchor to (no spectrum, or
    Y <= 0, which only divides by noise)."""
    if s is None or len(s.wavelengths) < 2 or y is None or y <= 0:
        return None
    ybar = np.interp(np.clip(s.wavelengths, 380.0, 780.0),
                     _CMF_WL, _CMF_Y)
    w = np.clip(np.asarray(s.values, dtype=float), 0.0, None) * ybar
    tot = float(_trapz(w, s.wavelengths))
    if tot <= 0.0:
        return None
    return Spectrum(s.wavelengths, w * (y / tot), s.xyz, s.label,
                    s.timestamp, s.source)


def peak_info(s: Spectrum, lo: float, hi: float) -> dict:
    wl, v = s.band(lo, hi)
    if len(wl) == 0:
        return {"wl": float("nan"), "val": float("nan"),
                "fwhm": float("nan"), "centroid": float("nan")}
    base = (v[0] + v[-1]) / 2.0
    v0 = v - base
    i = int(np.argmax(v0))
    pwl, pval = float(wl[i]), float(v0[i] + base)
    fwhm = float("nan")
    if pval > base:
        half = v0[i] / 2.0
        # walk left / right from the peak to the half crossings
        left = right = pwl
        for j in range(i, 0, -1):
            if v0[j - 1] <= half:
                t = (half - v0[j]) / (v0[j - 1] - v0[j])
                left = wl[j] + t * (wl[j - 1] - wl[j])
                break
        for j in range(i, len(wl) - 1):
            if v0[j + 1] <= half:
                t = (half - v0[j]) / (v0[j + 1] - v0[j])
                right = wl[j] + t * (wl[j + 1] - wl[j])
                break
        if right > left and math.isfinite(left) and math.isfinite(right):
            fwhm = right - left
    pos = np.clip(v0, 0.0, None)
    centroid = float((wl * pos).sum() / pos.sum()) if pos.sum() > 0 else float("nan")
    return {"wl": pwl, "val": pval, "fwhm": fwhm, "centroid": centroid}


def find_peaks(s: Spectrum, prominence: float, n_max: int = 8) -> list[tuple[float, float]]:
    """Honest local-maximum search (no scipy):

    * an interior point is a peak when it is a local maximum and it
      rises at least `prominence` (fraction of ITS OWN level — 0.04 =
      4 %) above the saddle between its flanking valleys — the saddle
      is the HIGHER of the two side minima, so a bump on a pedestal
      counts by its own height, not by the global range;
    * ENDPOINTS count as peaks when they dominate their single
      neighbour by the same margin: a red-reflectance curve rising
      into 730 nm HAS its maximum there, and pretending otherwise
      would hide the most important feature of the reading;
    * everything must also clear a robust noise floor (≈3× the
      σ estimated from the second differences, which slopes and
      ramps don't inflate), so jitter on a plateau never shows up as
      a mountain of fake peaks."""
    wl = np.asarray(s.wavelengths, dtype=float)
    v = np.asarray(s.values, dtype=float)
    if len(v) < 3 or len(wl) != len(v):
        return []
    # robust noise estimate: second differences are blind to smooth
    # slopes and ramps; white noise gives sigma ≈ median|d2| / 1.653
    d2 = np.abs(np.diff(v, 2))
    noise = float(np.median(d2)) / 1.653 if len(d2) else 0.0
    floor = 4.0 * noise

    def saddle(i: int) -> float:
        # true-prominence-lite: walk each side until the curve rises to
        # the candidate's own height (or the trace ends), tracking the
        # lowest point seen — the saddle is the HIGHER of the two
        # valley bottoms. Small bumps on big structures then count by
        # their own height, like a color scientist would judge them.
        def side_min(step: int) -> float:
            j, m = i + step, float(v[i])
            while 0 <= j < len(v):
                if v[j] > v[i]:
                    break                # strictly higher ground reached
                m = min(m, float(v[j]))  # equal tops are walkable
                j += step
            return m
        return max(side_min(-1), side_min(+1))

    def required(i: int) -> float:
        return max(prominence * max(float(v[i]), 0.0), floor)

    # collect candidate indices, merging consecutive bins: a flat
    # plateau top (every bin a "local maximum") must yield ONE peak,
    # not a row of identical ones — keep the representative bin
    cand: list[int] = []
    last_seen: list[int] = []           # the last CONSIDERED bin (even folded)

    def push(i: int):
        if cand and last_seen and i <= last_seen[-1] + 1:
            # adjacent bin: same feature — keep the higher side, but
            # the chain keeps running across folded bins too
            if float(v[i]) > float(v[cand[-1]]):
                cand[-1] = i
        else:
            cand.append(i)
        last_seen.append(i)

    peaks: list[tuple[float, float]] = []
    n = len(v)
    for i in range(1, n - 1):
        if v[i] < v[i - 1] or v[i] < v[i + 1]:
            continue                       # not a local maximum
        if float(v[i]) - saddle(i) >= required(i):
            push(i)
    # endpoints: peaks only when they clearly dominate their neighbour
    if v[0] - v[1] >= max(prominence * max(float(v[0]), 0.0), floor):
        push(0)
    if v[-1] - v[-2] >= max(prominence * max(float(v[-1]), 0.0), floor):
        push(n - 1)
    peaks = [(float(wl[i]), float(v[i])) for i in cand]
    peaks.sort(key=lambda p: -p[1])
    return peaks[:n_max]


def dominant_wavelength(xyz: tuple[float, float, float]) -> tuple[float | None, bool]:
    """Dominant wavelength (nm) of a colour given as XYZ, plus a flag:
    True = dominant, False = complementary (the colour lies on the
    purple side: the negative-spectrum-line intercept is reported)."""
    x, y = xy_from_xyz(xyz)
    wl = np.arange(380.0, 781.0, 5.0)
    XBAR, YBAR, ZBAR = _CMF_X, _CMF_Y, _CMF_Z
    loc_x = np.empty(len(wl)); loc_y = np.empty(len(wl))
    for i in range(len(wl)):
        s = XBAR[i] + YBAR[i] + ZBAR[i]
        loc_x[i] = XBAR[i] / s
        loc_y[i] = YBAR[i] / s
    # search the line from white E through (x,y) for the spectral crossing
    ex, ey = 1/3, 1/3
    best = None
    for i in range(len(wl) - 1):
        t = _seg_cross(ex, ey, x, y, loc_x[i], loc_y[i],
                       loc_x[i+1], loc_y[i+1])
        if t is not None and (best is None or t > best[0]):
            best = (t, wl[i])
    if best is not None and best[0] > 0:
        return float(best[1]), True
    # complementary: extend the line in the opposite direction
    best = None
    for i in range(len(wl) - 1):
        t = _seg_cross(x, y, 2*x - ex, 2*y - ey, loc_x[i], loc_y[i],
                       loc_x[i+1], loc_y[i+1])
        if t is not None and (best is None or t < best[0]):
            best = (t, wl[i])
    if best is not None:
        return float(best[1]), False
    return None, True


def _seg_cross(x0, y0, x1, y1, ax, ay, bx, by):
    """Parametric crossing of ray (x0,y0)->(x1,y1) with segment a-b; t on ray."""
    dx, dy = x1 - x0, y1 - y0
    ux, uy = bx - ax, by - ay
    den = dx * uy - dy * ux
    if abs(den) < 1e-12:
        return None
    t = ((ax - x0) * uy - (ay - y0) * ux) / den
    s = ((ax - x0) * dy - (ay - y0) * dx) / den
    if 0.0 <= s <= 1.0:
        return t
    return None


def list_peaks_text(s: Spectrum, prominence: float) -> str:
    peaks = find_peaks(s, prominence)
    if not peaks:
        return "No peaks above threshold."
    out = []
    for i, (w, v) in enumerate(peaks, 1):
        info = peak_info(s, w - 12.0, w + 12.0)
        out.append(f"  {i}. λ = {w:8.1f} nm   E = {v:10.4f}   "
                   f"FWHM ≈ {info['fwhm']:6.1f} nm")
    return "\n".join(out)


# --- colorimetry -----------------------------------------------------------
def wavelength_to_rgb(wl: float) -> tuple[float, float, float]:
    """Approximate sRGB appearance of a isolated spectral colour (Dan
    Bruton's classic piecewise-linear approximation, gamma 0.8, with the
    intensity fall-off at the ends of the visual range). This is the usual
    'rainbow' you see printed on optical spectrometers."""
    wl = min(max(wl, 380.0), 780.0)
    if wl < 440:
        r, g, b = (440.0 - wl) / 60.0, 0.0, 1.0
    elif wl < 490:
        r, g, b = 0.0, (wl - 440.0) / 50.0, 1.0
    elif wl < 510:
        r, g, b = 0.0, 1.0, (510.0 - wl) / 20.0
    elif wl < 580:
        r, g, b = (wl - 510.0) / 70.0, 1.0, 0.0
    elif wl < 645:
        r, g, b = 1.0, (645.0 - wl) / 65.0, 0.0
    else:
        r, g, b = 1.0, 0.0, 0.0
    if wl < 420:
        f = 0.30 + 0.70 * (wl - 380.0) / 40.0
    elif wl <= 700:
        f = 1.0
    else:
        f = 0.30 + 0.70 * (780.0 - wl) / 80.0
    return tuple(max(0.0, x) ** 0.8 * f for x in (r, g, b))


def rainbow_palette(lo: float = 380.0, hi: float = 730.0,
                    step: float = 1.0) -> np.ndarray:
    """(1, n, 3) float array of RGB colours from lo to hi, for imshow."""
    n = int(round((hi - lo) / step)) + 1
    cols = np.zeros((1, n, 3))
    for i in range(n):
        cols[0, i] = wavelength_to_rgb(lo + i * step)
    return cols


_D65_WP = (0.3127, 0.3290)
_GAMUT_PRIMS = {
    "sRGB":       ((0.640, 0.330), (0.300, 0.600), (0.150, 0.060)),
    "Adobe RGB":  ((0.640, 0.330), (0.210, 0.710), (0.150, 0.060)),
    "Display P3": ((0.680, 0.320), (0.265, 0.690), (0.150, 0.060)),
    "Rec. 2020":  ((0.708, 0.292), (0.170, 0.797), (0.131, 0.046)),
}
_GAMUT_MATS: dict | None = None


def _xyz_to_rgb_matrix(prim, wp):
    """XYZ(D65) -> linear RGB matrix for the given (x,y) primaries."""
    cols = []
    for x, y in prim:
        cols.append((x / y, 1.0, (1.0 - x - y) / y))
    M = np.column_stack(cols)               # XYZ of unit R, G, B columns
    wx, wy = wp
    W = np.array((wx / wy, 1.0, (1.0 - wx - wy) / wy))
    S = np.linalg.solve(M, W)               # channel luminance weights
    return np.linalg.inv(M @ np.diag(S))


def cri_from_spectrum(spectrum) -> dict | None:
    """General colour rendering index (classical CIE Ra, CIE 13.3:1995,
    via the colour-science package) of a LIGHT SOURCE spectrum.

    Ra is the mean of the colour differences R1..R8 against a reference
    illuminant of the same CCT (Planckian below 5000 K, CIE D above);
    R9 (saturated red) is reported too — LED makers love to hide it.

    Returns {"Ra": float, "R9": float} clipped to [-100, 100], or None
    when no spectrum exists (colorimeters) or the math is not possible.
    Only meaningful for Emissive/Ambient readings of actual light
    sources, not for surface reflectances."""
    if spectrum is None or len(spectrum.wavelengths) < 8:
        return None
    try:
        import colour
        from colour.quality.cri import colour_rendering_index
    except ImportError:
        return None
    try:
        wl = np.clip(np.asarray(spectrum.wavelengths, dtype=float), 0, None)
        v = np.clip(np.asarray(spectrum.values, dtype=float), 0.0, None)
        if not v.any():
            return None
        sd = colour.SpectralDistribution(dict(zip(wl, v)))
        with np.errstate(all="ignore"):
            res = colour_rendering_index(sd, additional_data=True)
        ra = float(res.Q_a)
        r9 = float(res.Q_as[9].Q_a) if 9 in res.Q_as else None
        return {"Ra": max(-100.0, min(100.0, ra)),
                "R9": None if r9 is None else max(-100.0, min(100.0, r9))}
    except Exception:
        return None


def gamut_report(xyz: tuple[float, float, float]) -> dict[str, bool]:
    """In-gamut flags for sRGB, Adobe RGB, Display P3 and Rec. 2020,
    tested in linear-light with a 0.2% tolerance. Assumes XYZ in D50
    (Bradford-adapted to D65)."""
    global _GAMUT_MATS
    if _GAMUT_MATS is None:
        _GAMUT_MATS = {n: _xyz_to_rgb_matrix(p, _D65_WP)
                       for n, p in _GAMUT_PRIMS.items()}
    X, Y, Z = (max(float(c), 0.0) if i == 1 else float(c)
               for i, c in enumerate(xyz))
    d65 = _BRADFORD_D50_D65 @ np.array([X, Y, Z])
    out = {}
    for name, M in _GAMUT_MATS.items():
        lin = M @ d65
        m = float(lin.max())
        if m <= 0.0:
            out[name] = False
            continue
        # luminance-INDEPENDENT test: scale to the brightest channel;
        # the chromaticity is inside the gamut iff every channel is
        # then non-negative (the max is 1 by construction).
        out[name] = bool((lin / m).min() >= -0.002)
    return out


def linear_channels(xyz: tuple[float, float, float]):
    """Linear sRGB (D65) channels from D50 XYZ, Y clamped at zero.
    Returns the raw (unclamped) linear values so callers can test for
    out-of-range conditions before preview encoding."""
    X, Y, Z = (max(float(c), 0.0) if i == 1 else float(c)
               for i, c in enumerate(xyz))
    return _XYZ_TO_RGB_D65 @ (_BRADFORD_D50_D65 @ np.array([X, Y, Z]))


def srgb_preview(xyz: tuple[float, float, float], normalize: bool = True):
    """Preview colour for the measured colour.

    With ``normalize=True`` (light sources) the chromaticity is scaled so
    its brightest linear-sRGB channel is exactly 1.0 — hue and saturation
    "as bright as sRGB allows", independent of the light's actual
    luminance (a 20 cd/m² lamp and the same lamp at 0.2 cd/m² preview
    identically; normalising to Y = 1 instead would wrongly blow warm
    chromaticities out of gamut, since a warm colour cannot reach Y = 1
    without saturating red).

    With ``normalize=False`` (reflective spot readings) the factors are
    shown at their real value: a white surface previews near-white, a
    dark wall dark — i.e. how the surface looks under the instrument's
    LED illumination.

    Returns (rgb, out_of_gamut, over, black):
      rgb      8-bit clamped sRGB triple
      oog      True when the colour is outside the sRGB gamut
      over     worst channel trespass after scaling (0..1)
      black    True when the measurement is too dark to evaluate
    """
    X, Y, Z = (max(float(c), 0.0) if i == 1 else float(c)
               for i, c in enumerate(xyz))
    lin = _XYZ_TO_RGB_D65 @ (_BRADFORD_D50_D65 @ np.array([X, Y, Z]))
    m = float(lin.max())
    if m <= 1e-6:
        # too dark for a meaningful chromaticity (sensor noise)
        return (0, 0, 0), False, 0.0, True
    if normalize:
        lin = lin / m
        over = max(0.0, -lin.min())
        oog = over > 0.002
    else:
        over = max(0.0, -lin.min(), lin.max() - 1.0)
        oog = over > 0.002
        lin = np.clip(lin, 0.0, 1.0)
    gamma = np.where(lin <= 0.0031308, 12.92 * lin,
                     1.055 * np.power(np.clip(lin, 0.0031308, None),
                                      1.0 / 2.4) - 0.055)
    rgb = tuple(int(round(255 * c)) for c in np.clip(gamma, 0.0, 1.0))
    return rgb, oog, float(over), False


# Bradford chromatic adaptation D50 -> D65
_BRADFORD_D50_D65 = np.array([
    [ 0.9555766, -0.0230393,  0.0631636],
    [-0.0282895,  1.0099416,  0.0210077],
    [ 0.0122982, -0.0204830,  1.3299098]])
# linear sRGB (D65) from XYZ (D65)
_XYZ_TO_RGB_D65 = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252]])


def xyz_to_srgb(xyz: tuple[float, float, float]) -> tuple[tuple[int, int, int],
                                                          bool, float]:
    """XYZ (assumed D50, as Argyll reports) -> 8-bit sRGB triple, an
    out-of-gamut flag, and the worst overshoot (fraction above 1.0 of
    the most-exceeded linear channel; 0 when in gamut).
    The returned colour is clamped for display."""
    X, Y, Z = (max(float(c), 0.0) if i == 1 else float(c)
               for i, c in enumerate(xyz))
    lin = _XYZ_TO_RGB_D65 @ (_BRADFORD_D50_D65 @ np.array([X, Y, Z]))
    # linear-light values must all be inside [0, 1] for in-gamut sRGB
    # (numerical overshoot below 0.2% counts as in-gamut)
    over = max(0.0, -lin.min(), lin.max() - 1.0)
    oog = bool(over > 0.002)
    lin_c = np.clip(lin, 0.0, 1.0)
    gamma = np.where(lin_c <= 0.0031308, 12.92 * lin_c,
                     1.055 * np.power(lin_c, 1.0 / 2.4) - 0.055)
    rgb = tuple(int(round(255 * c)) for c in np.clip(gamma, 0.0, 1.0))
    return rgb, oog, float(over)


def xy_from_xyz(xyz: tuple[float, float, float]) -> tuple[float, float]:
    t = sum(xyz)
    if t <= 0:
        return 0.3127, 0.3290
    return xyz[0] / t, xyz[1] / t


# ---- chromaticity spaces (CIE 1931 xy, 1976 u'v', 1960 uv) -----------
_CHROM_SPACES = ("1931", "1976", "1960")


def xy_to_chrom_space(x, y, space: str = "1931"):
    """Map CIE 1931 xy chromaticities to another diagram's coordinates
    (arrays or scalars). '1931' is the identity; '1976' is CIE 1976
    u'v'; '1960' is CIE 1960 uv (the CCT-projection space)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if space == "1931":
        return x, y
    d = -2.0 * x + 12.0 * y + 3.0
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(np.abs(d) > 1e-12, 4.0 * x / d, np.nan)
        v = np.where(np.abs(d) > 1e-12, 6.0 * y / d, np.nan)
    if space == "1960":              # uv = u', 2/3·v'
        return u, v * (2.0 / 3.0)
    return u, v                        # 1976 u'v'


def chrom_space_to_xy(u, v, space: str = "1931"):
    """Inverse of xy_to_chrom_space: target-space coordinates back to
    CIE 1931 xy (invalid / degenerate points come back as NaN)."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if space == "1931":
        return u, v
    vp = v if space == "1976" else v * 1.5     # 1960: v' = 1.5 · v
    X = u / 4.0
    Y = vp / 6.0
    Z = (1.0 - X - 15.0 * Y) / 3.0
    s = X + Y + Z
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(np.abs(s) > 1e-12, X / s, np.nan)
        y = np.where(np.abs(s) > 1e-12, Y / s, np.nan)
    return x, y


def chrom_space_labels(space: str = "1931") -> tuple[str, str]:
    """Axis titles for the selected chromaticity diagram."""
    if space == "1976":
        return "CIE 1976 u′", "CIE 1976 v′"
    if space == "1960":
        return "CIE 1960 u", "CIE 1960 v"
    return "CIE 1931 x", "CIE 1931 y"


def chrom_space_name(space: str = "1931") -> str:
    return {"1931": "CIE 1931 (x, y)", "1976": "CIE 1976 (u′, v′)",
            "1960": "CIE 1960 (u, v)"}.get(space, "CIE 1931 (x, y)")


def planckian_locus_xy(t_start: float = 1200.0, t_end: float = 25000.0,
                       n: int = 160) -> tuple[np.ndarray, np.ndarray]:
    """The Planckian locus (black-body radiator chromaticities) in CIE
    1931 xy: Planck's law integrated against the CIE 1931 2° colour
    matching functions, from t_start to t_end kelvin. Waypoints for
    labels: ~2856 K ill. A (tungsten), ~6500 K D65-daylight region."""
    wl = _CMF_WL * 1e-9                      # metres
    h = 6.62607015e-34
    c = 2.99792458e8
    k = 1.380649e-23
    c2 = h * c / k                           # second radiation constant
    temps = np.linspace(t_start, t_end, n)
    xs = np.empty(n)
    ys = np.empty(n)
    for i, T in enumerate(temps):
        # spectral radiance (unnormalised — chromaticity is ratio-safe)
        spec = 1.0 / (wl ** 5 * np.expm1(c2 / (wl * T)))
        Xv = float(_trapz(spec * _CMF_X, _CMF_WL))
        Yv = float(_trapz(spec * _CMF_Y, _CMF_WL))
        Zv = float(_trapz(spec * _CMF_Z, _CMF_WL))
        s = Xv + Yv + Zv
        xs[i] = Xv / s
        ys[i] = Yv / s
    return xs, ys


def cct_from_xyz(xyz: tuple[float, float, float]) -> float:
    """Correlated colour temperature, McCamy's cubic approximation."""
    x, y = xy_from_xyz(xyz)
    n = (x - 0.3320) / (0.1858 - y)
    return 449.0 * n ** 3 + 3525.0 * n ** 2 + 6823.3 * n + 5520.33


# CIE 1931 2-degree observer colour matching functions, 5 nm, 380–780 nm
_CMF_WL = np.arange(380.0, 781.0, 5.0)
_CMF_X = np.array([0.001368, 0.002236, 0.004243, 0.00765, 0.01431, 0.02319,
                   0.04351, 0.07763, 0.13438, 0.21477, 0.2839, 0.3285, 0.34828,
                   0.34806, 0.3362, 0.3187, 0.2908, 0.2511, 0.19536, 0.1421,
                   0.09564, 0.05795, 0.03201, 0.0147, 0.0049, 0.0024, 0.0093,
                   0.0291, 0.06327, 0.1096, 0.1655, 0.2257499, 0.2904, 0.3597,
                   0.4334499, 0.5120501, 0.5945, 0.6784, 0.7621, 0.8425,
                   0.9163, 0.9786, 1.0263, 1.0567, 1.0622, 1.0456, 1.0026,
                   0.9384, 0.8544499, 0.7514, 0.6424, 0.5419, 0.4479, 0.3608,
                   0.2835, 0.2187, 0.1649, 0.1212, 0.0874, 0.0636, 0.04677,
                   0.0329, 0.0227, 0.01584, 0.0113592, 0.0081109, 0.0057903,
                   0.0041095, 0.0028993, 0.0020492, 0.00144, 0.0009999,
                   0.0006901, 0.000476, 0.0003323, 0.0002348, 0.0001662,
                   0.0001174, 8.31e-05, 5.87e-05, 4.15e-05])
_CMF_Y = np.array([3.9e-05, 6.4e-05, 0.00012, 0.000217, 0.000396, 0.00064,
                   0.00121, 0.00218, 0.004, 0.0073, 0.0116, 0.01684, 0.023,
                   0.0298, 0.038, 0.048, 0.06, 0.0739, 0.09098, 0.1126,
                   0.13902, 0.1693, 0.20802, 0.2586, 0.323, 0.4073, 0.503,
                   0.6082, 0.71, 0.7932, 0.862, 0.9148501, 0.954, 0.9803,
                   0.9949501, 1.0, 0.995, 0.9786, 0.952, 0.9154, 0.87, 0.8163,
                   0.757, 0.6949, 0.631, 0.5668, 0.503, 0.4412, 0.381, 0.321,
                   0.265, 0.217, 0.175, 0.1382, 0.107, 0.0816, 0.061, 0.04458,
                   0.032, 0.0232, 0.017, 0.01192, 0.00821, 0.005723, 0.004102,
                   0.002929, 0.002091, 0.001484, 0.001047, 0.00074, 0.00052,
                   0.0003611, 0.0002492, 0.0001719, 0.00012, 8.48e-05, 6e-05,
                   4.24e-05, 3e-05, 2.12e-05, 1.5e-05])
_CMF_Z = np.array([0.00645, 0.01055, 0.02005, 0.03621, 0.06785, 0.1102,
                   0.2074, 0.3713, 0.6456, 1.0390501, 1.3856, 1.62296,
                   1.74706, 1.7826, 1.77211, 1.7441, 1.6692, 1.5281, 1.28764,
                   1.0419, 0.8129501, 0.6162, 0.46518, 0.3533, 0.272, 0.2123,
                   0.1582, 0.1117, 0.07825, 0.05725, 0.04216, 0.02984, 0.0203,
                   0.0134, 0.00875, 0.00575, 0.0039, 0.00275, 0.0021, 0.0018,
                   0.00165, 0.0014, 0.0011, 0.001, 0.0008, 0.0006, 0.00034,
                   0.00024, 0.00019, 0.0001, 5e-05, 3e-05, 2e-05, 1e-05,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
_CMF_Z = _CMF_Z[:len(_CMF_WL)]


def spectral_locus_xy() -> tuple[np.ndarray, np.ndarray]:
    """(x, y) chromaticities of the CIE 1931 2° spectral locus. Uses
    the colour-science 1 nm CMFs when available (the 5 nm polygon
    misses near-boundary gamut primaries like Rec. 2020 green even
    after spline upsampling); otherwise falls back to the embedded
    5 nm table upsampled with a monotone spline. The polygon closed
    with the purple line gives the visible gamut."""
    wl = None
    XYZ = None
    try:
        import colour
        cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
        shp = cmfs.shape
        wl = np.arange(shp.start, shp.end + shp.interval, shp.interval)
        XYZ = np.asarray(cmfs.values, dtype=float)
        band = (wl >= 380.0) & (wl <= 780.0)
        wl, XYZ = wl[band], XYZ[band]
    except Exception:
        pass
    if wl is None:
        wl, X, Y, Z = _CMF_WL, _CMF_X, _CMF_Y, _CMF_Z
        XYZ = np.column_stack([X, Y, Z])
        keep = XYZ.sum(axis=1) > 1e-9
        wl, XYZ = wl[keep], XYZ[keep]
        if len(wl) >= 3:          # monotone 5× upsample of the coarse grid
            wl_fine = pchip_smooth(wl, wl, up=5)[1]
            XYZ = np.column_stack(
                [pchip_smooth(wl, XYZ[:, c], up=5)[1] for c in range(3)])
            wl = wl_fine
    tot = XYZ.sum(axis=1)
    keep = tot > 1e-9
    tot = np.where(tot > 1e-12, tot, 1.0)
    return XYZ[keep, 0] / tot[keep], XYZ[keep, 1] / tot[keep]


def chromaticity_image(n: int = 220, space: str = "1931",
                       true_srgb: bool = False):
    """The filled CIE chromaticity diagram as an RGBA image, with each
    chromaticity rendered in its closest sRGB colour (out-of-sRGB
    chromaticities are clipped at the gamut, like every printed CIE
    diagram). `space` selects the diagram: '1931' (classic xy),
    '1976' (u'v') or '1960' (uv). With `true_srgb=True` the pixels
    inside the sRGB triangle keep their true colours at full opacity
    (they ARE exact there), while everything else inside the locus is
    alpha-muted — the "true sRGB diagram" some colour tools show.
    Returns (extent, rgba, (locus_u, locus_v)): `extent` is the
    (x0, x1, y0, y1) data rectangle for imshow, in the SELECTED
    space's coordinates."""
    lx, ly = spectral_locus_xy()
    from matplotlib.path import Path
    poly = Path(np.column_stack([lx, ly]))

    # grid directly in the target space, mapped back to xy for the
    # inside-locus test and the colour maths (both are invariant
    # under the projective space change, mixing only rescales XYZ)
    if space == "1931":
        xs = np.linspace(-0.06, 0.78, n)
        ys = np.linspace(-0.05, 0.90, n)
    else:
        lu, lv = xy_to_chrom_space(lx, ly, space)
        upad = 0.06 * (np.nanmax(lu) - np.nanmin(lu))
        vpad = 0.06 * (np.nanmax(lv) - np.nanmin(lv))
        xs = np.linspace(np.nanmin(lu) - upad, np.nanmax(lu) + upad, n)
        ys = np.linspace(np.nanmax(lv) + vpad, np.nanmin(lv) - vpad, n)
        ys = np.sort(ys)
    gx, gy = np.meshgrid(xs, ys)
    cx, cy = chrom_space_to_xy(gx, gy, space)
    finite = np.isfinite(cx) & np.isfinite(cy)
    cx = np.where(finite, cx, 0.3127)
    cy = np.where(finite, cy, 0.3290)
    yv = np.where(cy > 1e-6, cy, 1e-6)
    X = cx / yv
    Z = (1.0 - cx - cy) / yv
    global _GAMUT_MATS
    if _GAMUT_MATS is None:      # initialise gamut_report's cache too
        _GAMUT_MATS = {nm: _xyz_to_rgb_matrix(p, _D65_WP)
                       for nm, p in _GAMUT_PRIMS.items()}
    lin = np.stack([X, np.ones_like(X), Z], axis=-1) @ \
        _GAMUT_MATS["sRGB"].T
    # PER-PIXEL CHROMA NORMALISATION, then clipping. A chromaticity is
    # a DIRECTION in XYZ — multiplying by a constant leaves xy exactly
    # where it was — so scaling each pixel's linear sRGB so its LARGEST
    # channel equals 1 is the same colour, rendered at full chroma
    # (the classic saturated CIE-diagram look). The old code clipped
    # instead: un-clipped channels stayed small and the whole triangle
    # washed out (reported: "I don't see the blue" — xy (0.17, 0.08)
    # rendered lavender (215,126,255) instead of blue-violet (74,42,255)).
    # Chromaticities outside sRGB still clip to the gamut edge like
    # every printed CIE diagram, because their negative channels sink
    # the scale and force clipping back to the closest in-gamut colour.
    peak = lin.max(axis=-1, keepdims=True)
    peak = np.where(peak > 1e-6, peak, 1.0)   # degenerate: leave as-is
    lin = np.clip(lin / peak, 0.0, 1.0)
    rgb = np.where(lin <= 0.0031308, 12.92 * lin,
                   1.055 * np.power(lin, 1.0 / 2.4) - 0.055)
    # ---- supersampled coverage masks (antialiasing) -----------------
    # contains_points() answers a binary inside/outside for ONE sample
    # point per cell, so a single-sample mask makes the boundary a
    # pixel staircase (and imshow's bilinear interpolation is not
    # rotation-invariant: diagonal boundaries come out as diamond
    # artefacts). Instead: sample every output pixel at ss×ss sub-cell
    # positions and average — boundary pixels get a fractional
    # coverage, i.e. a true smooth perimeter.
    ss = 2                      # 2x2 sub-samples per output pixel
    # fine grid: ss subsamples per output pixel, each subsample at the
    # centre of its sub-cell (pixel i covers [x_i - h/2, x_i + h/2];
    # its subsample centres sit at x_i ± h/(2·ss))
    k = np.arange(n * ss)
    fine_of = lambda centres: \
        centres[0] + (k + 0.5 - 0.5 * ss) / ss * (centres[1] - centres[0])
    fxs = fine_of(xs)
    fys = fine_of(ys)
    fgx, fgy = np.meshgrid(fxs, fys)
    fcx, fcy = chrom_space_to_xy(fgx, fgy, space)
    finite2 = np.isfinite(fcx) & np.isfinite(fcy)
    fcx = np.where(finite2, fcx, 0.3127)
    fcy = np.where(finite2, fcy, 0.3290)
    fpts = np.column_stack([fcx.ravel(), fcy.ravel()])
    # (n*ss, n*ss) -> blocks of ss×ss -> mean = per-pixel coverage 0..1
    cov = poly.contains_points(fpts).reshape(
        n, ss, n, ss).mean(axis=(1, 3))
    if true_srgb:
        # true-colour sRGB island: full opacity inside the sRGB
        # triangle, muted everywhere else in the horseshoe — both
        # perimeters antialiased via the same coverage trick. The
        # sRGB triangle lies fully inside the locus, so per-pixel:
        # alpha = 0.35 + 0.65 · (sRGB sub-sample coverage)
        srgb_tri = Path(np.array(_GAMUT_PRIMS["sRGB"]))
        scov = srgb_tri.contains_points(fpts).reshape(
            n, ss, n, ss).mean(axis=(1, 3))
        alpha = 0.35 + 0.65 * scov
    else:
        alpha = 1.0              # the caller mutes the whole image
    rgba = np.concatenate(
        [rgb, (cov * alpha)[..., None].astype(float)], axis=-1)
    extent = (xs[0], xs[-1], ys[0], ys[-1])
    pus, pvs = xy_to_chrom_space(lx, ly, space)
    return extent, rgba, (pus, pvs)


def pchip_smooth(wl, v, up: int = 4):
    """Monotone-cubic (Fritsch–Carlson PCHIP) upsampling of a sampled
    curve: smooth for display, but mathematically incapable of
    overshoot — an interpolated point can never claim a value above or
    below the measured ones, so the curve stays honest about what the
    sensor read. Falls back to the raw points when they are too few."""
    wl, v = np.asarray(wl, dtype=float), np.asarray(v, dtype=float)
    if len(wl) < 3 or up < 2:
        return wl, v
    h = np.diff(wl)
    d = np.diff(v) / h
    # PCHIP slopes; endpoints use Brodlie's one-sided three-point
    # estimate (same as scipy's PchipInterpolator), with the same
    # sign-preservation clamps scipy applies (melt to 0 or 3x secant)
    m = np.zeros_like(v)

    def _end_slope(h0, h1, d0, d1):
        s = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if np.sign(s) != np.sign(d0):
            return 0.0
        if (np.sign(d0) != np.sign(d1)) and (abs(s) > 3.0 * abs(d0)):
            return 3.0 * d0
        return s

    m[0] = _end_slope(h[0], h[1], d[0], d[1])
    m[-1] = _end_slope(h[-1], h[-2], d[-1], d[-2])
    for i in range(1, len(v) - 1):
        if d[i - 1] * d[i] <= 0.0:
            m[i] = 0.0
        else:
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    out_x, out_y = [], []
    for i in range(len(v) - 1):
        t = np.linspace(0.0, 1.0, up, endpoint=(i == len(v) - 2))
        hh = h[i]
        h00 = 2 * t ** 3 - 3 * t ** 2 + 1
        h10 = t ** 3 - 2 * t ** 2 + t
        h01 = -2 * t ** 3 + 3 * t ** 2
        h11 = t ** 3 - t ** 2
        out_x.append(wl[i] + t * hh)
        out_y.append(h00 * v[i] + h10 * hh * m[i]
                     + h01 * v[i + 1] + h11 * hh * m[i + 1])
    return np.concatenate(out_x), np.concatenate(out_y)


# Sensor headroom model for the saturation bar (ColorMunki Photo).
# Argyll's munki_imp.c documents its display-spot integration times
# and the luminance ceilings they are "good up to": 2.0 s → ~110 cd/m²,
# 0.3 s → ~275 cd/m², 0.1 s (adaptive readings switch down to it) →
# ~700 cd/m². Log-interpolating the app's fast non-adaptive 0.7 s
# integration between Argyll's own 2.0 s and 0.3 s figures gives
# ≈160 cd/m². spotread's spectral values are linear in radiance: a
# white lamp measured at Y = 661 cd/m² showed a spectral peak of
# 11.04 units → ≈0.0167 spectral units per cd/m² of white-range light.
# 0.0167 spectral units per cd/m² of white-range light. BUT that is
# only the ADAPTIVE time-swap trigger (munki_imp.c: "Simply measure the
# full display white, and if it's close to saturation, switch to the
# alternate display integration time" — 0.7 s -> 0.3 s -> 0.1 s), NOT
# the clip: spotread normalises readings to the integration time, so a
# source 7× brighter integrates 7× shorter and still reports ~the same
# readable values. The sensor only clips when even the SHORTEST time
# (DISP_INTT3 = 0.1 s) saturates: in reported units that is
# 11.7 × (0.7 / 0.1) ≈ 82. Fast mode uses a fixed integration and so
# its ceiling stands. These remain estimates from Argyll's own source,
# not a calibration of your unit; spotread's "Reading is saturated"
# error is still the real authority on actual clipping.
# Per-instrument clip ceilings in the units spotread PRINTS for
# spectral readings (device-neutral by construction: instruments
# without a vetted Argyll-derived model get NO estimate — the bar
# honestly shows n/a rather than a guess):
#   ColorMunki Photo (munki_imp.c): a bright 700 lx/cd·m²-class reading
#   prints ~11.7 units; adaptive readings time-swap 0.7 s -> 0.3 s ->
#   0.1 s, and the reported values are per-time normalised, so sensor
#   clip ≈ 82 units. NON-adaptive ("fast", -Y A) clips at the SAME
#   ceiling: munki_imp.c's emis_int_time calibration integrates a
#   display white and swaps 0.7 -> 0.3 -> 0.1 s even when adaptive=0
#   (mk_emiss_spot_na sets s->adaptive=0 but still carries
#   dark_int_time2/3 and done_dintsel) — the swap runs once at the
#   session's calibration instead of per reading, and since this app
#   calibrates at every start, both modes share the ~82-unit ceiling.
_SENSOR_CLIP = {
    # Per-FAMILY headroom models. Each entry says what 100 % means in
    # spotread's printed units and HOW it is substantiated — because
    # for most Argyll instruments no source-derivable ceiling exists:
    # Argyll checks raw counts against a per-instrument EEPROM
    # saturation limit (munki_imp.c:4962, i1pro_imp.c:4971) BEFORE any
    # scaling and divides the counts by each reading's integration
    # time (munki_imp.c:5013), so the printed-unit ceiling can only be
    # known where either a fixed driver conversion exists or the knee
    # was observed on the unit.
    "colormunki": {
        "ceiling": 255.0,
        # the sensor's ANALOG KNEE observed on live logs of a real
        # unit: 27 frames pinned at 255.0-255.3 while raw counts stayed
        # below satlimit (Argyll accepted readings up to 340.6, past
        # the knee, i.e. no longer linear); no software 255 clamp
        # exists in Argyll. A per-unit analog property: other
        # ColorMunkis may knee slightly differently.
        "basis": "observed analog knee (unit under test)",
    },
    "dtp20": {
        "ceiling": 200.0,
        # the one Argyll driver with a FIXED count->printed conversion:
        # spot values are printed as raw16 * 200.0/65535.0
        # (dtp20.c:578/818/1055), so 200 printed units IS the 16-bit
        # full scale — an upper bound (analog/firmware clip may occur
        # below it).
        "basis": "fixed 16-bit full-scale conversion (dtp20.c:578)",
    },
    # i1 Pro 1/2, i1 Pro 3, EX1, specbos, PhotoResearch, and every
    # colorimeter: the printed scale is per-unit calibration (EEPROM
    # emis_coef / sens_sat / irradiance cal), so no honest ceiling
    # number exists for them — they stay n/a rather than a guess.
}


def unreliable_band_lo(inst_type: str, mode_name: str) -> float | None:
    """Shortest wavelength the instrument's calibration can actually
    vouch for in this reading mode — everything below it is NOT
    measured (the ColorMunki's Argyll driver literally duplicates the
    boundary bin's filter below this limit: munki_imp.c 'Duplicate
    below this wl', WL_EMIS_MIN 400 nm emissive/ambient, WL_REF_MIN
    420 nm reflective — the values are bit-identical clones by
    construction). None when the instrument family has no vetted
    knowledge, so other devices are unaffected."""
    key = _clip_key(inst_type)
    if key != "colormunki":
        return None
    if mode_name == "Reflective":
        return 420.0
    if mode_name in ("Emissive", "Ambient"):
        return 400.0
    return None


def _clip_key(inst_type: str) -> str | None:
    """The Argyll-derivable headroom FAMILY this instrument belongs
    to, if any. Family, not model: all ColorMunkis share the munki
    driver (its relative-mode behaviour; the observed knee is a per-unit
    analog property), and every DTP20 badge shares dtp20.c's fixed
    16-bit conversion. Anything else returns None — n/a, no honest
    ceiling exists for it (the printed scale is per-unit calibration:
    i1 Pro 1/2/3, EX1, specbos, PhotoResearch units, colorimeters)."""
    low = str(inst_type or "").lower()
    if "colormunki" in low:
        return "colormunki"
    if "dtp20" in low or "dtp 20" in low:
        return "dtp20"
    return None


def saturation_fraction(raw: Spectrum | None, fast: bool = False,
                        mode_name: str = "Emissive",
                        inst_type: str = "") -> float | None:
    """Peak sensor-channel level as a fraction of the END OF LINEAR
    operation (1.0 = the sensor's analog knee; above it Argyll still
    accepts the reading but the brightest channels are no longer
    quantitative). Mode-independent by construction: Argyll divides the
    raw counts by each reading's integration time (munki_imp.c:5013),
    so printed values share one scale across adaptive and fast modes —
    which matches the instrument's behaviour (no visible difference
    between the modes for the same source).

    None whenever no honest estimate exists: instruments with no
    vetted family model (i1 Pro 1/2/3, EX1, specbos, PhotoResearch
    and all colorimeters print per-unit-calibrated values, so no
    source-derivable ceiling exists for them), and the reflective
    path, which has no honest ceiling (ambient and emissive share the
    emissive sensor path — the diffuser only replaces the aperture).
    spotread's own 'Sensor is saturated' verdict (raw counts over the
    per-instrument EEPROM saturation limit, checked before any
    scaling) remains the authority for EVERY instrument and drives
    the red flash directly."""
    if raw is None or len(raw.values) == 0 \
            or mode_name not in ("Emissive", "Ambient"):
        return None
    model = _SENSOR_CLIP.get(_clip_key(inst_type))
    if model is None:
        return None          # unknown instrument: n/a, never a guess
    peak = float(np.max(np.asarray(raw.values, dtype=float)))
    if peak <= 0.0:
        return 0.0
    return min(1.5, peak / model["ceiling"])


def radiometric_from_photometric(s: Spectrum, y: float | None) -> float | None:
    """Radiometric counterpart of an absolute photometric quantity,
    from the measured spectral SHAPE and the photometric anchor:

        X_e = X_v · ∫ S(λ) dλ / (683 · ∫ S(λ)·V(λ) dλ)

    with K_m = 683 lm/W, the maximum spectral luminous efficacy
    (at 555 nm).
    X_v is luminance in cd/m² (emissive) — then X_e is RADIANCE in
    W/(sr·m²) — or illuminance in lx (ambient diffuser, E = π·Y) —
    then X_e is IRRADIANCE in W/m². The two quantities are exact
    radiometric/photometric pairs: same geometry, weighting undone.

    Honest about where the absolute scale comes from: the instrument's
    printed spectrum is in ARBITRARY units (Argyll scales raw counts
    by per-unit coefficients), so the absolute magnitude is anchored
    entirely to the instrument's own calibrated Y (cd/m², or lx = π·Y
    on the ambient diffuser — Argyll's Lambertian-field model); the
    spectrum contributes only the shape needed to undo the photopic
    weighting (deep red/blue photons count in W but barely in lm).

    None when no spectral shape exists (XYZ-only colorimeter) or the
    photopic integral is zero (no luminous content in the band)."""
    if s is None or y is None or len(s.wavelengths) < 2 or y <= 0.0:
        return None
    wl = np.asarray(s.wavelengths, dtype=float)
    v = np.clip(np.asarray(s.values, dtype=float), 0.0, None)
    ybar = np.interp(np.clip(wl, 380.0, 780.0), _CMF_WL, _CMF_Y)
    total = float(_trapz(v, wl))
    photopic = 683.0 * float(_trapz(v * ybar, wl))
    if total <= 0.0 or photopic <= 0.0:
        return None
    return float(y) * total / photopic


def radiometric_absolute(s: Spectrum, y: float | None) -> Spectrum | None:
    """Spectral radiance / spectral irradiance ANCHORED to the
    instrument's photometric quantity — the radiometric counterpart of
    photometric_absolute:

        c(λ) = X_e · S(λ) / ∫ S dλ        so that ∫ c dλ = X_e exactly,

    with X_e = radiometric_from_photometric(S, y) (W·(sr·m²)⁻¹ radiance
    when y is the calibrated luminance cd/m²; W·m⁻² irradiance when y
    is the mode's illuminance lx = π·Y). The V(λ) weighting is thereby
    fully UNDONE: deep blue and red photons, discounted in photometry,
    carry their real energy here. The spectral SHAPE is the
    instrument's own measured curve; the absolute SCALE comes solely
    from the calibrated Y anchor (the printed spectrum itself is in
    arbitrary units — see radiometric_from_photometric).

    Returns None when there is nothing to anchor to (no spectrum,
    no Y > 0, or no photopic content in the band)."""
    if s is None or len(s.wavelengths) < 2 or y is None or y <= 0:
        return None
    wl = np.asarray(s.wavelengths, dtype=float)
    v = np.clip(np.asarray(s.values, dtype=float), 0.0, None)
    x_e = radiometric_from_photometric(s, y)
    if x_e is None:
        return None
    total = float(_trapz(v, wl))
    if total <= 0.0:
        return None
    return Spectrum(wl, v * (x_e / total), s.xyz, s.label,
                    s.timestamp, s.source)


def approx_xyz_from_spectrum(s: Spectrum) -> tuple[float, float, float]:
    """XYZ from an arbitrary-unit spectrum using the CIE 1931 2-degree
    observer at 5 nm steps (380–780 nm). Used when the instrument itself
    does not report XYZ (e.g. the simulator or files)."""
    WL, XBAR, YBAR, ZBAR = _CMF_WL, _CMF_X, _CMF_Y, _CMF_Z
    vals = np.interp(WL, s.wavelengths, s.values)
    k = 683.0  # lm/W — arbitrary here, we only need relative XYZ
    return (float(k * _trapz(vals * XBAR, WL)),
            float(k * _trapz(vals * YBAR, WL)),
            float(k * _trapz(vals * ZBAR, WL)))
