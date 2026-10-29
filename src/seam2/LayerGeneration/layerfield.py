"""Layer field between two boundaries of a mesh and its contours (print layers).

After "Non-planar boundaries for branching structures" (Goto 2020, chapter Layer
Sequence Distortion), based on the interpolation of print paths by Mitropoulou,
Bernhard, Dillenburger (2020):

    D1 = distance from the bottom boundary, D2 = distance from the top boundary
    field u = D1 / (D1 + D2 A)          (0 on the bottom, 1 on the top)
    layer k = contour u = k / n

A is the distortion weight of the thesis (1 = plain interpolation):

    way length w = D1 + D2 (bottom to top through the vertex), L / S its max / min
    x = (L - w) / (L - S)                       (0 on the longest way, 1 on the shortest)
    A = (1 - ratio) / 2 * cos(frequency * pi * x) + (1 + ratio) / 2      ("long way extension")
    A = (1 - ratio) / 2 * sin(frequency * pi * x) + (1 + ratio) / 2      ("short way extension")

so A swings between `ratio` (the 2020 code's minValue) and 1 as the way length changes;
`frequency` sets the number of waves along the boundary. The 2020 code (seam,
boundary/distance_calculation.py) put A on one chosen "base" boundary: the layer at time t is
where D_base (1 - t) A = D_other t, i.e.

    base = top    : u = D1 / (D1 + D2 A)        (the default here, as in the first version)
    base = bottom : u = D1 A / (D1 A + D2)

`fade` (28 Oct 2026, GlossColumn joints) blends A back to 1 near the boundaries, so the layers
there stay the plain interpolation: A' = 1 + (A - 1) s, s = smoothstep over the distance from
each boundary. The layers have to move by the whole distortion over the fade, so a short fade
squeezes them on one side (10 mm on a 100 mm leaning cylinder: 2.8 x thicker on one side than the
other, no spacing can hold that); give it a width of several times the layers' shift. The local layer height is about w / n, so n is taken from the longest way and
the largest layer height, and the shortest way must still give at least the smallest layer
height (with a distortion the real heights differ by up to 1 / ratio: check the contours).
"""
from __future__ import division

import math

import numpy as np

from .caldistances import HeatGeodesic

__all__ = ["distortion_weight", "fade_weight", "layer_field", "patch_angle", "patch_warp", "angle_wave", "layer_count", "layer_levels", "face_gradient_norm",
           "layer_levels_gradient", "contour_loops", "layers_from_mesh"]


def distortion_weight(way, frequency=0.0, ratio=1.0, wave="cos"):
    """The thesis' extra weight A per vertex (all 1 when frequency is 0 or ratio is 1).
    wave = "cos" (long way extension, 2020 default) or "sin" (short way extension)."""
    way = np.asarray(way, float)
    if frequency == 0 or ratio == 1:
        return np.ones_like(way)
    L, S = way.max(), way.min()
    x = (L - way) / max(L - S, 1e-9)
    f = np.sin if wave == "sin" else np.cos
    return np.clip((1.0 - ratio) / 2.0 * f(frequency * math.pi * x) + (1.0 + ratio) / 2.0, ratio, 1.0)


def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def fade_weight(d1, d2, fade_bottom=None, fade_top=None):
    """s in [0, 1]: 0 within fade_bottom[0] mm of the bottom (fade_top[0] of the top), rising to 1
    over the next fade_*[1] mm (smoothstep). None = no fade on that side."""
    s = np.ones_like(np.asarray(d1, float))
    for d, fd in ((d1, fade_bottom), (d2, fade_top)):
        if fd:
            start, width = fd
            s = s * _smoothstep((np.asarray(d, float) - start) / max(width, 1e-9))
    return s


def layer_field(d1, d2, frequency=0.0, ratio=1.0, base="top", wave="cos", fade_bottom=None, fade_top=None):
    """u in [0, 1] from the two distance fields, and the way length D1 + D2.
    base = which boundary's distance carries the weight A ("top" or "bottom", see the module
    notes); fade_bottom / fade_top = (start, width) mm where A is blended back to 1."""
    d1 = np.maximum(np.asarray(d1, float), 0.0)
    d2 = np.maximum(np.asarray(d2, float), 0.0)
    way = d1 + d2
    A = distortion_weight(way, frequency, ratio, wave)
    if fade_bottom or fade_top:
        A = 1.0 + (A - 1.0) * fade_weight(d1, d2, fade_bottom, fade_top)
    if base == "bottom":
        return d1 * A / np.maximum(d1 * A + d2, 1e-12), way
    return d1 / np.maximum(d1 + d2 * A, 1e-12), way


def patch_angle(V, axis):
    """Angle (rad) of every vertex round an axis {"o": point, "d": direction, "x": the 0 direction}."""
    V = np.asarray(V, float)
    o, d, x = (np.asarray(axis[k], float) for k in ("o", "d", "x"))
    d = d / np.linalg.norm(d)
    x = x - d * x.dot(d)
    x = x / np.linalg.norm(x)
    y = np.cross(d, x)
    r = V - o
    return np.arctan2(r.dot(y), r.dot(x))


def _bell(x):
    """1 at 0, 0 from |x| = 1 on, smooth (cos^2)."""
    x = np.clip(np.abs(x), 0.0, 1.0)
    return np.cos(0.5 * math.pi * x) ** 2


def patch_warp(V, u, patch, samples=400):
    """Field u with a one-sided thick patch (29 Oct 2026, GlossColumn: a height band of u goes all the
    way round, so the thick dot layers made a ring; the patch keeps them on one side).

    Inside zone = (u_a, u_b) the field is re-timed per vertex: the layer height there is multiplied by
    m = 1 + (amp - 1) a(theta) b(t), a = bell over `width` deg either side of `theta` (deg, round
    `axis`), b = bell over `sigma` either side of t0 + tilt sin(theta - theta_c) (t = fraction of the
    zone; tilt > 0 turns the patch diagonal). Each angle keeps its own total: the field goes from u_a to
    u_b at every angle, so outside the patch on that side the layers get a little thinner, the other
    sides and everything outside the zone are untouched (the joints stay as they are).
    Optional flat tops: "flat_deg" (deg) and "flat" (zone fraction) hold the bell at 1 that far either
    side of the centre before it falls over width / sigma (a bigger patch at the same peak: widening the
    bell alone thins the rest of that angle and lowers the peak, a flat top keeps more layers near it).
    Optional narrow peak: "peak_w" (0..1) of the opening goes into a second bell over "sigma_peak", the
    rest stays on the wide one over sigma: a gentle rise with a narrow strongest band (29 Oct, user: the
    dots fade in gradually, the strongest part ~30 mm tall, the patch as wide round the column).
    Optional "sigma_up": the wide bell's half-height on the zone-start side of the centre (29 Oct, user:
    the dots in the top quarter of each piece, next to the zone start; the gradual rise stays below).
    patch = {"axis": {...}, "theta", "width", "zone", "t0", "sigma", "tilt", "amp", "flat_deg", "flat",
    "peak_w", "sigma_peak", "sigma_up"}."""
    u = np.asarray(u, float)
    ua, ub = patch["zone"]
    th = patch_angle(V, patch["axis"])
    dth = (th - math.radians(patch["theta"]) + math.pi) % (2.0 * math.pi) - math.pi
    fd = math.radians(patch.get("flat_deg", 0.0))
    a = (patch["amp"] - 1.0) * _bell(np.maximum(np.abs(dth) - fd, 0.0) / math.radians(patch["width"]))
    c = patch.get("t0", 0.5) + patch.get("tilt", 0.0) * np.sin(dth)
    sig = patch.get("sigma", 0.3)
    fl = patch.get("flat", 0.0)
    inside = (u > ua) & (u < ub) & (a > 1e-9)
    if not inside.any():
        return u
    s = (np.arange(samples) + 0.5) / samples                 # midpoints of the zone fraction
    ai, ci = a[inside, None], c[inside, None]
    pw, sp = patch.get("peak_w", 0.0), patch.get("sigma_peak", sig)
    d = s[None, :] - ci
    x = np.maximum(np.abs(d) - fl, 0.0)
    sw = np.where(d < 0.0, patch.get("sigma_up", sig), sig)   # wide bell: own half-height above the centre
    b = (1.0 - pw) * _bell(x / sw) + pw * _bell(x / sp)
    inv = 1.0 / (1.0 + ai * b)  # d(new t) / d(old t), before scaling
    cum = np.concatenate([np.zeros((len(ai), 1)), np.cumsum(inv, axis=1)], axis=1)
    t = (u[inside] - ua) / (ub - ua)
    k = np.clip(t * samples, 0.0, samples - 1e-9)
    j = k.astype(int)
    g = cum[np.arange(len(j)), j] + (k - j) * inv[np.arange(len(j)), j]
    out = u.copy()
    out[inside] = ua + (ub - ua) * g / cum[:, -1]
    return out


def angle_wave(V, u, way, wave):
    """Field u with the layers waved up and down round an axis (30 Oct 2026, GlossColumn: the steep
    waving layers seen on the bore, wanted on the outside too). Every layer is lifted by
    dz = amp b(u) sin(n (theta - theta0)) mm, b = 1 in the middle, easing to 0 over fade[0] / fade[1]
    (u) at the bottom / top, so the boundaries (joints) stay as they are. In the middle all layers are
    lifted alike: the layer heights only change in the fades.
    wave = {"axis": {...}, "n", "amp" (mm), "theta" (deg, a crest), "fade": [u_bottom, u_top]}."""
    u = np.asarray(u, float)
    th = patch_angle(V, wave["axis"])
    fb, ft = wave["fade"]
    b = _smoothstep(u / max(fb, 1e-9)) * _smoothstep((1.0 - u) / max(ft, 1e-9))
    dz = wave["amp"] * b * np.cos(wave["n"] * (th - math.radians(wave["theta"])))
    return u - dz / np.maximum(np.asarray(way, float), 1e-9)


def layer_count(way_min, way_max, h_max, h_min=0.0):
    """Number of layers n so that the longest way gives at most h_max per layer.

    Returns (n, h_lo, h_hi, ok): ok is False when the shortest way then gives less than h_min
    (the boundaries are too uneven for one layer sequence: split the segment or move a boundary).

    >>> layer_count(398.0, 409.0, 1.8, 1.0)[0]
    228
    """
    n = int(math.ceil(way_max / h_max - 1e-9))
    return n, way_min / n, way_max / n, way_min / n >= h_min - 1e-6


def layer_levels(way_ref, h_layer, bands=()):
    """Field values u of the layers, bottom (excl. 0) to top (= 1), for a layer-height profile.

    Every layer is h_layer high, except inside bands = [(u_from, u_to, h), ...] (u = 0 bottom,
    1 top), where it is h: e.g. 2.4 mm bands that print as a dotted line. The step in u is
    h / way_ref; all steps are then shrunk
    a little so the last layer ends exactly on the top. way_ref = the way length the heights
    are meant for (layers_from_mesh: the mean way). Returns (levels, target heights).

    >>> lv, hs = layer_levels(100.0, 2.0, [(0.5, 0.6, 2.5)])
    >>> len(lv), round(lv[-1], 6), max(hs)
    (49, 1.0, 2.5)
    """
    u, levels, heights = 0.0, [], []
    while u < 1.0 - 1e-9:
        h = next((bd[2] for bd in bands if bd[0] <= u + 1e-9 < bd[1]), h_layer)
        u += h / way_ref
        levels.append(u)
        heights.append(h)
    s = 1.0 / levels[-1]
    return [x * s for x in levels], heights


def face_gradient_norm(V, F, f):
    """|grad f| per triangle for a vertex field f that is linear on each triangle, and the areas."""
    V = np.asarray(V, float)
    F = np.asarray(F, int)
    f = np.asarray(f, float)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(b - a, c - a)
    A2 = np.linalg.norm(n, axis=1)                       # twice the area
    nn = n / np.maximum(A2, 1e-12)[:, None]
    # grad f = sum_i f_i (n x e_i) / (2A), e_i = edge opposite vertex i
    g = (f[F[:, 0], None] * np.cross(nn, c - b) + f[F[:, 1], None] * np.cross(nn, a - c)
         + f[F[:, 2], None] * np.cross(nn, b - a)) / np.maximum(A2, 1e-12)[:, None]
    return np.linalg.norm(g, axis=1), A2 / 2.0


def layer_levels_gradient(V, F, u, h_layer, bands=(), h_min=1.0, h_max=2.4, margin=0.01, pct=0.5, pct_max=5.0,
                          blend=0.0, h_max_zone=None, mean_mask=None):
    """Field values of the layers from the real field (28 Oct 2026, for distorted fields).

    The layer between levels a and b is about (b - a) / |grad u| high, so on the faces the band
    crosses its height runs from (b - a) / g_max to (b - a) / g_min. Each step aims at the wanted
    height (h_layer, or h inside bands [(u_from, u_to, h)]) on the area-weighted mean gradient and is
    then held between h_min g_max and h_max g_min (pct / 100 - pct percentiles, so a few bad faces
    don't decide), with `margin` for the final rescaling. A band given as (u_from, u_to, h, "max")
    aims h at the THICKEST part of the layer instead of the mean (28 Oct, GlossColumn: with a
    distorted field the thick side of each layer reaches h = 2.4 and prints as dots, the thin side
    stays smooth). There the thick side is taken at the pct_max percentile of the gradient (5 %: the
    extreme 0.5 % left the printed thick side ~0.2 mm under h on piece 4) and may reach h, even above
    h_max (GC_N20R3: dots print clean up to 2.8 mm). Where both limits can't hold (the gradient
    varies more than h_max / h_min across the band) the upper one wins and the band is reported.
    blend (in u) > 0: each band fades in / out over that width OUTSIDE it (smoothstep weights, the
    step is the weighted mean of the band's and the base's steps), so the layer height ramps between
    bands instead of jumping in one layer (28 Oct, GlossColumn: sharp height edges were visible).
    A band may set its own widths: (u_from, u_to, h, mode, blend_lo, blend_hi). A zero-width band
    (u_from = u_to) with wide fades is a smooth swell to h and back; where two fades overlap with the
    same width their smoothstep weights add to 1, so one height eases straight into the next.
    h_max_zone = (u_from, u_to, h[, ramp]): h_max is h there instead (29 Oct, GlossColumn: a one-sided
    patch needs 2.85 mm, the rest of the piece keeps its 2.4 mm cap), easing back to h_max over `ramp`
    (u) outside it (smoothstep): a sudden step of the cap thinned a whole layer, a stripe round the piece.
    mean_mask (bool per face): only these faces count for the mean height (29 Oct, GlossColumn: with a
    big one-sided patch its thick faces pulled the mean up, so a "mean" band thinned the rest of the
    level, 3-6 layers ~20 % thinner all round); the h_min / h_max limits still see every face.
    Returns (levels, target heights, [(h_lo, h_hi)] per layer, number of infeasible bands)."""
    g, area = face_gradient_norm(V, F, u)
    uf = np.asarray(u, float)[np.asarray(F, int)]
    fmin, fmax = uf.min(axis=1), uf.max(axis=1)
    ok_face = g > 1e-9
    lo_lim = h_min * (1.0 + margin)

    def h_max_at(u0):
        if not h_max_zone:
            return h_max
        a, b, hz = h_max_zone[:3]
        ramp = h_max_zone[3] if len(h_max_zone) > 3 else 0.0
        d = max(a - u0, u0 - b, 0.0)                       # distance outside the zone
        if d <= 0.0:
            return hz
        if ramp <= 0.0 or d >= ramp:
            return h_max
        return h_max + (hz - h_max) * float(_smoothstep(1.0 - d / ramp))

    def stats(a, b):
        m = ok_face & (fmax >= a) & (fmin <= b)
        if not m.any():
            return None
        gm = g[m]
        mm = m & mean_mask if mean_mask is not None else m
        if not mm.any():
            mm = m
        return (np.percentile(gm, pct), np.percentile(gm, 100.0 - pct), float(np.average(g[mm], weights=area[mm])),
                np.percentile(gm, pct_max))

    def step(u0, h_t, on_max):
        """Field step from u0 for target h_t (thick side on h_t when on_max, else the mean)."""
        hm = h_max_at(u0)
        hi_lim = hm * (1.0 - margin)
        cap = max(hm, h_t) if on_max else hm             # a "max" band may go above h_max (dots, 2.8 mm)
        h_t = min(max(h_t, lo_lim), cap if on_max else hi_lim)
        st = stats(u0, u0 + 1e-6)
        du = h_t * (st[3] if on_max else st[2]) if st else 1.0 - u0
        for _ in range(4):                               # the band depends on the step: iterate
            st = stats(u0, u0 + du)
            if st is None:
                break
            g_lo, g_hi, g_mean, g_dot = st
            if on_max:                                   # thick side on h (pct_max), extreme <= cap + 3 %
                du_lo, du_hi = lo_lim * g_hi, 1.03 * cap * g_lo
                du = min(max(h_t * g_dot, du_lo), du_hi)
            else:
                du_lo, du_hi = lo_lim * g_hi, hi_lim * g_lo
                du = min(max(h_t * g_mean, du_lo), du_hi)
        return du, h_t

    def weights(u):
        """(weight, band) of every band at u; a band is full inside (u_from, u_to) and fades to 0 over
        `blend` outside it (smoothstep), so the height ramps instead of jumping. A band may carry its
        own widths below / above it as items 5 and 6: (u_from, u_to, h, mode, blend_lo, blend_hi)."""
        out = []
        for b in bands:
            lo = b[4] if len(b) > 4 else blend
            hi = b[5] if len(b) > 5 else blend
            if b[0] <= u + 1e-9 < b[1]:
                w = 1.0
            elif lo > 0 and b[0] - lo < u < b[0]:
                w = _smoothstep((u - (b[0] - lo)) / lo)
            elif hi > 0 and b[1] <= u < b[1] + hi:
                w = _smoothstep(1.0 - (u - b[1]) / hi)
            else:
                continue
            out.append((w, b))
        s = sum(w for w, _ in out)
        if s > 1.0:
            out = [(w / s, b) for w, b in out]
        return out

    levels, heights, spans, bad = [], [], [], 0
    u0 = 0.0
    while u0 < 1.0 - 1e-9:
        ws = weights(u0)
        rest = max(0.0, 1.0 - sum(w for w, _ in ws))
        parts = [(w, step(u0, b[2], len(b) > 3 and b[3] == "max")) for w, b in ws]
        if rest > 1e-9 or not parts:
            parts.append((rest if parts else 1.0, step(u0, h_layer, False)))
        du = sum(w * d for w, (d, _) in parts)
        h_t = float(sum(w * h for w, (_, h) in parts))
        st = stats(u0, u0 + du)
        if st is not None:
            g_lo, g_hi = st[0], st[1]
            if lo_lim * g_hi > h_max_at(u0) * (1.0 - margin) * g_lo:
                bad += 1
            spans.append((du / g_hi, du / g_lo))
        else:
            spans.append((h_t, h_t))
        u0 += du
        levels.append(u0)
        heights.append(h_t)
    s = 1.0 / levels[-1]
    return [x * s for x in levels], heights, [(a * s, b * s) for a, b in spans], bad


def contour_loops(V, F, f, level):
    """Closed polylines where the vertex field f equals level (marching triangles).

    Returns a list of point lists, the longest loop first.
    """
    V = np.asarray(V, float)
    f = np.asarray(f, float)
    seg = {}
    for a, b, c in np.asarray(F, int):
        cut = []
        for p, q in ((a, b), (b, c), (c, a)):
            if (f[p] < level) != (f[q] < level):
                cut.append((min(p, q), max(p, q)))
        if len(cut) == 2:
            seg.setdefault(cut[0], []).append(cut[1])
            seg.setdefault(cut[1], []).append(cut[0])

    def point(key):
        p, q = key
        s = (level - f[p]) / (f[q] - f[p])
        return (V[p] + (V[q] - V[p]) * s).tolist()

    loops, seen = [], set()
    for start in seg:
        if start in seen:
            continue
        loop, prev, cur = [start], None, start
        seen.add(start)
        while True:
            nxt = [x for x in seg[cur] if x != prev]
            if not nxt or nxt[0] == start:
                break
            prev, cur = cur, nxt[0]
            if cur in seen:
                break
            seen.add(cur)
            loop.append(cur)
        loops.append([point(k) for k in loop])

    def length(L):
        return sum(math.sqrt(sum((L[i][c] - L[i - 1][c]) ** 2 for c in range(3))) for i in range(len(L)))

    loops.sort(key=lambda L: -length(L))
    return loops


def layers_from_mesh(mesh, h_max=1.8, h_min=1.0, frequency=0.0, ratio=1.0, t_factor=100.0,
                     h_layer=None, bands=(), base="top", wave="cos", fade_bottom=None, fade_top=None,
                     spacing="mean", blend=0.0, patch=None, pct=0.5, h_max_zone=None, layer_wave=None):
    """Print layers of one segment.

    Parameters
    ----------
    mesh : dict {"V": [[x, y, z], ..], "F": [[i, j, k], ..], "bottom": [ids], "top": [ids]}
        the segment surface; bottom / top = the vertices on the two boundaries.
    h_layer, bands : optional layer-height profile (see layer_levels): every layer h_layer
        high, inside bands [(u_from, u_to, h)] h high. Without them all layers are h_max.
    frequency, ratio, base, wave, fade_bottom, fade_top : the distortion (see layer_field).
    spacing : "mean" = steps sized on the mean way (plain fields); "gradient" = steps from the real
        field gradient so every layer stays within h_min..h_max (layer_levels_gradient).
    blend : with spacing "gradient", width (u) over which each band's height fades in / out.
    patch : optional one-sided thick patch (see patch_warp); use spacing "gradient" with h_max = the
        patch's peak height, which then holds the patch's centre at h_max.
    h_max_zone : (u_from, u_to, h) where h_max is h instead ("gradient" spacing; a patch's range).
    layer_wave : optional up-and-down wave of the layers round an axis (see angle_wave).
    pct : percentile of the field gradient that the h_min / h_max limits use ("gradient" spacing); 0 =
        the extreme face (a patch's narrow centre is less than the default 0.5 % of a band's faces).

    Returns
    -------
    dict: "layers" (point lists, bottom to top; the last one is the top boundary itself),
    "n", "way_min", "way_max", "h_min", "h_max", "ok", "level_h" (target height per layer),
    and the mean D1 on the top and D2 on the bottom (a check: both ~ the segment height for
    a straight segment).
    """
    V, F = np.asarray(mesh["V"], float), np.asarray(mesh["F"], int)
    bottom, top = list(mesh["bottom"]), list(mesh["top"])
    hg = HeatGeodesic(V, F, t_factor)
    d1, d2 = hg.distance(bottom), hg.distance(top)
    u, way = layer_field(d1, d2, frequency, ratio, base, wave, fade_bottom, fade_top)
    mean_mask = None
    if patch:
        u = patch_warp(V, u, patch)
        dth = (patch_angle(V, patch["axis"]) - math.radians(patch["theta"]) + math.pi) % (2.0 * math.pi) - math.pi
        out_v = np.abs(dth) >= math.radians(patch.get("flat_deg", 0.0) + patch["width"])
        mean_mask = out_v[np.asarray(F, int)].all(axis=1)  # the mean height from outside the patch only
    if layer_wave:
        u = angle_wave(V, u, way, layer_wave)
    inner =np.ones(len(V), bool)
    inner[bottom + top] = False
    w_min, w_max = float(way[inner].min()), float(way[inner].max())
    extra = {}
    if spacing == "gradient":
        levels, level_h, spans, bad = layer_levels_gradient(V, F, u, h_layer or h_max, bands, h_min, h_max,
                                                             pct=pct, blend=blend, h_max_zone=h_max_zone,
                                                             mean_mask=mean_mask)
        n = len(levels)
        h_lo, h_hi = min(a for a, _ in spans), max(b for _, b in spans)
        ok = bad == 0
        extra = {"level_span": spans, "infeasible": bad}
    elif h_layer is None and not bands:
        n, h_lo, h_hi, ok = layer_count(w_min, w_max, h_max, h_min)
        levels, level_h = [k / float(n) for k in range(1, n + 1)], [h_hi] * n
    else:
        # heights are set on the mean way (the 2.4 mm dot effect needs ~2.4 mm on most of the wall),
        # the longest way may get up to 3 % more
        levels, level_h = layer_levels(float(way[inner].mean()), h_layer or h_max, bands)
        steps = [b - a for a, b in zip([0.0] + levels[:-1], levels)]
        n = len(levels)
        h_lo, h_hi = w_min * min(steps), w_max * max(steps)
        ok = h_lo >= h_min - 1e-6 and h_hi <= 1.03 * h_max + 1e-6
    layers = [contour_loops(V, F, u, lv)[0] for lv in levels[:-1]]
    layers.append(V[top].tolist())
    res = {"layers": layers, "n": n, "way_min": w_min, "way_max": w_max, "h_min": h_lo, "h_max": h_hi,
           "ok": ok, "level_h": level_h, "d1_top_mean": float(np.maximum(d1, 0)[top].mean()),
           "d2_bottom_mean": float(np.maximum(d2, 0)[bottom].mean())}
    res.update(extra)
    return res
