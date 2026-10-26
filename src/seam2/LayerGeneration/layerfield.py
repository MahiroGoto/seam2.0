"""Layer field between two boundaries of a mesh and its contours (print layers).

After "Non-planar boundaries for branching structures" (Goto 2020, chapter Layer
Sequence Distortion), based on the interpolation of print paths by Mitropoulou,
Bernhard, Dillenburger (2020):

    D1 = distance from the bottom boundary, D2 = distance from the top boundary
    field u = D1 / (D1 + D2 A)          (0 on the bottom, 1 on the top)
    layer k = contour u = k / n

A is the distortion weight of the thesis (1 = plain interpolation):

    way length w = D1 + D2 (bottom to top through the vertex), L / S its max / min
    x = (L - w) / (L - S)
    A = (1 - ratio) / 2 * cos(frequency * pi * x) + (1 + ratio) / 2

so A swings between `ratio` and 1 as the way length changes; `frequency` sets the
number of waves along the boundary. The local layer height is about w / n, so n is
taken from the longest way and the largest layer height, and the shortest way must
still give at least the smallest layer height.
"""
from __future__ import division

import math

import numpy as np

from .caldistances import HeatGeodesic

__all__ = ["distortion_weight", "layer_field", "layer_count", "contour_loops", "layers_from_mesh"]


def distortion_weight(way, frequency=0.0, ratio=1.0):
    """The thesis' extra weight A per vertex (all 1 when frequency is 0 or ratio is 1)."""
    way = np.asarray(way, float)
    if frequency == 0 or ratio == 1:
        return np.ones_like(way)
    L, S = way.max(), way.min()
    x = (L - way) / max(L - S, 1e-9)
    return (1.0 - ratio) / 2.0 * np.cos(frequency * math.pi * x) + (1.0 + ratio) / 2.0


def layer_field(d1, d2, frequency=0.0, ratio=1.0):
    """u in [0, 1] from the two distance fields, and the way length D1 + D2."""
    d1 = np.maximum(np.asarray(d1, float), 0.0)
    d2 = np.maximum(np.asarray(d2, float), 0.0)
    way = d1 + d2
    A = distortion_weight(way, frequency, ratio)
    return d1 / np.maximum(d1 + d2 * A, 1e-12), way


def layer_count(way_min, way_max, h_max, h_min=0.0):
    """Number of layers n so that the longest way gives at most h_max per layer.

    Returns (n, h_lo, h_hi, ok): ok is False when the shortest way then gives less than h_min
    (the boundaries are too uneven for one layer sequence: split the segment or move a boundary).

    >>> layer_count(398.0, 409.0, 1.8, 1.0)[0]
    228
    """
    n = int(math.ceil(way_max / h_max - 1e-9))
    return n, way_min / n, way_max / n, way_min / n >= h_min - 1e-6


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


def layers_from_mesh(mesh, h_max=1.8, h_min=1.0, frequency=0.0, ratio=1.0, t_factor=100.0):
    """Print layers of one segment.

    Parameters
    ----------
    mesh : dict {"V": [[x, y, z], ..], "F": [[i, j, k], ..], "bottom": [ids], "top": [ids]}
        the segment surface; bottom / top = the vertices on the two boundaries.

    Returns
    -------
    dict: "layers" (point lists, bottom to top; the last one is the top boundary itself),
    "n", "way_min", "way_max", "h_min", "h_max", "ok", and the mean D1 on the top and D2 on
    the bottom (a check: both ~ the segment height for a straight segment).
    """
    V, F = np.asarray(mesh["V"], float), np.asarray(mesh["F"], int)
    bottom, top = list(mesh["bottom"]), list(mesh["top"])
    hg = HeatGeodesic(V, F, t_factor)
    d1, d2 = hg.distance(bottom), hg.distance(top)
    u, way = layer_field(d1, d2, frequency, ratio)
    inner = np.ones(len(V), bool)
    inner[bottom + top] = False
    w_min, w_max = float(way[inner].min()), float(way[inner].max())
    n, h_lo, h_hi, ok = layer_count(w_min, w_max, h_max, h_min)
    layers = [contour_loops(V, F, u, k / float(n))[0] for k in range(1, n)]
    layers.append(V[top].tolist())
    return {"layers": layers, "n": n, "way_min": w_min, "way_max": w_max, "h_min": h_lo, "h_max": h_hi,
            "ok": ok, "d1_top_mean": float(np.maximum(d1, 0)[top].mean()),
            "d2_bottom_mean": float(np.maximum(d2, 0)[bottom].mean())}
