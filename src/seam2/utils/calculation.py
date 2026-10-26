"""Fundamental calculations on closed polylines (lists of [x, y, z])."""
from __future__ import division

import math

__all__ = ["signed_area_xy", "loop_length", "align_loops"]


def signed_area_xy(loop):
    """Signed area in XY (positive = counter-clockwise).

    >>> signed_area_xy([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    1.0
    """
    return 0.5 * sum(loop[i - 1][0] * loop[i][1] - loop[i][0] * loop[i - 1][1] for i in range(len(loop)))


def loop_length(loop):
    """Length of the closed polyline (last point back to the first)."""
    return sum(math.sqrt(sum((loop[i][c] - loop[i - 1][c]) ** 2 for c in range(3))) for i in range(len(loop)))


def align_loops(loops):
    """Turn every loop counter-clockwise and start it at the point nearest (XY) the start of
    the loop before; a repeated closing point is dropped. Returns new lists."""
    out = []
    for L in loops:
        L = [list(p) for p in L]
        if len(L) > 1 and all(abs(L[0][c] - L[-1][c]) < 1e-9 for c in range(3)):
            L = L[:-1]
        if signed_area_xy(L) < 0:
            L = L[::-1]
        if out:
            a = out[-1][0]
            j0 = min(range(len(L)), key=lambda j: (L[j][0] - a[0]) ** 2 + (L[j][1] - a[1]) ** 2)
            L = L[j0:] + L[:j0]
        out.append(L)
    return out
