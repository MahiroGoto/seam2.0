"""Segments of a tall, stacked object: height ranges, joint overlap and the segment mesh.

Linear connection (Goto 2020): every printed piece has one planar boundary to print
on and one boundary that overlaps the next piece. For a column printed upright,
segment k is printed bottom-up on its planar bottom; its top band is the socket and
the next segment starts JOINT_D lower with a spigot that goes into it (see joint.py).
"""
from __future__ import division

from ..utils.calculation import align_loops

__all__ = ["segment_ranges", "mesh_from_sections"]


def segment_ranges(z_bottom, z_top, count, joint_depth=0.0):
    """(z0, z1, has_spigot, has_socket) of each of `count` equal segments.

    Segment k >= 2 starts joint_depth below the cut (its spigot); every segment but the
    last ends with a socket of joint_depth.

    >>> segment_ranges(0.0, 2400.0, 3, 20.0)
    [(0.0, 800.0, False, True), (780.0, 1600.0, True, True), (1580.0, 2400.0, True, False)]
    """
    h = (z_top - z_bottom) / float(count)
    out = []
    for k in range(count):
        z0 = z_bottom + k * h - (joint_depth if k > 0 else 0.0)
        z1 = z_bottom + (k + 1) * h
        out.append((z0, z1, k > 0 and joint_depth > 0, k < count - 1 and joint_depth > 0))
    return out


def mesh_from_sections(rows):
    """Triangle mesh from stacked closed sections (bottom to top), each a list of [x, y, z].

    All rows must have the same number of points; they are turned to run the same way
    round and to start next to the row below (align_loops). Returns the mesh dict used by
    layers_from_mesh: {"V", "F", "bottom", "top"}.

    >>> m = mesh_from_sections([[[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]],
    ...                         [[1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1]]])
    >>> len(m["V"]), len(m["F"]), m["top"]
    (8, 8, [4, 5, 6, 7])
    """
    rows = align_loops([list(map(list, r)) for r in rows])
    n = len(rows[0])
    if any(len(r) != n for r in rows):
        raise ValueError("all sections need the same number of points")
    V = [p for r in rows for p in r]
    F = []
    for k in range(len(rows) - 1):
        a, b = k * n, (k + 1) * n
        for i in range(n):
            j = (i + 1) % n
            F += [[a + i, a + j, b + j], [a + i, b + j, b + i]]
    return {"V": V, "F": F, "bottom": list(range(n)), "top": list(range((len(rows) - 1) * n, len(rows) * n))}
