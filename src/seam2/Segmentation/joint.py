"""Male-female joint between two stacked segments (planar linear connection).

* socket = the top band of segment k (depth D): only the outer wall is printed, an open
  sleeve. Walls that stop are always printable.
* spigot = the bottom band of segment k+1 (length D, overlapping the socket): the full
  structure with the outer wall moved in by bead width W + clearance, so it slides into
  the sleeve; above it the wall goes back out to the form over a ramp, gently enough to
  print without support.

The other way round (socket at the bottom) cannot be printed: the inner walls would
start in mid-air. 26 Oct 2026 (GlossColumn): D 20 mm, W 3.5 mm, clearance 0.3 mm, ramp 4 mm.
"""
from __future__ import division

__all__ = ["layer_band", "outer_offset"]


def outer_offset(z_rel, spigot_depth, bead_width, clearance=0.3, ramp=4.0):
    """How far (mm) the outer wall moves in at height z_rel above the segment bottom.

    >>> outer_offset(10.0, 20.0, 3.5)
    3.8
    >>> round(outer_offset(22.0, 20.0, 3.5), 2)
    1.9
    >>> outer_offset(30.0, 20.0, 3.5)
    0.0
    """
    if spigot_depth <= 0:
        return 0.0
    off = bead_width + clearance
    if z_rel <= spigot_depth:
        return off
    return off * max(0.0, 1.0 - (z_rel - spigot_depth) / float(ramp))


def layer_band(z, z0, z1, socket_depth=0.0, spigot_depth=0.0, bead_width=3.5, clearance=0.3, ramp=4.0):
    """("socket" or "ring", outer offset mm) of a layer at mean height z in segment z0..z1.

    >>> layer_band(395.0, 0.0, 400.0, socket_depth=20.0)
    ('socket', 0.0)
    >>> layer_band(385.0, 380.0, 800.0, socket_depth=20.0, spigot_depth=20.0)
    ('ring', 3.8)
    """
    socket = socket_depth > 0 and z1 - z < socket_depth
    off = outer_offset(z - z0, spigot_depth, bead_width, clearance, ramp)
    return ("socket" if socket else "ring"), off
