"""Example: print layers of one segment from a distance field, and its joint bands.

A leaning, slightly twisted tube stands in for a column segment. Run in the seam2.0 env:

    python example/01_segment_layers.py

For a real form, export the segment's sections from Rhino (closed loops, bottom to top,
same point count), build the mesh with mesh_from_sections and call layers_from_mesh,
or run  python -m seam2.LayerGeneration mesh.json layers.json 1.8 1.0
"""
import math
import os

from seam2.LayerGeneration import layers_from_mesh
from seam2.Segmentation import mesh_from_sections, segment_ranges, layer_band
from seam2.utils import save_json, loop_length

HERE = os.path.dirname(os.path.abspath(__file__))


def sections(z0, z1, dz=2.0, n=120):
    rows = []
    for k in range(int(round((z1 - z0) / dz)) + 1):
        z = z0 + k * dz
        r = 50.0 + 6.0 * math.sin(z / 60.0)                       # bulging
        cx = 0.08 * z                                             # leaning
        tw = z / 400.0                                            # twisting
        rows.append([[cx + r * math.cos(2 * math.pi * i / n + tw) * (1.0 + 0.15 * math.cos(2 * math.pi * i / n)),
                      r * math.sin(2 * math.pi * i / n + tw), z] for i in range(n)])
    return rows


if __name__ == "__main__":
    W, D = 3.5, 20.0
    for k, (z0, z1, spigot, socket) in enumerate(segment_ranges(0.0, 800.0, 2, joint_depth=D)):
        res = layers_from_mesh(mesh_from_sections(sections(z0, z1)), h_max=1.8, h_min=1.0)
        bands = [layer_band(sum(p[2] for p in L) / len(L), z0, z1, D if socket else 0.0, D if spigot else 0.0, W)
                 for L in res["layers"]]
        print("segment %d: z %.0f-%.0f, way %.1f-%.1f mm, %d layers of %.2f-%.2f mm (ok %s), "
              "socket layers %d, spigot layers %d, top loop %.0f mm"
              % (k + 1, z0, z1, res["way_min"], res["way_max"], res["n"], res["h_min"], res["h_max"], res["ok"],
                 sum(b[0] == "socket" for b in bands), sum(b[1] > 0 for b in bands), loop_length(res["layers"][-1])))
        save_json(res, os.path.join(HERE, "out", "segment_%d_layers.json" % (k + 1)))
