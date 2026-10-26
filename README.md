# seam2.0

Library for segmented, non-planar layered printing (version 2): print layers from distance
fields between two boundaries of a segment, and male-female joints between stacked segments.

It follows the strategies of the MAS thesis *Non-planar boundaries for branching structures*
(Mahiro Goto, ETH Zürich 2020, library `seam`), rewritten for the GlossColumn (2026): a 2.3 m
PETG column printed in segments by a KUKA KR 10 with a 2.0 mm nozzle.

## Installation

```
conda activate seam2.0
pip install -e <path to this folder>
```

Needs numpy and scipy (no libigl): the geodesic distances use the heat method.

## What is in it

| Module | |
|---|---|
| `seam2.LayerGeneration` | `HeatGeodesic` (heat-method distances on a triangle mesh), `layer_field` (u = D1 / (D1 + D2 A), with the thesis' distortion weight A), `layer_count`, `contour_loops`, `layers_from_mesh` (everything for one segment) |
| `seam2.Segmentation` | `segment_ranges` (equal segments with joint overlap), `mesh_from_sections` (mesh from stacked closed sections), `layer_band` / `outer_offset` (socket and spigot bands of the joint) |
| `seam2.utils` | `align_loops`, `signed_area_xy`, `loop_length`, JSON exchange |

### Layer generation

On a mesh of one segment, D1 is the distance from the bottom boundary and D2 from the top
boundary. Layer k is the contour of u = D1 / (D1 + D2) at k / n. The way length D1 + D2 (bottom
to top through a point) sets the local layer height w / n, so n comes from the longest way and the
largest layer height, and `layer_count` reports when the shortest way would give less than the
smallest layer height. With the distortion weight A (`frequency`, `ratio`) the layers can be bent
in waves without changing the mesh (thesis, chapter *Layer Sequence Distortion*).

The heat method needed two changes for long segments: t = 100 x (mean edge)^2 instead of
1 x, and a Poisson solve with the distance held at 0 on the boundary. Error on a 400 mm segment: < 1 %.

### Joint (linear connection, planar)

Each segment is printed bottom-up on its planar bottom. The top band is the **socket** (outer wall
only, an open sleeve); the next segment starts that much lower with a **spigot** (full structure,
outer wall moved in by bead width + clearance, then back out over a ramp).

## Usage

```python
from seam2.Segmentation import mesh_from_sections, segment_ranges, layer_band
from seam2.LayerGeneration import layers_from_mesh

mesh = mesh_from_sections(rows)            # rows: closed sections [[x, y, z], ...], bottom to top
res = layers_from_mesh(mesh, h_max=1.8, h_min=1.0)
res["layers"], res["n"], res["h_min"], res["h_max"]
```

From Rhino (CPython in a subprocess): `python -m seam2.LayerGeneration mesh.json layers.json 1.8 1.0`.
See `example/01_segment_layers.py` and `tests/`.

## Tests

```
python -m pytest -q
python -m pytest -q --doctest-modules src/seam2
```
