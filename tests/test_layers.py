import math

import numpy as np
import pytest

from seam2.LayerGeneration import HeatGeodesic, layers_from_mesh, layer_count, distortion_weight
from seam2.Segmentation import mesh_from_sections, segment_ranges, layer_band
from seam2.utils import align_loops, signed_area_xy


def cylinder_rows(radius=30.0, height=100.0, dz=2.0, n=90, lean=0.0):
    """Sections of a (leaning) cylinder: the centre moves `lean` mm in x per mm of height."""
    rows = []
    for k in range(int(round(height / dz)) + 1):
        z = k * dz
        rows.append([[radius * math.cos(2 * math.pi * i / n) + lean * z, radius * math.sin(2 * math.pi * i / n), z]
                     for i in range(n)])
    return rows


def test_heat_distance_on_cylinder_is_height():
    m = mesh_from_sections(cylinder_rows())
    V = np.asarray(m["V"])
    d = HeatGeodesic(V, m["F"]).distance(m["bottom"])
    assert np.allclose(d[m["bottom"]], 0.0)
    mid = (V[:, 2] > 49) & (V[:, 2] < 51)
    assert abs(d[mid].mean() - 50.0) < 1.0           # within 2 %
    assert abs(d[m["top"]].mean() - 100.0) < 2.0


def test_layers_on_cylinder_are_flat_and_sized():
    res = layers_from_mesh(mesh_from_sections(cylinder_rows()), h_max=1.8, h_min=1.0)
    assert res["ok"]
    assert res["n"] == len(res["layers"]) == int(math.ceil(res["way_max"] / 1.8 - 1e-9))
    for k in (0, res["n"] // 2):
        zs = [p[2] for p in res["layers"][k]]
        assert max(zs) - min(zs) < 0.5                  # a straight cylinder gives flat layers
        assert abs(np.mean(zs) - (k + 1) * 100.0 / res["n"]) < 1.0


def test_leaning_wall_gives_longer_way():
    res = layers_from_mesh(mesh_from_sections(cylinder_rows(lean=0.3)), h_max=1.8)
    assert res["way_max"] > res["way_min"] + 5.0       # the leaning side is longer bottom to top


def test_layer_count_flags_uneven_boundaries():
    assert layer_count(100.0, 400.0, 1.8, 1.0)[3] is False
    assert layer_count(398.0, 409.0, 1.8, 1.0)[:1] == (228,)


def test_distortion_weight_range():
    A = distortion_weight(np.linspace(100, 120, 50), frequency=2.8, ratio=0.35)
    assert A.min() >= 0.35 - 1e-9 and A.max() <= 1.0 + 1e-9
    assert np.allclose(distortion_weight([1.0, 2.0]), 1.0)


def test_align_loops_direction_and_seam():
    a = [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]]
    b = [[0, 1, 1], [1, 0, 1], [0, -1, 1], [-1, 0, 1]]      # clockwise, different start
    out = align_loops([a, b])
    assert signed_area_xy(out[1]) > 0
    assert out[1][0][:2] == [1, 0]


def test_segments_and_joint_bands():
    segs = segment_ranges(0.0, 2400.0, 6, 20.0)
    assert segs[0] == (0.0, 400.0, False, True)
    assert segs[1][0] == pytest.approx(380.0) and segs[1][2]
    assert not segs[-1][3]
    assert layer_band(390.0, 0.0, 400.0, socket_depth=20.0)[0] == "socket"
    assert layer_band(200.0, 0.0, 400.0, socket_depth=20.0) == ("ring", 0.0)
    assert layer_band(381.0, 380.0, 800.0, spigot_depth=20.0)[1] == pytest.approx(3.8)


def test_dot_bands_give_taller_layers_in_the_band():
    res = layers_from_mesh(mesh_from_sections(cylinder_rows()), h_max=2.4, h_min=1.0,
                           h_layer=1.8, bands=[(0.4, 0.6, 2.4)])
    assert res["ok"]
    zs = [np.mean([p[2] for p in L]) for L in res["layers"]]
    gaps = np.diff([0.0] + zs)
    mid = [g for z, g in zip(zs, gaps) if 45 < z < 58]
    low = [g for z, g in zip(zs, gaps) if 5 < z < 35]
    assert abs(np.mean(mid) - 2.4) < 0.15 and abs(np.mean(low) - 1.8) < 0.15
