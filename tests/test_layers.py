import math

import numpy as np
import pytest

from seam2.LayerGeneration import (HeatGeodesic, layers_from_mesh, layer_count, distortion_weight, layer_field,
                                   face_gradient_norm)
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
    S = distortion_weight(np.linspace(100, 120, 50), frequency=2.0, ratio=0.75, wave="sin")
    assert S.min() >= 0.75 - 1e-9 and S.max() <= 1.0 + 1e-9
    assert abs(S[-1] - 0.875) < 1e-9                    # x = 0 on the longest way: sin 0 -> middle


def test_layer_field_base_and_fade():
    d1 = np.array([0.0, 10.0, 50.0, 90.0, 100.0])
    d2 = 100.0 - d1
    d2[2] = 60.0                                        # one longer way, so A is not 1 everywhere
    u_top, _ = layer_field(d1, d2, frequency=1.0, ratio=0.5)
    u_bot, _ = layer_field(d1, d2, frequency=1.0, ratio=0.5, base="bottom")
    assert u_top[0] == u_bot[0] == 0.0 and abs(u_top[-1] - 1.0) < 1e-12 and abs(u_bot[-1] - 1.0) < 1e-12
    A = distortion_weight(d1 + d2, 1.0, 0.5)
    assert abs(u_top[1] - 10.0 / (10.0 + 90.0 * A[1])) < 1e-12
    assert abs(u_bot[1] - 10.0 * A[1] / (10.0 * A[1] + 90.0)) < 1e-12
    u_f, _ = layer_field(d1, d2, frequency=1.0, ratio=0.5, fade_bottom=(20.0, 10.0), fade_top=(20.0, 10.0))
    assert abs(u_f[1] - 0.1) < 1e-12 and abs(u_f[3] - 0.9) < 1e-12   # faded out near both ends


def test_face_gradient_of_height_field():
    m = mesh_from_sections(cylinder_rows())
    V = np.asarray(m["V"])
    g, area = face_gradient_norm(V, m["F"], V[:, 2] / 100.0)
    assert np.allclose(g, 0.01, rtol=1e-6) and area.min() > 0


def test_gradient_spacing_keeps_heights_with_distortion():
    rows = cylinder_rows(lean=0.3)
    kw = dict(h_max=2.4, h_min=1.0, h_layer=1.8, frequency=2.0, ratio=0.75, base="bottom",
              fade_bottom=(10.0, 40.0), fade_top=(10.0, 40.0))   # a 10 mm fade is infeasible (ratio 2.8)
    mean = layers_from_mesh(mesh_from_sections(rows), **kw)
    grad = layers_from_mesh(mesh_from_sections(rows), spacing="gradient", **kw)
    assert grad["infeasible"] == 0
    assert grad["h_min"] >= 1.0 - 1e-6 and grad["h_max"] <= 2.4 + 1e-6
    lo = min(a for a, _ in grad["level_span"])
    assert lo >= 1.0 - 1e-6
    assert abs(grad["n"] - mean["n"]) < 0.3 * mean["n"]   # about the same number of layers


def test_max_band_puts_thick_side_on_the_dot_height():
    rows = cylinder_rows(lean=0.3)
    kw = dict(h_max=2.4, h_min=1.0, h_layer=1.8, frequency=2.0, ratio=0.75, base="bottom",
              fade_bottom=(10.0, 40.0), fade_top=(10.0, 40.0), spacing="gradient")
    mean = layers_from_mesh(mesh_from_sections(rows), bands=[(0.4, 0.6, 2.4)], **kw)
    peak = layers_from_mesh(mesh_from_sections(rows), bands=[(0.4, 0.6, 2.4, "max")], **kw)
    hi_mean = max(b for (a, b), h in zip(mean["level_span"], mean["level_h"]) if h > 2.3)
    hi_peak = [b for (a, b), h in zip(peak["level_span"], peak["level_h"]) if h > 2.3]
    assert min(hi_peak) > 2.3 and max(hi_peak) <= 1.03 * 2.4 + 1e-6   # every dot layer's thick side ~2.4
    assert peak["infeasible"] == 0 and peak["h_min"] >= 1.0 - 1e-6
    assert len(hi_peak) >= 1 and hi_mean <= 2.4 + 1e-6


def test_distortion_bends_layers_but_keeps_ends():
    rows = cylinder_rows(lean=0.3)
    flat = layers_from_mesh(mesh_from_sections(rows), h_max=1.8)
    bent = layers_from_mesh(mesh_from_sections(rows), h_max=1.8, frequency=2.0, ratio=0.75, base="bottom",
                            fade_bottom=(10.0, 10.0), fade_top=(10.0, 10.0))
    k = flat["n"] // 2
    z_f = np.array([p[2] for p in flat["layers"][k]])
    z_b = np.array([p[2] for p in bent["layers"][k]])
    assert abs(z_b.mean() - z_f.mean()) > 1.0           # the middle layer is moved by the weight
    z0f = np.mean([p[2] for p in flat["layers"][1]])
    z0b = np.mean([p[2] for p in bent["layers"][1]])
    assert abs(z0f - z0b) < 0.2                         # the layers next to the bottom are not


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


def test_max_band_may_go_above_h_max():
    rows = cylinder_rows(lean=0.3)
    kw = dict(h_max=2.4, h_min=1.0, h_layer=1.8, frequency=2.0, ratio=0.75, base="bottom",
              fade_bottom=(10.0, 40.0), fade_top=(10.0, 40.0), spacing="gradient")
    res = layers_from_mesh(mesh_from_sections(rows), bands=[(0.4, 0.6, 2.8, "max")], **kw)
    hi = [b for (a, b), h in zip(res["level_span"], res["level_h"]) if h > 2.7]
    assert hi and min(hi) > 2.6 and max(hi) <= 1.03 * 2.8 + 1e-6       # thick side ~2.8 in the band
    rest = [b for (a, b), h in zip(res["level_span"], res["level_h"]) if h < 2.5]
    assert max(rest) <= 1.03 * 2.4 + 1e-6                               # outside it h_max still holds


def test_blend_ramps_the_band_height():
    rows = cylinder_rows(lean=0.3)
    kw = dict(h_max=2.4, h_min=1.0, h_layer=1.8, frequency=2.0, ratio=0.75, base="bottom",
              fade_bottom=(10.0, 40.0), fade_top=(10.0, 40.0), spacing="gradient")
    bands = [(0.0, 0.1, 1.2), (0.4, 0.6, 2.8, "max")]
    sharp = layers_from_mesh(mesh_from_sections(rows), bands=bands, **kw)["level_h"]
    soft = layers_from_mesh(mesh_from_sections(rows), bands=bands, blend=0.2, **kw)["level_h"]
    jump = lambda hs: max(abs(b - a) for a, b in zip(hs, hs[1:]))
    assert jump(sharp) > 0.9                     # 1.8 -> 2.8 in one layer without blending
    assert jump(soft) < 0.25                     # ramps over several layers with it
    assert max(soft) > 2.7 and min(soft) < 1.3   # the bands still reach their heights


def test_zero_width_band_makes_a_smooth_swell():
    rows = cylinder_rows(lean=0.3)
    kw = dict(h_max=2.4, h_min=1.0, h_layer=1.8, frequency=2.0, ratio=0.75, base="bottom",
              fade_bottom=(10.0, 40.0), fade_top=(10.0, 40.0), spacing="gradient")
    bands = [(0.0, 0.1, 1.3, "mean", 0.0, 0.4), (0.5, 0.5, 2.8, "max", 0.4, 0.3)]
    hs = layers_from_mesh(mesh_from_sections(rows), bands=bands, **kw)["level_h"]
    peak = hs.index(max(hs))
    assert max(hs) > 2.7 and min(hs[:peak]) < 1.35                       # from 1.3 up to the peak
    assert all(b >= a - 1e-9 for a, b in zip(hs[:peak], hs[1:peak + 1]))  # rises without a shelf
    assert max(abs(b - a) for a, b in zip(hs, hs[1:])) < 0.25            # and never jumps


def test_patch_is_thick_on_one_side_only():
    rows = cylinder_rows(height=200.0)
    axis = {"o": [0.0, 0.0, 0.0], "d": [0.0, 0.0, 1.0], "x": [1.0, 0.0, 0.0]}
    patch = {"axis": axis, "theta": 0.0, "width": 60.0, "zone": [0.15, 0.85], "t0": 0.5, "sigma": 0.3,
             "tilt": 0.0, "amp": 2.0}
    res = layers_from_mesh(mesh_from_sections(rows), h_max=2.7, h_min=1.0, h_layer=1.8, spacing="gradient",
                           patch=patch)

    def gap_at(theta):                      # layer heights along the line at this angle
        def off(p):
            return abs((math.atan2(p[1], p[0]) - theta + math.pi) % (2 * math.pi) - math.pi)
        z = [min(L, key=off)[2] for L in res["layers"]]
        return [b - a for a, b in zip(z, z[1:])]
    near, far = gap_at(0.0), gap_at(math.pi)
    assert max(near) > 2.55 and max(near) < 2.75     # the patch peaks at about h_max
    assert max(far) < 2.0                            # the opposite side keeps the usual height
    assert abs(sum(near) - sum(far)) < 1.0           # both sides still run bed to top
    z0 = [p[2] for p in res["layers"][0]]
    assert max(z0) - min(z0) < 0.2                   # first layer (outside the zone) still flat


def test_patch_warp_keeps_the_zone_ends():
    from seam2.LayerGeneration import patch_warp
    V = np.array([[math.cos(a), math.sin(a), 0.0] for a in np.linspace(0, 6, 7)] * 3)
    u = np.repeat([0.15, 0.5, 0.85], 7)
    patch = {"axis": {"o": [0, 0, 0], "d": [0, 0, 1], "x": [1, 0, 0]}, "theta": 0.0, "width": 90.0,
             "zone": [0.15, 0.85], "t0": 0.4, "sigma": 0.3, "amp": 2.0}
    w = patch_warp(V, u, patch)
    assert np.allclose(w[:7], 0.15) and np.allclose(w[-7:], 0.85)
    assert w[7] < 0.5                                # past a peak at t0 = 0.4: the field runs behind


def test_patch_flat_top_holds_the_peak_longer():
    from seam2.LayerGeneration.layerfield import patch_warp
    V = np.array([[1.0, 0.0, 0.0]] * 201)
    u = np.linspace(0.0, 1.0, 201)
    base = {"axis": {"o": [0, 0, 0], "d": [0, 0, 1], "x": [1, 0, 0]}, "theta": 0.0, "width": 60.0,
            "zone": [0.0, 1.0], "t0": 0.5, "sigma": 0.15, "amp": 2.0}
    def thick(p):                                    # fraction of the zone with layers > 1.3 x mean
        d = np.diff(np.interp(np.linspace(0, 1, 41), patch_warp(V, u, p), u))
        return np.mean(d > 1.3 * d.mean())
    assert thick(dict(base, flat=0.15)) > 1.5 * thick(base)
    w = patch_warp(V, u, dict(base, flat=0.15, flat_deg=40.0))
    assert abs(w[0]) < 1e-9 and abs(w[-1] - 1.0) < 1e-9


def test_patch_narrow_peak_keeps_the_strongest_band_short():
    from seam2.LayerGeneration.layerfield import patch_warp
    V = np.array([[1.0, 0.0, 0.0]] * 401)
    u = np.linspace(0.0, 1.0, 401)
    base = {"axis": {"o": [0, 0, 0], "d": [0, 0, 1], "x": [1, 0, 0]}, "theta": 0.0, "width": 60.0,
            "zone": [0.0, 1.0], "t0": 0.5, "sigma": 0.3, "amp": 2.0}
    def heights(p):                                  # layer height along the zone, as a multiple of its mean
        d = np.diff(np.interp(np.linspace(0, 1, 201), patch_warp(V, u, p), u))
        return d / d.mean()
    wide, peak = heights(base), heights(dict(base, peak_w=0.6, sigma_peak=0.08))
    assert np.sum(peak > 0.9 * peak.max()) < 0.6 * np.sum(wide > 0.9 * wide.max())   # narrower top
    assert np.sum(peak > 1.05) > 0.8 * np.sum(wide > 1.05)                           # still a wide rise


def test_patch_sigma_up_shortens_the_upper_fade():
    from seam2.LayerGeneration.layerfield import patch_warp
    V = np.array([[1.0, 0.0, 0.0]] * 401)
    u = np.linspace(0.0, 1.0, 401)
    base = {"axis": {"o": [0, 0, 0], "d": [0, 0, 1], "x": [1, 0, 0]}, "theta": 0.0, "width": 60.0,
            "zone": [0.0, 1.0], "t0": 0.3, "sigma": 0.3, "amp": 2.0}
    def heights(p):
        d = np.diff(np.interp(np.linspace(0, 1, 201), patch_warp(V, u, p), u))
        return d / d.mean()
    up, sym = heights(dict(base, sigma_up=0.1)), heights(base)
    assert abs(int(np.argmax(up)) - 60) <= 4                   # peak still at t0 = 0.3
    assert up[20] < sym[20] - 0.1                              # thinner above the patch than the symmetric bell
    assert abs(up[5] - up[0]) < 0.05                           # flat at the zone start: no kink
