"""Command line, e.g. called from Rhino:

    python -m seam2.LayerGeneration mesh.json layers.json [H_MAX] [H_MIN] [FREQUENCY] [RATIO] [H_LAYER] [BANDS] [DISTORT]

mesh.json = {"V", "F", "bottom", "top"}; BANDS = JSON list [[u_from, u_to, h], ...] (layer-height
bands, u = 0 bottom .. 1 top); DISTORT = JSON object with any of "base" ("top" / "bottom"), "wave"
("cos" / "sin"), "fade_bottom" / "fade_top" ([start, width] mm), "spacing" ("mean" / "gradient"), "blend" (u width
over which the bands fade in / out, "gradient" spacing only), "patch" (a one-sided thick patch, see
patch_warp), "pct" (gradient percentile for the height limits, default 0.5), "h_max_zone" ([u_from, u_to, h]: h_max there), "angle_wave" (layers
waved round an axis, see angle_wave).
Writes the layers_from_mesh result and prints its numbers.
"""
import json
import sys

from seam2.LayerGeneration import layers_from_mesh
from seam2.utils import load_json, save_json


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    nums = [float(a) for a in argv[2:6]]
    h_max, h_min, freq, ratio = (nums + [1.8, 1.0, 0.0, 1.0][len(nums):])[:4]
    h_layer = float(argv[6]) if len(argv) > 6 and argv[6] not in ("", "-") else None
    bands = [tuple(b) for b in json.loads(argv[7])] if len(argv) > 7 else ()
    dist = json.loads(argv[8]) if len(argv) > 8 and argv[8] not in ("", "-") else {}
    res = layers_from_mesh(load_json(argv[0]), h_max, h_min, freq, ratio, h_layer=h_layer, bands=bands,
                           base=dist.get("base", "top"), wave=dist.get("wave", "cos"),
                           fade_bottom=dist.get("fade_bottom"), fade_top=dist.get("fade_top"),
                           spacing=dist.get("spacing", "mean"), blend=dist.get("blend", 0.0),
                           patch=dist.get("patch"), pct=dist.get("pct", 0.5),
                           h_max_zone=dist.get("h_max_zone"), layer_wave=dist.get("angle_wave"))
    save_json(res, argv[1])
    print(json.dumps({k: v for k, v in res.items() if k not in ("layers", "level_h")}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
