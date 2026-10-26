"""Command line, e.g. called from Rhino:

    python -m seam2.LayerGeneration mesh.json layers.json [H_MAX] [H_MIN] [FREQUENCY] [RATIO]

mesh.json = {"V", "F", "bottom", "top"}; writes the layers_from_mesh result and prints its numbers.
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
    res = layers_from_mesh(load_json(argv[0]), h_max, h_min, freq, ratio)
    save_json(res, argv[1])
    print(json.dumps({k: v for k, v in res.items() if k != "layers"}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
