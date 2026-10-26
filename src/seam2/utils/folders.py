"""Folder management and JSON exchange with Rhino / Grasshopper."""
import json
import os

__all__ = ["ensure_folder", "save_json", "load_json"]


def ensure_folder(path):
    """Create the folder if it does not exist; returns the path."""
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def save_json(data, path):
    ensure_folder(os.path.dirname(os.path.abspath(path)))
    with open(path, "w") as fh:
        json.dump(data, fh)


def load_json(path):
    with open(path) as fh:
        return json.load(fh)
