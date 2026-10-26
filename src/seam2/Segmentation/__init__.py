"""Segmentation: segment heights with joint overlap, segment meshes, male-female joint bands."""
from .segment import segment_ranges, mesh_from_sections  # noqa: F401
from .joint import layer_band, outer_offset  # noqa: F401
