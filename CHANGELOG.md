# Changelog

## Unreleased

### Added

- Initial package structure.
- `LayerGeneration`: heat-method geodesic distances (`HeatGeodesic`, t = 100 h², Dirichlet Poisson),
  layer field u = D1 / (D1 + D2 A) with the thesis' distortion weight, layer count check, contour
  loops, `layers_from_mesh`, command line `python -m seam2.LayerGeneration`.
- `Segmentation`: `segment_ranges` with joint overlap, `mesh_from_sections`, male-female joint bands
  (`layer_band`, `outer_offset`: socket on top, spigot below).
- `utils`: loop alignment, signed area, loop length, JSON exchange.
- Tests (leaning cylinder) and `example/01_segment_layers.py`.
