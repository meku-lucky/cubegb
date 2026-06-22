# Changelog

All notable changes to CubeGB are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- **Multi-view precision mode now abstracts the carved voxel solid into *varied*
  primitives** (`recognition/primfit.py`): a top-down recursive decomposition
  picks cube/cylinder/cone/sphere per part by IoU and splits at part junctions
  (1-step lookahead), replacing the cubes-only box tiling. On the sample sheets
  this cut primitive **overlap from ~0.6 to ~0.0** while raising coverage, and
  produces cylinders/cones/spheres where the shape warrants (e.g. round legs,
  domed caps). It is the default for `image_to_cgb_multiview`.

### Added
- **`.cgb` format v0.1** — JSON Schema (`cgb/schema.json`), IO + builders
  (`cgb/io.py`), and schema + semantic validation (`cgb/validate.py`).
  Primitives: `cube`, `sphere`, `cylinder`, `cone`. `operations` reserved.
- Hand-authored sample documents: chair, table, simple building.
- **Web viewer** — single-file three.js page rendering `.cgb` with
  drag-and-drop, OrbitControls, and per-primitive listing.
- **Mesh baker** — `.cgb` → glTF/GLB/OBJ with named, separable, low-poly nodes
  and a CLI (`python -m bake.baker`).
- **Blender importer add-on** — imports `.cgb` as editable native primitives
  with Y-up → Z-up conversion.
- **Recognition pipeline** — SAM segmentation, Depth Anything V2 depth,
  back-projection, and PCA-based primitive fitting → `.cgb`
  (`python -m recognition.fit`).
- **ComfyUI nodes** — CubeGB Generate / Save / Bake / Preview.
- **CubeGB Studio** — all-in-one local web GUI (FastAPI + three.js):
  image → `.cgb` → 3D view → export, on one page (`python -m app.server`).
  three.js is **vendored locally** (`app/static/vendor/three`, MIT) so Studio
  needs no CDN/internet; the 3D viewer is dynamically imported so health/load/
  export keep working even if it fails to load.
- Documentation and pytest suite (format + baker + Studio backend).

[Unreleased]: https://github.com/cubegb/cubegb
