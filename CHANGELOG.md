# Changelog

All notable changes to CubeGB are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Carved voxel debug view**: multi-view generation now also emits the carved
  voxel solid as a viewable `.cgb` of cubes (`occupancy_to_voxel_doc`), in the
  same world frame as the fitted primitives.
- **Studio 2×2 debug quad**: the 3D viewport is split into four panels — ① carved
  voxels, ② final primitives, ③④ reserved for future intermediate-stage views.

### Changed
- **Multi-view carving quality**: silhouettes are now re-centred and commonly
  scaled across the four views (`align_views`) before space carving, fixing
  off-centre / mismatched-scale sheets that previously collapsed to an almost
  empty hull (e.g. one figure went from ~3.7k to ~17k+ voxels and became a
  recognisable blockout). Default carving resolution raised 64 → 96.

## [0.0.1] - 2026-06-23

First tagged pre-release. Image → editable parametric-primitive blockout, end to
end: `.cgb` format → viewer → baker → Blender importer → recognition → ComfyUI
nodes → CubeGB Studio GUI.

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

[Unreleased]: https://github.com/meku-lucky/cubegb/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/meku-lucky/cubegb/releases/tag/v0.0.1
