# Changelog

All notable changes to CubeGB are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Oriented (OBB) primitive fitting** (`recognition/oriented_fit.py`):
  PCA-align an object to its principal axes, fit/decompose primitives there
  (free to combine several), then inverse-transform each primitive back — so a
  cube/cylinder returns *rotated* to hug a tilted part instead of being
  approximated by an upright box.
- **Per-object reconstruction** (`recognition/object_recon.py`): reconstruct a
  single object from one clean silhouette by extruding it along depth (with a
  distance-transform *dome* for a convex solid) into a voxel + fitted primitives.
  Cleanly isolates one part (e.g. a shield prompted out of a cat-knight concept)
  instead of letting the whole-scene carve fuse it into the body — the building
  block for an object-by-object scene pipeline.

### Fixed
- **Side/top carving convention** in multi-view: the side/top depth axes were
  reversed, mirroring the side (face toward the back of the head) and merging a
  front-held sword with the back tail. Correct default (confirmed on the sample
  sheets) is `side: u=1-z`, `top: v=z`. The convention varies by art tool, so
  Studio also exposes **flip-side / flip-top** toggles (multi-view step) to switch
  it per sheet. Symmetric props are unaffected.

### Added
- **Carved voxel debug view**: multi-view generation now also emits the carved
  voxel solid as a viewable `.cgb` of cubes (`occupancy_to_voxel_doc`), in the
  same world frame as the fitted primitives, with per-voxel **front colour** and
  **object-group id** (`material.name = obj{N}` from front SAM segments).
- **Studio 2×2 debug quad** (four live 3D panels): ① carved voxels (colour),
  ② final primitives, ③ pure voxels (shape only), ④ voxels coloured by object
  group. Voxels render via `InstancedMesh` (one draw call) so several panels of
  thousands of cubes stay smooth.
- **Multi-view voxel colour**: each voxel is coloured from the view facing its
  exposed surface (front/side/back/top), so sides and the back get their own
  colour instead of the front colour smeared across them. `align_views` now also
  registers the RGB so colour sampling lines up with the carved silhouette.
  Studio panel ① shows the multi-view colour, ③ the front-only colour (for
  comparison); both colours are carried on the voxel `.cgb` (multi-view in
  `material.color`, front in the cube `name` as hex).
- **Voxel resolution selector** in Studio (96–512, default 128). Carving and
  primitive-fitting resolutions are **decoupled** (`fit_res`), so a high voxel
  resolution (e.g. 256/512) keeps the voxel view crisp while primitive fitting
  stays fast (~15–20s instead of minutes).

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
