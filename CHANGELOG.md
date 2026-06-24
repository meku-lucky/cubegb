# Changelog

All notable changes to CubeGB are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Boolean / CSG operations** (Deformation & Boolean spec, Priority 3): the
  top-level `operations` array is now live — `difference` / `union` /
  `intersection`, stored **declaratively** (`operands[0]` is the target, the rest
  are operands/cutters). The baker resolves the real mesh boolean **once** with
  the verified manifold3d backend (robust to coplanar faces) and drops consumed
  cutters; the web viewers draw difference cutters **semi-transparent red**
  without subtracting; the Blender add-on maps each to a native Boolean modifier.
  `cgb.difference/union/intersection/operation/add_operation` helpers, operand
  validation, `samples/keyhole_lock.cgb` (a beveled plate with a drilled
  keyhole + bolt holes), `tests/test_boolean.py`. Adds `manifold3d` to
  requirements.
- **Shear deformation** (Deformation & Boolean spec, Priority 2 — completes the
  deform set): optional `deform.shear` `[x_slope, z_slope]` tilts a primitive
  along +Y (the +Y end offsets by `slope × height`), volume-preserving — slanted
  roofs, bases, leaning posts. Shared math across baker and both viewers; the
  Blender add-on bakes object scale first so the lean stays in real units.
  `cgb.shear(x, z)` helper, composable with `taper` and `bevel`.
- **Bevel deformation** (Deformation & Boolean spec, Priority 2): optional
  `deform.bevel` (ratio `0..0.5` of the shortest edge) chamfers a cube's edges
  into a low-poly 44-triangle box (6 faces + 12 edge bevels + 8 corner tris),
  softening the blocky tone. The 24-vertex construction (with a convex
  outward-winding fix so no `fix_normals` is needed) is shared between the baker
  and both web viewers; the Blender add-on maps it to a native Bevel modifier.
  `cgb.bevel(w)` helper, composable with `taper`. The `cat_knight_deformed.cgb`
  showcase now bevels its armour plates.
- **Taper deformation** (Deformation & Boolean spec, Priority 2 — first deform):
  an optional per-primitive `deform: { "taper": [x_ratio, z_ratio] }` linearly
  scales the cross-section along +Y (the -Y end stays at 1), turning a cylinder
  into a frustum, a box into a wedge, or a thin box into a blade that narrows to a
  point. The math lives in one place (`bake/baker.py:_apply_deform`) and is mirrored
  verbatim by both web viewers and the Blender add-on. `cgb.taper(x, z)` builder
  helper. New `tests/test_deform.py`.
- **Deform showcase sample** `samples/cat_knight_deformed.cgb`: the hand-authored
  cat knight rebuilt with the new features — a tapered sword blade tapering to a
  point, a flared tabard, tapered greaves and tail segments, and curved
  partial-sweep half-cylinder pauldrons.
- **Partial sweep for cylinders & cones** (Deformation & Boolean spec, Priority 1):
  optional `sweep_start` / `sweep_end` (degrees, default `0`–`360`) draw a primitive
  for only part of a revolution — curved lids, arches, barrels, tunnels — with
  `sweep_caps` (default `true`) closing the radial cut faces into a solid wedge.
  Fully backward compatible (omit the params for the unchanged full shape). The
  arc convention (`x = r·sin θ`, `z = r·cos θ`) is shared verbatim by the baker
  (`bake/baker.py`), both web viewers, and the Blender add-on so all four agree on
  the open-arc direction; the baker emits watertight wedges. New
  `samples/treasure_chest.cgb` (half-cylinder lid) and `tests/test_partial_sweep.py`.
- **Selective per-object 3D-ification** in Studio: segment the image into parts
  (`POST /api/segment` → thumbnail grid), tick the parts you want, and Generate
  reconstructs **only those** — each part **in isolation** (`image_to_cgb_selected`
  → `reconstruct_object` + oriented fit), so a shield comes out a clean disc and a
  sword a blade instead of being squashed in a shared scene grid. SAM masks are
  cached so the follow-up generate reuses them (~1 s).
- **Oriented (OBB) primitive fitting** (`recognition/oriented_fit.py`):
  PCA-align an object to its principal axes, fit/decompose primitives there
  (free to combine several), then inverse-transform each primitive back — so a
  cube/cylinder returns *rotated* to hug a tilted part instead of being
  approximated by an upright box. Used by per-object reconstruction.
- **Cube tie-break** in primitive fitting (`primfit._CUBE_BIAS`): a curved
  primitive must beat the cube's IoU by a clear margin to be chosen, so boxy
  parts stay cubes (simplest/most editable) while genuinely round parts still
  pick cylinder/sphere/cone.
- **Studio side/top carving toggles** (multi-view step): the depth-axis
  convention varies by art tool, so *flip-side* / *flip-top* checkboxes switch it
  per sheet. The default is correct for the sample sheets.
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
