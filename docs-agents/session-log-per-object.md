# Session log — per-object reconstruction & oriented fitting

Agent notes (now committed). Everything below is merged into `main`.

## Update — current state (latest)

- **Multi-view orientation: SOLVED.** Correct convention is `side u=1-z`, `top v=z`
  (confirmed visually in Studio — eyes front, cape back, sword/tail separate). It
  is the default; Studio also has flip-side/flip-top toggles for other art tools.
  (My cell-reading detour to `side=z`/`top=1-z` was wrong — don't revisit.)
- **Selective per-object 3D-ification WORKS and is the current direction.** Flow:
  `/api/segment` → thumbnail grid → tick parts → generate only those, each via
  `image_to_cgb_selected` → `reconstruct_object` (dome) + `fit_oriented_primitives`.
  A picked shield → clean disc, sword → blade. SAM cleanly separates shield/sword/
  head/armour on the sample concept. This is high quality for ISOLATED parts.
- **Open problem = composition.** Assembling multiple parts at correct depth is
  next (single-image monocular depth is too flat; see "did NOT work" below).
  Promising lever: novel-view models (Zero123++/Stable Zero123) to synthesise a
  part's back view → carve → real per-object depth.
- `_CUBE_BIAS` (primfit) prefers cubes on near-ties; oriented fit gives rotated
  primitives. Both help isolated-part quality.

---

## Earlier overnight notes (kept for history)

## What landed

1. **Side/top carving fix (on `main`, a7a683b).** Read from the sample sheets:
   side cell faces image-right → `side u=z`; top cell head at image-top →
   `top v=1-z`. The earlier "flip side only" left side/top disagreeing on depth,
   which mirrored the side and merged the front sword with the back tail. Needs a
   human eyeball in Studio (cat_knight multiview).

2. **Cube tie-break (`primfit._CUBE_BIAS=0.05`).** A curved primitive must beat
   the cube's IoU by a clear margin to win. Fixes solid boxes losing to a
   dome-cylinder cap by a hair; round parts still pick cylinder/sphere. Makes the
   default multiview noticeably more cube-dominant (cat_knight: 7→13 cubes).
   Regression-safe (table still 1 cube + 4 cylinders).

3. **`recognition/oriented_fit.py` — OBB fitting.** PCA-align → decompose in the
   aligned frame (`_solidify` 3-axis sweep so re-voxelised rotated points read
   solid) → inverse-transform (`R_pca · prim_rot`) so primitives come back
   rotated. Verified: tilted box → rotated cubes, tilted cylinder → rotated
   cylinder, sphere → sphere. tests/test_oriented_fit.py.

4. **Per-object reconstruction now oriented.** `object_to_documents` and
   `image_to_cgb_objects` fit each object with `fit_oriented_primitives`.

## Honest findings (what did NOT work)

- **Single-image full-scene assembly is depth-limited.** Monocular depth on flat
  concept art barely separates parts in z → the assembled scene is a near-flat
  relief; oriented fitting can't add depth that isn't there.
- **Multiview `method="objects"` (carving depth + per-object oriented) fragments.**
  Front-projected object ids smear in depth; per-object × several prims → 56
  messy primitives, worse than the default whole-solid decompose (24). Kept as an
  experimental entry point, not default.

## Where it actually shines

- **Isolated object reconstruction** (`reconstruct_object`): a point-prompted
  shield → clean domed disc voxel + primitives. This is the proven win.
- **Oriented fitting** helps any part that has real 3D shape + tilt.

## Recommended next directions

- Per-object **multi-view** (each object's silhouette carved across the 4 views)
  for real per-object depth — the hard part is cross-view object correspondence
  (match front-obj to side-obj by position/size).
- Or an **interactive** flow: point-prompt objects one at a time in Studio,
  reconstruct each (works well), place with a depth/position nudge.
- Improve the default multiview decompose further (it's the best automatic path).

## Test/run state

- 78 tests pass. Studio running on this branch at http://127.0.0.1:8000.
- Models installed in `.venv`; SAM at `models/sam_vit_h_4b8939.pth`.
