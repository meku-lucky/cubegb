# HANDOFF — Direction 1: image-to-3D → per-part → editable `.cgb`

> **For the next agent (Windows + NVIDIA GPU).** This continues work started on a
> macOS box. The non-trivial blocker on the Mac was simply that the
> single-image-to-3D model wouldn't install (Python 3.14, no CUDA, `torchmcubes`
> won't compile). On Windows+CUDA that disappears. Everything *around* the model
> is already built and unit-tested; your job is to slot the model in and measure
> the quality ceiling.

## TL;DR — what you're doing

End-to-end goal:

```
image → SAM (part masks) → per part: single-image-to-3D model (real volume)
      → recognition.compose.part_from_mesh (voxelise via the bridge)
      → recognition.compose.compose_parts (place by 2D position + depth)
      → editable .cgb  →  compare against the hand-authored quality bar
```

The **quality bar** is `samples/cat_knight_master.cgb` (open it in the viewer).
The question we're answering: *can existing libraries (no LLM, no training) reach
that semantic, part-aware blockout quality?* If yes → product. If no → the user
will start a separate project to train a dedicated image→`.cgb` model (out of
scope here).

## Why per-part (read this, it's the whole design)

Whole-shape geometric abstraction **loses semantics**: feeding a 40-part character
mesh to the primitive fitter collapses it into ~20 geometric blobs (verified — see
the commit `feat: mesh_fit`). A clean *single* object abstracts faithfully, though.
So the design is **per part**: segment first, reconstruct/abstract each part on its
own (identity preserved), then compose with correct depth. That's why the model is
run **per SAM part**, not on the whole image.

## What's already built & tested (don't rebuild)

| Module | What it gives you | Tested |
|---|---|---|
| `recognition/mesh_fit.py` | dense mesh → voxel occupancy → primitives → `.cgb` (model-agnostic bridge) | `tests/test_mesh_fit.py` |
| `recognition/compose.py` | per-part composition with **depth** placement | `tests/test_compose.py` |
| `recognition/segment.py` | `Segmenter`, `load_image_rgb` (SAM) | existing |
| `recognition/depth.py` | `DepthEstimator(...).estimate(img)` (Depth Anything) | existing |
| `recognition/object_recon.py` | `partition_objects`, `reconstruct_object` | existing |
| `bake/baker.py` | `.cgb` → mesh (for rendering/compare) | existing |

Run `python -m pytest` first — all should pass (≈155 tests). If they don't, fix the
environment before anything else.

### Exact API seam (signatures you'll call)

```python
# recognition/compose.py
part_from_mesh(mesh, mask, rgb=None, *, obj_id=0, z=None, res=64, up="y") -> dict
part_from_silhouette(mask, rgb=None, *, obj_id=0, z=None, res=72, depth_frac=0.35) -> dict
compose_parts(parts, *, target_size=1.5, depth_span=0.6, ground=True,
              per_object_prims=8, source_image=None) -> dict   # returns a validated .cgb doc
image_to_cgb_composed(image_path, out_path, *, sam_checkpoint, depth_checkpoint=None,
                      device=None, sam_model_type="vit_h", ...) -> dict  # SILHOUETTE baseline (no 3D model)

# a "part" dict (what compose_parts consumes):
#   {"id": int, "occ": (R,R,R) bool, "bbox": (x0,x1,y0,y1), "z": float|None,
#    "color": (r,g,b) 0..1, "hw": (H, W)}
#   - occ: occupancy grid, index i -> world (i+0.5)/R - 0.5, Y-UP, centred in [-0.5,0.5]
#   - z:   relative depth in [0,1], 0=back 1=front (None = flat z=0)

# recognition/mesh_fit.py
mesh_to_occupancy(mesh, *, res=64, up="y") -> (R,R,R) bool      # the voxeliser
mesh_to_cgb(mesh_or_path, out_path, *, res, up, max_prims, target_size, ...) -> dict
# CLI:  python -m recognition.mesh_fit model.glb --out out.cgb --up y
```

`segment` + `partition_objects` give you parts:

```python
from recognition.segment import Segmenter, load_image_rgb
from recognition.object_recon import partition_objects
img = load_image_rgb(path)                     # (H,W,3) uint8
H, W = img.shape[:2]
masks = Segmenter(SAM_CKPT, model_type="vit_h", device="cuda").segment(img, max_masks=12)
objects = partition_objects(masks, H, W)       # -> list[(obj_id, bool_mask (H,W))]
```

## Environment setup (Windows + CUDA)

1. **Use Python 3.10 or 3.11** (NOT 3.14 — that was the Mac blocker; ML wheels lag).
2. Core + recognition:
   ```
   python -m pip install -r requirements.txt -r requirements-recognition.txt
   ```
   (Includes `manifold3d` for boolean baking and the SAM/Depth stack.)
3. Install **one** single-image-to-3D model. Recommended order (simplest first):
   - **TripoSR** — fast feed-forward, one mesh per call. `pip install` from the repo;
     needs `torchmcubes` (compiles fine with CUDA toolkit) — or use a fork that
     falls back to `skimage.measure.marching_cubes`.
   - **InstantMesh** — higher quality, heavier (multi-view diffusion + LRM).
   - **Hunyuan3D-2 / TRELLIS** — SOTA, heaviest; only if the above underperform.
   Verify CUDA: `python -c "import torch; print(torch.cuda.is_available())"` → `True`.
4. SAM weights: `models/sam_vit_h_4b8939.pth` (see `README.md` for the URL). On
   Windows/CUDA SAM runs on GPU (the Mac CPU-forcing was an MPS-only workaround).

## The task — step by step

**Step 0 — baseline (no 3D model).** Run the silhouette-only composer to see the
floor and confirm the plumbing:
```python
from recognition.compose import image_to_cgb_composed
image_to_cgb_composed("test_images/cat_knight_concept.png", "out/baseline.cgb",
                      sam_checkpoint="models/sam_vit_h_4b8939.pth", device="cuda")
```
View `out/baseline.cgb` in the viewer. Parts will be flattish (dome extrude) but
placed with depth. This is the bar to beat with real 3D.

**Step 1 — wire the image-to-3D model per part.** Write `recognition/image3d_fit.py`
(new) implementing the glue below. Keep the model behind a single function
`part_mesh(crop_rgb) -> trimesh.Trimesh` so it's swappable.

```python
# pseudo-glue — flesh out run_model() with TripoSR/InstantMesh
import numpy as np, trimesh, cgb
from recognition.segment import Segmenter, load_image_rgb
from recognition.object_recon import partition_objects
from recognition.depth import DepthEstimator
from recognition.compose import part_from_mesh, compose_parts

def isolate(img, mask, pad=0.12):
    """Mask out the part, white background, crop to bbox + padding -> RGB(A) for the model."""
    ys, xs = np.nonzero(mask)
    y0,y1,x0,x1 = ys.min(),ys.max(),xs.min(),xs.max()
    h,w = y1-y0, x1-x0
    py,px = int(h*pad)+1, int(w*pad)+1
    out = np.full_like(img, 255)
    out[mask] = img[mask]
    Y0,Y1 = max(0,y0-py), min(img.shape[0],y1+py)
    X0,X1 = max(0,x0-px), min(img.shape[1],x1+px)
    return out[Y0:Y1, X0:X1]

def run_model(crop_rgb):
    # TODO: call TripoSR / InstantMesh here; return a trimesh.Trimesh
    ...

def image_to_cgb_3d(image_path, out_path, sam_ckpt, *, device="cuda",
                    target_size=1.5, depth_span=0.7, up="y"):
    img = load_image_rgb(image_path); H,W = img.shape[:2]
    masks = Segmenter(sam_ckpt, model_type="vit_h", device=device).segment(img, max_masks=12)
    objects = partition_objects(masks, H, W)
    dmap = DepthEstimator(device=device).estimate(img).astype(float)
    dmap = (dmap-dmap.min())/max(dmap.ptp(),1e-9)        # 0..1, 1=near  (use np.ptp on numpy 2.x)
    parts = []
    for oid, sel in objects:
        mesh = run_model(isolate(img, sel))
        z = float(dmap[sel].mean())
        parts.append(part_from_mesh(mesh, sel, img, obj_id=int(oid), z=z, up=up))
    doc = compose_parts(parts, target_size=target_size, depth_span=depth_span,
                        source_image=str(image_path))
    cgb.save(doc, out_path)
    return doc
```

**Step 2 — fix orientation.** Each image-to-3D model has its own up/forward
convention. `.cgb` is **Y-up**. Render one part with the baker and check: if it's
lying down or upside down, set `part_from_mesh(..., up="z")` or `up="-y"` (those are
the supported remaps), or pre-rotate the mesh. Verify with a single clean part
(e.g. the shield) before running the whole figure.

**Step 3 — evaluate vs the bar.** Render the result and `samples/cat_knight_master.cgb`
from front / side / 3-4 (use `bake.baker` + any renderer, or the standalone
`viewer/index.html`). Judge: do parts have **real volume** (not flat discs)? Is the
**depth ordering** right (shield in front, tail behind)? Are parts **recognisable**
(head a sphere, sword a blade)? Write findings into
`docs-agents/session-log-per-object.md`.

**Step 4 — tune.** Knobs: `res` (voxel detail, 48–96), `per_object_prims` (2–4 keeps
parts clean), `depth_span` (front-back exaggeration), SAM `max_masks` /
`partition_objects` thresholds (over/under-segmentation). The `mesh_fit` colour
sampling is currently a single mean per mesh — improving per-region colour is a
nice-to-have.

## Conventions & gotchas (so you don't rediscover them)

- **`.cgb` is Y-up, right-handed, metres.** Occupancy grids are `(R,R,R)` bool,
  index `i` → world `(i+0.5)/R - 0.5`, centred in `[-0.5, 0.5]`. `compose_parts`
  handles world placement; you only provide `occ`, the 2D `bbox`, `z`, `color`.
- **numpy 2.x removed `ndarray.ptp()`** — use `np.ptp(arr)` (bit us before).
- **Depth on flat concept art is coarse** — fine for relative front/back ordering,
  not for absolute metric depth. The per-part *mesh* gives the real volume; depth
  only positions the part. If ordering looks wrong, the 3D model is the better
  signal anyway.
- **Isolate parts before the model.** These models expect a single centred object
  on a clean background. Mask + white bg + bbox crop (the `isolate()` above); for
  best results run `rembg` on the crop too.
- **Model output is often a shell** — the bridge calls `trimesh` `.fill()` to
  solidify before voxelising; if a model returns non-watertight meshes that don't
  fill, raise `res` or repair with `trimesh`’s `fill_holes()`.
- **Don't commit model weights or `output/`** (already git-ignored). Third-party
  test images stay ignored except the `*_concept.png` ones already committed.

## Acceptance criteria

- `python -m pytest` stays green.
- `image_to_cgb_3d("test_images/cat_knight_concept.png", ...)` produces a valid
  `.cgb` that opens in the viewer, with parts that have **real 3D volume** and
  **correct depth ordering**, visibly better than the Step-0 silhouette baseline.
- A written verdict in the session log: *how close to `cat_knight_master.cgb` did
  existing models get, and what's the gap?* — this is the deliverable that decides
  whether direction 1 ships or the dedicated-model project starts.

## Pointers

- Quality bar: `samples/cat_knight_master.cgb` · concept: `test_images/cat_knight_concept.png`
- Spec/conventions: `docs/cgb-format.md` · agent orientation: `AGENTS.md`
- Strategy rationale (why per-part, why not just an LLM): the latest entry in
  `docs-agents/session-log-per-object.md`.
