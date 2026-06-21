<!-- markdownlint-disable MD033 MD041 -->
<p align="center">
  <img src="images/cubegb-logo.png" alt="큐브공방 / CubeGB" width="420">
</p>
<h1 align="center">큐브공방 · CubeGB</h1>
<p align="center"><b>Image → editable parametric-primitive blockout.</b></p>
<p align="center">
  Turn a single image into a low-poly arrangement of cubes, cylinders, cones and
  spheres you can immediately edit in Blender — not a dense, un-editable mesh.
</p>
<p align="center">
  <a href="README.md">한국어</a> · <b>English</b>
</p>
<p align="center">
  <a href="#license">License: MIT</a> ·
  <a href="docs/cgb-format.md">.cgb format</a> ·
  <a href="#status">Status</a>
</p>

---

## What is CubeGB?

CubeGB ("큐브공방", *cube workshop*) is a lightweight **image-to-blockout**
generator. Given one image of a hard-surface object (furniture, buildings,
machines, props), it reconstructs it as a small set of **editable parametric
primitives** and writes a tiny `.cgb` JSON file.

**Why not just use Hunyuan3D / TRELLIS / Tripo?** Those produce high-density
textured meshes or Gaussian splats that are great to look at but painful to
*edit*. CubeGB instead targets the **blockout (greybox) stage** of a 3D
artist's workflow: it gives you clean primitives — KB-sized, axis-aligned,
named, and instantly re-editable in Blender — as a fast starting point you
refine by hand.

> **Scope.** CubeGB is intentionally specialized for **hard-surface, man-made
> objects**. It does *not* try to reconstruct organic forms (faces, animals,
> cloth), generate high-quality textures, or recover exact metric measurements
> from a single image — occluded surfaces are reasonably *estimated*.

## The core idea: `.cgb` is the source of truth

```
                 (recognition)            (bake)
   image  ───────────────────────►  .cgb  ─────────►  glTF / GLB / OBJ
                                      │
                                      │  (Blender add-on)
                                      └─────────────►  native editable primitives
```

- **`.cgb`** (parametric JSON) is the **single source of truth** — lossless,
  human-readable, `git diff`-friendly, kilobytes in size.
- Meshes (glTF/OBJ) are **derived artifacts** *baked* from `.cgb`.
- The **Blender add-on** restores `.cgb` as *real Blender primitives* (it does
  **not** bake them to mesh), so they stay grab-and-scale editable.

CubeGB is built **middle-out**: the downstream tooling (format → viewer → baker
→ importer) is complete and verifiable with hand-authored `.cgb` *first*, and
the AI recognition pipeline simply *fills in* that format. The whole skeleton
works even when recognition is imperfect.

## Repository layout

```
cubegb/
├── cgb/                 # .cgb format: JSON Schema, IO, validation
├── viewer/             # three.js single-file web viewer (index.html)
├── bake/               # .cgb → glTF/GLB/OBJ baker (low-poly)
├── blender_addon/      # Blender importer add-on (editable primitives)
├── recognition/        # image → .cgb: SAM segmentation, depth, primitive fitting
├── comfyui_nodes/      # ComfyUI custom nodes
├── samples/            # hand-authored .cgb examples (chair, table, building)
├── tests/              # pytest suite (format + baker)
└── docs/               # documentation
```

## Install

CubeGB has a light **core** (format + baker + viewer tooling) and a heavy
**recognition** extra (PyTorch + SAM + Depth Anything).

```bash
# Core: enough to author/validate .cgb and bake meshes
python -m pip install -r requirements.txt        # Python 3.10+

# (Optional) recognition pipeline — large; a GPU is recommended
python -m pip install -r requirements.txt -r requirements-recognition.txt
```

Pretrained **model weights are downloaded separately** — see
[docs/recognition.md](docs/recognition.md).

## Quickstart

**View a sample** — open [`viewer/index.html`](viewer/index.html) in a browser
and drag `samples/chair.cgb` onto the page. See [docs/viewer.md](docs/viewer.md).

**Bake a `.cgb` to a mesh:**

```bash
python -m bake.baker samples/chair.cgb --format glb --out chair.glb
python -m bake.baker samples/table.cgb --format obj --out table.obj
```

**Import into Blender** — install [`blender_addon/cubegb_import.py`](blender_addon/cubegb_import.py)
and use *File ▸ Import ▸ CubeGB (.cgb)*. See [docs/blender-addon.md](docs/blender-addon.md).

**Generate `.cgb` from an image** (needs the recognition extra + model weights):

```bash
python -m recognition.fit photo.jpg --sam-checkpoint sam_vit_h_4b8939.pth --out result.cgb
```

**In ComfyUI** — clone this repo into `ComfyUI/custom_nodes/` and use the
**CubeGB Generate / Save / Bake / Preview** nodes. See [docs/comfyui.md](docs/comfyui.md).

## Status

CubeGB is developed in phases (see [docs/cgb-format.md](docs/cgb-format.md) and
the per-component docs). Phases 0–3 (the downstream skeleton) are testable
without any ML; Phases 4–6 add recognition and packaging.

| Phase | Component | State |
|---|---|---|
| 0 | `.cgb` format, IO, validation, samples | ✅ tested |
| 1 | three.js web viewer | ✅ |
| 2 | mesh baker (glTF/OBJ) | ✅ tested |
| 3 | Blender importer add-on | ✅ |
| 4 | segmentation (SAM) + depth (Depth Anything V2) | ✅ code (needs weights) |
| 5 | primitive fitting & pose normalization → `.cgb` | ✅ code (needs weights) |
| 6 | ComfyUI custom nodes | ✅ |

Run the test suite:

```bash
python -m pytest
```

## Documentation

- [The `.cgb` format](docs/cgb-format.md) — spec & geometry conventions
- [Web viewer](docs/viewer.md)
- [Mesh baker](docs/baker.md)
- [Blender add-on](docs/blender-addon.md)
- [Recognition pipeline](docs/recognition.md)
- [ComfyUI nodes](docs/comfyui.md)
- [Contributing](CONTRIBUTING.md)

## Model & data licenses

CubeGB's own code is **MIT**. The recognition pipeline relies on third-party
pretrained models — **you are responsible for complying with their licenses**:

| Model | Use | License |
|---|---|---|
| [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) | segmentation | Apache-2.0 |
| [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) | depth | varies by variant — **verify before redistribution/commercial use** |
| [MiDaS](https://github.com/isl-org/MiDaS) | depth (fallback) | MIT |

See [docs/recognition.md](docs/recognition.md) for checkpoint download
instructions and license notes.

## License

MIT — see [LICENSE](LICENSE).

## Trademark

**“큐브공방 / CubeGB”** (the name and logo) is a registered trademark. The MIT
license covers the **source code** only; it does **not** grant rights to use the
“큐브공방 / CubeGB” name or logo. You may use the software under the MIT terms,
but please do not use the project name or logo in a way that implies endorsement
or affiliation without permission. The logo in [`images/`](images/) is provided
for referring to this project, not for redistribution as your own mark.
