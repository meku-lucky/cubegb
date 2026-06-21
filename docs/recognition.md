# Recognition Pipeline (image → `.cgb`)

The recognition pipeline fills a `.cgb` from a single image. It is the AI part of
CubeGB and is intentionally **"good-enough blockout"**, not exact
reconstruction: a faster starting point than modeling from zero.

```
image ─► segment (SAM) ─► depth (Depth Anything V2) ─► back-project ─►
        per-segment point clouds ─► fit primitives + normalize pose ─► .cgb
```

Stages (`recognition/`):

| File | Stage | Does |
|---|---|---|
| `segment.py` | Phase 4 | SAM automatic mask generation → salient region masks. |
| `depth.py`   | Phase 4 | Depth Anything V2 (MiDaS fallback) → depth map; pinhole back-projection → per-segment 3D point clouds. |
| `fit.py`     | Phase 5 | PCA pose normalization + world-axis snapping; fit cube/cylinder/cone/sphere by lowest residual; occlusion/thickness recovery; write `.cgb`. |

## Install dependencies

```bash
pip install -r requirements.txt -r requirements-recognition.txt
```

A CUDA GPU is recommended; CPU works for small images, just slowly.

## Model weights (download separately)

The Python deps do **not** include model checkpoints.

- **SAM (Segment Anything)** — Apache-2.0. Download a checkpoint
  (`sam_vit_h_4b8939.pth` etc.) from the
  [SAM model checkpoints](https://github.com/facebookresearch/segment-anything#model-checkpoints).
- **Depth Anything V2** — license **varies by variant**; verify the variant you
  download before any redistribution or commercial use. See the
  [Depth Anything V2 repo](https://github.com/DepthAnything/Depth-Anything-V2).
- **MiDaS** (optional depth fallback) — MIT.

> CubeGB's own code is MIT, but you are responsible for complying with each
> model's license. See the model table in the [README](../README.md#model--data-licenses).

## CLI

```bash
python -m recognition.fit IMAGE.jpg \
  --sam-checkpoint sam_vit_h_4b8939.pth \
  --sam-model-type vit_h \
  --out result.cgb \
  [--depth-checkpoint ...] [--depth-backend {auto,depth_anything_v2,midas}] \
  [--device cuda|cpu] [--max-segments N] [--fov 55] [--target-size 1.5]
```

The result is a schema-valid `.cgb` you can open in the [viewer](viewer.md),
[bake](baker.md), or [import into Blender](blender-addon.md).

## Conventions & assumptions

- **World frame:** the pipeline emits points in CubeGB's Y-up, right-handed,
  metric frame so output is consistent with the baker/viewer/add-on.
- **Scale is ambiguous** from a single image, so the cloud is normalized into a
  sane bounding box (`--target-size`, ~1.5 m by default).
- **Occlusion:** only front surfaces are visible, so a fitted primitive's depth
  extent is padded (and its center pushed back) using a symmetry/thickness
  heuristic — this keeps boxes from being paper-thin. Hidden geometry is
  *estimated*, not measured.

If model weights or heavy dependencies are missing, the modules import fine but
raise a clear, actionable error when you actually run them.
