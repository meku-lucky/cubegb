# ComfyUI Nodes

CubeGB ships as a set of [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
custom nodes so the whole image → `.cgb` → mesh flow runs in a node graph.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/cubegb/cubegb.git
# install deps into ComfyUI's Python environment:
pip install -r cubegb/requirements.txt
# for the Generate node (recognition), also:
pip install -r cubegb/requirements-recognition.txt
```

Restart ComfyUI. The nodes appear under the **CubeGB** category. (ComfyUI Manager
/ Registry support is planned.)

Model weights for **CubeGB Generate** are downloaded separately — see
[recognition.md](recognition.md).

## Nodes

| Node | Inputs | Output | Notes |
|---|---|---|---|
| **CubeGB Generate** | `IMAGE`, `device`, `max_segments`, *(opt)* `sam_checkpoint`, `depth_checkpoint` | `CGB` | Runs the recognition pipeline on the first image of the batch → a `.cgb` document. Needs the recognition extra + model weights. |
| **CubeGB Save** | `CGB`, `filename_prefix` | `STRING` (path) | Writes a pretty-printed `.cgb` into ComfyUI's output directory. |
| **CubeGB Bake** | `CGB`, `format` (`glb`/`gltf`/`obj`), `segments`, `filename_prefix` | `STRING` (path) | Bakes the `.cgb` to a mesh in the output directory. `segments = 0` uses per-primitive defaults. |
| **CubeGB Preview** | `CGB` | `STRING` (stats), `IMAGE` | Primitive count / type breakdown / bounding box, plus a best-effort rendered thumbnail (falls back to a placeholder image if offscreen rendering is unavailable). |

The `.cgb` document is passed between nodes on a custom **`CGB`** socket type.

## Robustness

- The node pack **loads even without the recognition stack** installed — only
  `cgb` (pure JSON) is imported at module load. `torch`, SAM, Depth Anything,
  and even the baker's `trimesh` are imported lazily inside node execution, so a
  missing dependency surfaces as a clear runtime error on the relevant node
  rather than breaking ComfyUI startup.
- Save / Bake / Preview work with just the core install; only **Generate**
  requires the recognition extra.

## Typical graph

```
Load Image ─► CubeGB Generate ─► CubeGB Save     (writes .cgb)
                               └► CubeGB Bake     (writes .glb/.obj)
                               └► CubeGB Preview  (stats + thumbnail)
```
