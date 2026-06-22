# Recognition Pipeline (image → `.cgb`)

> [한국어](recognition.md) · **English**

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

> The commands below are for macOS / Linux. Where Windows differs, look for the
> **🪟 Windows** notes.

## Install dependencies

```bash
pip install -r requirements.txt -r requirements-recognition.txt
```

(This command is the same on Windows.)

- **GPU**: CUDA is used with an NVIDIA GPU, MPS on Apple Silicon — automatically.
  CPU works too, just slowly.

## Model weights (download separately)

The Python deps do **not** include model checkpoints.

### SAM (Segment Anything) — Apache-2.0

Download one (accuracy ↔ speed/size):

| Model | File | Size | URL |
|---|---|---|---|
| `vit_h` | `sam_vit_h_4b8939.pth` | ~2.4GB | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth |
| `vit_l` | `sam_vit_l_0b3195.pth` | ~1.2GB | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth |
| `vit_b` | `sam_vit_b_01ec64.pth` | ~375MB | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth |

**macOS / Linux:**

```bash
mkdir -p models
curl -L -o models/sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

**🪟 Windows** — Windows 10/11 ships `curl`, but **in PowerShell `curl` is an
alias for `Invoke-WebRequest`**, so the flags above won't work. Use `curl.exe`:

- PowerShell:

  ```powershell
  mkdir models
  curl.exe -L -o models\sam_vit_h_4b8939.pth `
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
  ```

  (Or the native PowerShell way:
  `Invoke-WebRequest -Uri https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -OutFile models\sam_vit_h_4b8939.pth`)

- Command Prompt (cmd):

  ```bat
  mkdir models
  curl -L -o models\sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
  ```

### Depth Anything V2 — auto-downloaded (nothing to do)

`transformers` fetches it from Hugging Face on first run
(`depth-anything/Depth-Anything-V2-Small-hf`). No manual step. License **varies by
variant**; verify before redistribution/commercial use. See the
[Depth Anything V2 repo](https://github.com/DepthAnything/Depth-Anything-V2).

### MiDaS (optional depth fallback) — MIT

Loaded automatically via `torch.hub`.

> CubeGB's own code is MIT, but you are responsible for complying with each
> model's license. See the model table in the [README](../README.en.md#model--data-licenses).

## Running it

### Point to the checkpoint via an environment variable (recommended)

**macOS / Linux (bash/zsh):**

```bash
export CUBEGB_SAM_CHECKPOINT="$PWD/models/sam_vit_h_4b8939.pth"
```

**🪟 Windows:**

- PowerShell:

  ```powershell
  $env:CUBEGB_SAM_CHECKPOINT = "$PWD\models\sam_vit_h_4b8939.pth"
  ```

- Command Prompt (cmd):

  ```bat
  set CUBEGB_SAM_CHECKPOINT=%CD%\models\sam_vit_h_4b8939.pth
  ```

> A value set with `set` / `$env:` only lives in **that terminal session**. For a
> persistent value, use the System Environment Variables dialog (or your
> PowerShell profile).

### CLI

**macOS / Linux:**

```bash
python -m recognition.fit IMAGE.jpg \
  --sam-checkpoint models/sam_vit_h_4b8939.pth \
  --sam-model-type vit_h \
  --out result.cgb \
  [--depth-checkpoint ...] [--depth-backend {auto,depth_anything_v2,midas}] \
  [--device cuda|cpu] [--max-segments N] [--fov 55] [--target-size 1.5]
```

**🪟 Windows** — trailing `\` line continuations won't work. Put it on **one
line**, or replace the continuation with `^` (cmd) or a backtick (`` ` ``,
PowerShell). Backslash path separators (`models\sam_vit_h_4b8939.pth`) are fine.
Example (PowerShell, one line):

```powershell
python -m recognition.fit IMAGE.jpg --sam-checkpoint models\sam_vit_h_4b8939.pth --sam-model-type vit_h --out result.cgb
```

`--sam-model-type` must match the checkpoint you downloaded (`vit_h`/`vit_l`/`vit_b`).

The result is a schema-valid `.cgb` you can open in the [viewer](viewer.md),
[bake](baker.md), or [import into Blender](blender-addon.md). To do it all in one
place, use [CubeGB Studio](studio.md).

## Troubleshooting

- **Apple Silicon (MPS):** SAM's automatic mask generator builds `float64`
  tensors, which MPS rejects (`Cannot convert a MPS Tensor to float64`). CubeGB
  therefore runs **SAM on CPU automatically** (you'll see a one-line warning);
  the depth model still uses MPS. `vit_h` on CPU is slow — use `vit_b` for quick
  iteration.
- **`open3d` won't install** on very new Python (e.g. 3.14): it's only needed for
  optional `.ply` debug export, not for generation — skip it, or use Python
  3.10–3.12 if you want point-cloud debugging.
- **Slow / out of memory:** `vit_h` is the largest SAM model. Drop to `vit_l` or
  `vit_b`, lower `--max-segments`, or downscale the input image.

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
