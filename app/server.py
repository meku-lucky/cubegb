"""CubeGB Studio backend — FastAPI server tying the pipeline into one GUI.

Endpoints
---------
- ``GET  /``              → the Studio single-page app.
- ``GET  /api/health``    → which capabilities are available (recognition deps,
                            SAM checkpoint) so the UI can guide the user.
- ``POST /api/generate``  → multipart image upload → runs the recognition
                            pipeline → returns the ``.cgb`` document as JSON.
- ``POST /api/bake``      → a ``.cgb`` document + format → returns a baked mesh
                            file (glTF/GLB/OBJ) as a download.

Design notes
------------
- Only ``cgb`` (pure JSON) and the baker (``trimesh``, a core dep) are imported
  at module load. The heavy **recognition** stack (torch/SAM/Depth Anything) is
  imported lazily *inside* ``/api/generate`` so the server — and the view/export
  half of the app — runs without it. Missing deps or model weights surface as a
  clear HTTP 400, never a startup crash.
- ``.cgb`` download is done client-side (the browser already holds the doc); the
  server only bakes meshes.

Launch::

    python -m app.server [--host 127.0.0.1] [--port 8000] [--no-browser]
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import os
import sys
import tempfile
import threading
import webbrowser
from pathlib import Path
from typing import Optional

# Allow `python app/server.py` as well as `python -m app.server`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover - depends on environment
    raise SystemExit(
        "CubeGB Studio needs the web extra. Install it with:\n"
        "    pip install -r requirements-app.txt\n"
        f"(missing: {exc.name})"
    )

import cgb  # core, always available

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Optional defaults for model checkpoints, so the UI fields can be pre-filled.
DEFAULT_SAM_CHECKPOINT = os.environ.get("CUBEGB_SAM_CHECKPOINT", "")
DEFAULT_DEPTH_CHECKPOINT = os.environ.get("CUBEGB_DEPTH_CHECKPOINT", "")

app = FastAPI(title="CubeGB Studio", version="0.0.1")


# --------------------------------------------------------------------------- #
# Static frontend
# --------------------------------------------------------------------------- #
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


# --------------------------------------------------------------------------- #
# Health / capabilities
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health() -> JSONResponse:
    """Report which capabilities are available so the UI can hint the user."""
    have_torch = False
    try:
        import torch  # noqa: F401

        have_torch = True
    except ImportError:
        pass

    sam_path = DEFAULT_SAM_CHECKPOINT
    return JSONResponse(
        {
            "ok": True,
            "can_view": True,  # always — client-side renderer + baker
            "can_bake": True,
            "recognition_available": have_torch,
            "default_sam_checkpoint": sam_path,
            "default_sam_checkpoint_exists": bool(sam_path) and Path(sam_path).exists(),
            "default_depth_checkpoint": DEFAULT_DEPTH_CHECKPOINT,
        }
    )


# --------------------------------------------------------------------------- #
# Segment: image -> object list (for selective 3D-ification)
# --------------------------------------------------------------------------- #
# Cache the last few segmentations (image-bytes hash -> SAM masks) so /api/generate
# can reuse them instead of re-running SAM.
_SEG_CACHE: dict = {}
_SEG_ORDER: list = []


def _seg_cache_put(key: str, masks) -> None:
    if key not in _SEG_CACHE:
        _SEG_ORDER.append(key)
        while len(_SEG_ORDER) > 4:
            _SEG_CACHE.pop(_SEG_ORDER.pop(0), None)
    _SEG_CACHE[key] = masks


def _object_thumb(img, mask, *, size: int = 84) -> str:
    """A small RGBA data-URI thumbnail of one object (masked crop)."""
    import numpy as np
    from PIL import Image

    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return ""
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = np.asarray(img)[y0:y1, x0:x1]
    alpha = (np.asarray(mask)[y0:y1, x0:x1].astype(np.uint8) * 255)
    rgba = np.dstack([crop, alpha])
    im = Image.fromarray(rgba, "RGBA")
    im.thumbnail((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.post("/api/segment")
async def segment(
    image: UploadFile = File(...),
    sam_checkpoint: str = Form(""),
    device: str = Form("auto"),
    sam_model_type: str = Form("vit_h"),
    max_objects: int = Form(12),
) -> JSONResponse:
    """Segment an image into selectable objects (id + thumbnail + area)."""
    sam_ckpt = sam_checkpoint.strip() or DEFAULT_SAM_CHECKPOINT
    if not sam_ckpt or not Path(sam_ckpt).exists():
        raise HTTPException(status_code=400, detail="SAM checkpoint not found (set it in 생성 옵션).")

    data = await image.read()
    key = hashlib.sha1(data).hexdigest()
    suffix = Path(image.filename or "x.png").suffix or ".png"
    try:
        from recognition.segment import Segmenter, load_image_rgb
        from recognition.object_recon import partition_objects
    except ImportError as exc:
        raise HTTPException(status_code=400, detail=f"Recognition deps missing: {exc.name}")

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / f"img{suffix}"
        p.write_bytes(data)
        img = load_image_rgb(str(p))
    H, W = img.shape[:2]
    dev = None if device in ("", "auto") else device
    try:
        masks = Segmenter(sam_ckpt, model_type=sam_model_type, device=dev).segment(
            img, max_masks=int(max_objects))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Segmentation failed: {exc}")

    _seg_cache_put(key, masks)
    objects = []
    for i, sel in partition_objects(masks, H, W):
        import numpy as np
        ys, xs = np.nonzero(sel)
        objects.append({
            "id": int(i), "area": int(sel.sum()),
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "thumb": _object_thumb(img, sel),
        })
    objects.sort(key=lambda o: -o["area"])
    return JSONResponse({"image_hash": key, "objects": objects})


# --------------------------------------------------------------------------- #
# Generate: image -> .cgb
# --------------------------------------------------------------------------- #
@app.post("/api/generate")
async def generate(
    image: Optional[UploadFile] = File(None),
    sheet: Optional[UploadFile] = File(None),
    sam_checkpoint: str = Form(""),
    depth_checkpoint: str = Form(""),
    device: str = Form("auto"),
    sam_model_type: str = Form("vit_h"),
    max_segments: int = Form(12),
    prior_weight: float = Form(0.6),
    fg_depth_thresh: float = Form(0.15),
    ground: bool = Form(True),
    voxel_res: int = Form(128),
    flip_side: bool = Form(False),
    flip_top: bool = Form(False),
    select_ids: str = Form(""),
) -> JSONResponse:
    """Run the recognition pipeline on an uploaded image and return a ``.cgb``.

    If a 2x2 multi-view ``sheet`` is also uploaded, the **precision** (multi-view
    space-carving) path is used. If ``select_ids`` (a JSON list of object ids from
    ``/api/segment``) is given with a single image, only those objects are
    reconstructed (object-by-object mode).
    """
    import json as _json
    sel_ids = None
    if select_ids.strip():
        try:
            sel_ids = [int(i) for i in _json.loads(select_ids)]
        except Exception:
            sel_ids = None
    sam_ckpt = sam_checkpoint.strip() or DEFAULT_SAM_CHECKPOINT
    depth_ckpt = depth_checkpoint.strip() or DEFAULT_DEPTH_CHECKPOINT
    if not sam_ckpt:
        raise HTTPException(
            status_code=400,
            detail=(
                "No SAM checkpoint provided. Set a path in the form (or the "
                "CUBEGB_SAM_CHECKPOINT env var). Download a checkpoint from "
                "https://github.com/facebookresearch/segment-anything#model-checkpoints"
            ),
        )
    if not Path(sam_ckpt).exists():
        raise HTTPException(status_code=400, detail=f"SAM checkpoint not found: {sam_ckpt}")

    has_image = image is not None and image.filename
    has_sheet = sheet is not None and sheet.filename
    if not has_image and not has_sheet:
        raise HTTPException(
            status_code=400,
            detail="Provide a single image and/or a 2x2 multi-view sheet.",
        )

    with tempfile.TemporaryDirectory(prefix="cubegb_studio_") as tmp:
        out_path = Path(tmp) / "result.cgb"

        img_path = None
        img_hash = None
        if has_image:
            suffix = Path(image.filename).suffix or ".png"
            img_path = Path(tmp) / f"input{suffix}"
            img_bytes = await image.read()
            img_path.write_bytes(img_bytes)
            img_hash = hashlib.sha1(img_bytes).hexdigest()

        sheet_path = None
        if has_sheet:
            sheet_suffix = Path(sheet.filename).suffix or ".png"
            sheet_path = Path(tmp) / f"sheet{sheet_suffix}"
            sheet_path.write_bytes(await sheet.read())

        # Lazy import: keeps the server alive without the recognition stack.
        try:
            from recognition.fit import image_to_cgb
            from recognition.multiview import image_to_cgb_multiview
        except ImportError as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Recognition dependencies are not installed. Install them with:\n"
                    "    pip install -r requirements-recognition.txt\n"
                    f"(missing: {exc.name})"
                ),
            )

        dev = None if device in ("", "auto") else device
        voxel_out = Path(tmp) / "voxel.cgb"
        try:
            if sheet_path is not None:
                # Precision mode: multi-view 2x2 sheet -> space carving. Also emit
                # the carved voxel solid as a debug .cgb for the side-by-side view.
                summary = image_to_cgb_multiview(
                    str(sheet_path), str(out_path),
                    sam_checkpoint=sam_ckpt, device=dev,
                    sam_model_type=sam_model_type,
                    res=max(64, min(512, int(voxel_res))),
                    max_segments=int(max_segments),
                    prior_weight=float(prior_weight),
                    ground=bool(ground),
                    voxel_out_path=str(voxel_out),
                    flip_side=bool(flip_side),
                    flip_top=bool(flip_top),
                )
            elif sel_ids is not None:
                # Object mode: reconstruct only the picked objects (reuse cached
                # SAM masks from /api/segment) + emit the coloured voxel.
                from recognition.object_recon import image_to_cgb_objects
                summary = image_to_cgb_objects(
                    str(img_path), str(out_path),
                    sam_checkpoint=sam_ckpt, depth_checkpoint=depth_ckpt or None,
                    device=dev, sam_model_type=sam_model_type,
                    max_objects=int(max_segments), ground=bool(ground),
                    voxel_out_path=str(voxel_out),
                    masks=_SEG_CACHE.get(img_hash), select_ids=sel_ids,
                )
            else:
                # Draft mode: single image (returns a summary, writes the .cgb).
                summary = image_to_cgb(
                    str(img_path), str(out_path),
                    sam_checkpoint=sam_ckpt,
                    depth_checkpoint=depth_ckpt or None,
                    device=dev,
                    sam_model_type=sam_model_type,
                    max_segments=int(max_segments),
                    prior_weight=float(prior_weight),
                    fg_depth_thresh=float(fg_depth_thresh),
                    ground=bool(ground),
                )
        except HTTPException:
            raise
        except Exception as exc:  # surface model/runtime errors to the UI
            raise HTTPException(status_code=400, detail=f"Generation failed: {exc}")

        doc = cgb.load(str(out_path))  # the full, schema-valid .cgb document
        voxel_doc = cgb.load(str(voxel_out)) if voxel_out.exists() else None

    return JSONResponse({"cgb": doc, "voxel_cgb": voxel_doc, "summary": summary})


# --------------------------------------------------------------------------- #
# Bake: .cgb -> mesh download
# --------------------------------------------------------------------------- #
class BakeRequest(BaseModel):
    doc: dict
    format: str = "glb"
    segments: int = 0  # 0 = per-primitive default
    filename: str = "cubegb"


@app.post("/api/bake")
def bake(req: BakeRequest) -> FileResponse:
    """Bake a ``.cgb`` document to a mesh and return it as a download."""
    fmt = (req.format or "glb").lower()
    if fmt not in ("glb", "gltf", "obj"):
        raise HTTPException(status_code=400, detail=f"Unsupported format: {fmt}")

    try:
        cgb.validate(req.doc)
    except cgb.ValidationError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid .cgb: {exc}")

    from bake.baker import bake_scene  # core dep (trimesh)

    seg = int(req.segments) or None
    try:
        scene = bake_scene(req.doc, segments_override=seg)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Bake failed: {exc}")

    # Write to a temp file and stream it back; clean up after the response.
    safe_name = "".join(c for c in (req.filename or "cubegb") if c.isalnum() or c in "-_") or "cubegb"
    tmp_dir = Path(tempfile.mkdtemp(prefix="cubegb_bake_"))
    out_path = tmp_dir / f"{safe_name}.{fmt}"
    try:
        scene.export(out_path, file_type=fmt)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Export failed: {exc}")

    media = {
        "glb": "model/gltf-binary",
        "gltf": "model/gltf+json",
        "obj": "text/plain",
    }[fmt]
    return FileResponse(
        str(out_path),
        media_type=media,
        filename=out_path.name,
        background=_cleanup_task(tmp_dir),
    )


def _cleanup_task(tmp_dir: Path):
    from starlette.background import BackgroundTask

    def _rm():
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)

    return BackgroundTask(_rm)


# --------------------------------------------------------------------------- #
# Launcher
# --------------------------------------------------------------------------- #
def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="app.server", description="Launch the CubeGB Studio web GUI."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser")
    args = parser.parse_args(argv)

    try:
        import uvicorn
    except ImportError:
        print("error: uvicorn is required (pip install -r requirements-app.txt)", file=sys.stderr)
        return 1

    url = f"http://{args.host}:{args.port}/"
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"CubeGB Studio → {url}  (Ctrl+C to stop)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
