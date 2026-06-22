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

app = FastAPI(title="CubeGB Studio", version="0.1.0")


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
# Generate: image -> .cgb
# --------------------------------------------------------------------------- #
@app.post("/api/generate")
async def generate(
    image: UploadFile = File(...),
    sam_checkpoint: str = Form(""),
    depth_checkpoint: str = Form(""),
    device: str = Form("auto"),
    sam_model_type: str = Form("vit_h"),
    max_segments: int = Form(12),
) -> JSONResponse:
    """Run the recognition pipeline on an uploaded image and return a ``.cgb``."""
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

    suffix = Path(image.filename or "upload.png").suffix or ".png"
    with tempfile.TemporaryDirectory(prefix="cubegb_studio_") as tmp:
        img_path = Path(tmp) / f"input{suffix}"
        img_path.write_bytes(await image.read())
        out_path = Path(tmp) / "result.cgb"

        # Lazy import: keeps the server alive without the recognition stack.
        try:
            from recognition.fit import image_to_cgb
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
        try:
            # image_to_cgb returns a small SUMMARY and writes the full .cgb
            # document to out_path — load that document to return to the client.
            summary = image_to_cgb(
                str(img_path),
                str(out_path),
                sam_checkpoint=sam_ckpt,
                depth_checkpoint=depth_ckpt or None,
                device=dev,
                sam_model_type=sam_model_type,
                max_segments=int(max_segments),
            )
        except HTTPException:
            raise
        except Exception as exc:  # surface model/runtime errors to the UI
            raise HTTPException(status_code=400, detail=f"Generation failed: {exc}")

        doc = cgb.load(str(out_path))  # the full, schema-valid .cgb document

    return JSONResponse({"cgb": doc, "summary": summary})


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
