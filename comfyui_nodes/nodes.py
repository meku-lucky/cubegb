"""ComfyUI custom nodes for the CubeGB pipeline (Phase 6).

These nodes wrap the CubeGB image -> parametric-primitive ``.cgb`` blockout
pipeline so it can be driven from a ComfyUI graph:

    [CubeGB Generate] -> CGB -> [CubeGB Save]    -> path (STRING)
                              -> [CubeGB Bake]    -> path (STRING)
                              -> [CubeGB Preview] -> stats (STRING) + IMAGE

The ``.cgb`` document is the source of truth and is passed between nodes as a
plain ``dict`` carried on a custom socket type, ``"CGB"``.

Design constraints (see Phase 6 dev request):

* **Loading must never fail.** ComfyUI imports every custom node at start-up,
  often in environments where the heavy recognition stack (torch / SAM /
  Depth-Anything) is *not* installed. Only ``cgb`` and ``bake`` (trimesh) are
  guaranteed. Therefore the recognition pipeline, ``numpy`` and ``PIL`` are
  imported **lazily, inside the executing method** — never at module scope.
  A missing dependency surfaces at *execution* time as a clear, user-readable
  exception, never as an import-time crash that would hide the whole node pack.
* **sys.path must be self-healing.** ComfyUI loads custom nodes from
  ``ComfyUI/custom_nodes/<repo>/comfyui_nodes/`` and does not add the repo root
  to ``sys.path``. We add the repo root (two levels up from this file) so the
  project packages (``cgb``, ``bake``, ``recognition``) import cleanly. This is
  done idempotently.
"""

from __future__ import annotations

import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional

# --------------------------------------------------------------------------- #
# sys.path bootstrap
# --------------------------------------------------------------------------- #
# This file lives at  <repo_root>/comfyui_nodes/nodes.py, so the repo root that
# holds the `cgb`, `bake` and `recognition` packages is two levels up. ComfyUI
# does not put that directory on sys.path, so we add it ourselves -- but only if
# the packages are not already importable (e.g. installed as a wheel), and only
# once (idempotent).
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _ensure_repo_on_path() -> None:
    """Add the CubeGB repo root to ``sys.path`` if its packages aren't found."""
    root = str(_REPO_ROOT)
    if root in sys.path:
        return
    # Only mutate sys.path when the packages can't already be resolved, to avoid
    # shadowing a deliberately installed copy of CubeGB.
    import importlib.util

    if importlib.util.find_spec("cgb") is None or importlib.util.find_spec("bake") is None:
        sys.path.insert(0, root)


_ensure_repo_on_path()

# `cgb` is a pure-JSON, zero-dependency package, so it is always safe to import
# at module load. `bake` pulls in numpy + trimesh; while those are expected in a
# ComfyUI install, we import `bake.baker` lazily (inside the baking/preview
# methods) so that node *loading* never fails if trimesh happens to be absent.
import cgb  # noqa: E402


def _import_baker() -> Any:
    """Lazily import ``bake.baker`` (numpy + trimesh) with a clear error."""
    try:
        from bake import baker

        return baker
    except Exception as exc:  # pragma: no cover - env dependent
        raise RuntimeError(
            "CubeGB baking requires numpy and trimesh, which could not be "
            f"imported in this ComfyUI environment: {exc}"
        ) from exc

# Custom ComfyUI socket type used to pass the `.cgb` document (a dict) between
# CubeGB nodes. Keeping it a distinct string prevents accidental wiring into,
# say, a STRING input.
CGB_SOCKET = "CGB"
CATEGORY = "CubeGB"


# --------------------------------------------------------------------------- #
# Directory helpers
# --------------------------------------------------------------------------- #
def _output_directory() -> str:
    """Return ComfyUI's output directory, falling back to ``./output``.

    ``folder_paths`` is provided by ComfyUI at runtime; when these nodes are
    exercised outside ComfyUI (tests, CLI), we degrade gracefully.
    """
    try:
        import folder_paths  # type: ignore

        return folder_paths.get_output_directory()
    except Exception:
        out = Path.cwd() / "output"
        out.mkdir(parents=True, exist_ok=True)
        return str(out)


def _temp_directory() -> str:
    """Return ComfyUI's temp directory, falling back to the OS temp dir."""
    try:
        import folder_paths  # type: ignore

        return folder_paths.get_temp_directory()
    except Exception:
        return tempfile.gettempdir()


def _unique_path(directory: str, prefix: str, suffix: str) -> str:
    """Build a collision-resistant ``<dir>/<prefix>_<hex>.<suffix>`` path.

    ``prefix`` may contain sub-directories (ComfyUI ``filename_prefix`` style,
    e.g. ``"cubegb/blockout"``); we create them and sanitise the leaf only.
    """
    prefix = prefix.strip() or "cubegb"
    p = Path(directory) / prefix
    p.parent.mkdir(parents=True, exist_ok=True)
    leaf = p.name or "cubegb"
    return str(p.parent / f"{leaf}_{uuid.uuid4().hex[:8]}.{suffix.lstrip('.')}")


# --------------------------------------------------------------------------- #
# CubeGB Generate
# --------------------------------------------------------------------------- #
class CubeGBGenerate:
    """Run image -> ``.cgb`` recognition on a ComfyUI ``IMAGE`` tensor.

    The recognition pipeline (``recognition.fit.image_to_cgb``) and its heavy
    dependencies (torch, SAM, numpy, PIL) are imported lazily so this node can
    *load* even when those wheels are absent; missing deps raise a clear error
    only when the node actually runs.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                # ComfyUI image tensor: torch.Tensor [B, H, W, 3], float 0..1.
                "image": ("IMAGE",),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "max_segments": ("INT", {"default": 8, "min": 1, "max": 64}),
            },
            "optional": {
                "sam_checkpoint": ("STRING", {"default": "", "multiline": False}),
                "depth_checkpoint": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = (CGB_SOCKET,)
    RETURN_NAMES = ("cgb",)
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(
        self,
        image: Any,
        device: str = "auto",
        max_segments: int = 8,
        sam_checkpoint: str = "",
        depth_checkpoint: str = "",
    ) -> tuple[dict]:
        # --- Lazy, guarded imports -------------------------------------- #
        # numpy + PIL are needed to materialise the tensor as a PNG, and the
        # recognition entrypoint pulls in the full ML stack. Import them here so
        # the node pack still loads when they are missing.
        try:
            import numpy as np
            from PIL import Image
        except Exception as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "CubeGB Generate requires numpy and Pillow to convert the input "
                f"image, but they could not be imported: {exc}"
            ) from exc

        try:
            from recognition.fit import image_to_cgb
        except Exception as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "CubeGB Generate could not import the recognition pipeline "
                "(recognition.fit.image_to_cgb). This usually means the heavy ML "
                "dependencies (torch, segment-anything, depth models) are not "
                f"installed in this ComfyUI environment. Original error: {exc}"
            ) from exc

        # --- Tensor -> uint8 RGB PNG on disk ---------------------------- #
        arr = self._tensor_to_uint8(image, np)
        tmp_dir = _temp_directory()
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)
        in_png = os.path.join(tmp_dir, f"cubegb_in_{uuid.uuid4().hex[:8]}.png")
        out_cgb = os.path.join(tmp_dir, f"cubegb_out_{uuid.uuid4().hex[:8]}.cgb")
        Image.fromarray(arr, mode="RGB").save(in_png)

        # `image_to_cgb` writes the .cgb to out_path and returns the dict.
        resolved_device = None if device == "auto" else device
        try:
            doc = image_to_cgb(
                in_png,
                out_cgb,
                sam_checkpoint=sam_checkpoint or None,
                depth_checkpoint=depth_checkpoint or None,
                device=resolved_device,
            )
        except Exception as exc:
            raise RuntimeError(
                f"CubeGB recognition failed while processing the image: {exc}"
            ) from exc
        finally:
            # The temp PNG is no longer needed once recognition has run.
            try:
                os.remove(in_png)
            except OSError:
                pass

        # `max_segments` is honoured here as a defensive cap in case the pipeline
        # returns more primitives than requested: keep the first N.
        if isinstance(doc, dict):
            prims = doc.get("primitives")
            if isinstance(prims, list) and len(prims) > max_segments:
                doc["primitives"] = prims[:max_segments]
        else:
            raise RuntimeError(
                "recognition.fit.image_to_cgb did not return a .cgb dict "
                f"(got {type(doc).__name__})."
            )

        return (doc,)

    @staticmethod
    def _tensor_to_uint8(image: Any, np: Any) -> Any:
        """Convert a ComfyUI IMAGE tensor ([B,H,W,3], 0..1) to a uint8 HxWx3 array.

        Only the first image of the batch is used.
        """
        # Accept either a torch tensor or anything array-like; avoid importing
        # torch directly (it may be absent), use numpy for the conversion.
        if hasattr(image, "detach"):  # torch.Tensor
            image = image.detach().cpu().numpy()
        arr = np.asarray(image)

        if arr.ndim == 4:  # [B, H, W, C] -> take first in batch
            arr = arr[0]
        if arr.ndim != 3 or arr.shape[-1] < 3:
            raise RuntimeError(
                "CubeGB Generate expected an IMAGE of shape [B,H,W,3]; got array "
                f"with shape {tuple(arr.shape)}."
            )

        arr = arr[..., :3]  # drop alpha if present
        # Float 0..1 -> uint8 0..255. Already-uint8 inputs pass through.
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0 + 0.5).astype(np.uint8)
        return np.ascontiguousarray(arr)


# --------------------------------------------------------------------------- #
# CubeGB Save
# --------------------------------------------------------------------------- #
class CubeGBSave:
    """Write a ``.cgb`` document to ComfyUI's output directory as pretty JSON."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "cgb": (CGB_SOCKET,),
                "filename_prefix": (
                    "STRING",
                    {"default": "cubegb/blockout", "multiline": False},
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "save"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def save(self, filename_prefix: str = "cubegb/blockout", **kwargs) -> tuple[str]:
        # ComfyUI passes inputs by their INPUT_TYPES key, so the document arrives
        # as the keyword ``cgb``. We receive it via **kwargs to avoid binding a
        # parameter literally named ``cgb`` (which would shadow the imported
        # ``cgb`` module used below).
        doc = kwargs.get("cgb")
        if not isinstance(doc, dict):
            raise RuntimeError("CubeGB Save received an invalid CGB document (expected a dict).")

        out_path = _unique_path(_output_directory(), filename_prefix, "cgb")
        # `cgb.save` writes pretty (indent=2) JSON with a trailing newline.
        cgb.save(doc, out_path)
        return (out_path,)


# --------------------------------------------------------------------------- #
# CubeGB Bake
# --------------------------------------------------------------------------- #
class CubeGBBake:
    """Bake a ``.cgb`` document to a glb/gltf/obj mesh in the output directory."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "cgb": (CGB_SOCKET,),
                "format": (["glb", "gltf", "obj"], {"default": "glb"}),
                # 0 => use each primitive's own `segments` (per-primitive default).
                "segments": ("INT", {"default": 0, "min": 0, "max": 256}),
                "filename_prefix": (
                    "STRING",
                    {"default": "cubegb/mesh", "multiline": False},
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "bake"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def bake(
        self,
        format: str = "glb",
        segments: int = 0,
        filename_prefix: str = "cubegb/mesh",
        **kwargs,
    ) -> tuple[str]:
        # See CubeGBSave.save: the CGB document arrives as the ``cgb`` keyword.
        doc = kwargs.get("cgb")
        if not isinstance(doc, dict):
            raise RuntimeError("CubeGB Bake received an invalid CGB document (expected a dict).")

        # Validate before baking so the user gets a clear .cgb-level error.
        try:
            cgb.validate(doc)
        except cgb.ValidationError as exc:
            raise RuntimeError(f"CubeGB Bake: the CGB document is invalid: {exc}") from exc

        seg_override: Optional[int] = segments if segments and segments > 0 else None
        out_path = _unique_path(_output_directory(), filename_prefix, format)

        # We already hold the dict in memory, so go straight through bake_scene +
        # scene.export rather than round-tripping via bake_file (which reloads
        # from disk). This mirrors bake_file's export behaviour.
        baker = _import_baker()
        try:
            scene = baker.bake_scene(doc, segments_override=seg_override)
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            scene.export(out_path, file_type=format)
        except Exception as exc:
            raise RuntimeError(f"CubeGB Bake failed to export the mesh: {exc}") from exc

        return (out_path,)


# --------------------------------------------------------------------------- #
# CubeGB Preview
# --------------------------------------------------------------------------- #
class CubeGBPreview:
    """Produce a textual stats summary and (best-effort) a preview thumbnail.

    Offscreen GL is unreliable inside ComfyUI, so rendering is best-effort:
    we bake to a temp scene and ask trimesh for a rasterised image. Any failure
    falls back to a blank placeholder image so the graph never crashes.
    """

    PREVIEW_SIZE = 512  # square preview resolution (pixels)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "cgb": (CGB_SOCKET,),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("stats", "image")
    FUNCTION = "preview"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def preview(self, **kwargs) -> tuple[str, Any]:
        # See CubeGBSave.save: the CGB document arrives as the ``cgb`` keyword.
        doc = kwargs.get("cgb")
        if not isinstance(doc, dict):
            raise RuntimeError("CubeGB Preview received an invalid CGB document (expected a dict).")

        stats = self._stats(doc)
        image = self._render(doc)
        return (stats, image)

    # -- stats ----------------------------------------------------------- #
    @staticmethod
    def _stats(doc: dict) -> str:
        """Human-readable summary: counts, type breakdown, world bounding box."""
        prims = doc.get("primitives", []) or []
        lines: list[str] = []
        lines.append(f"format: {doc.get('format', '?')} v{doc.get('version', '?')}")
        lines.append(f"units: {doc.get('units', 'meter')}")
        lines.append(f"primitives: {len(prims)}")

        # Type breakdown.
        breakdown: dict[str, int] = {}
        for p in prims:
            t = p.get("type", "?")
            breakdown[t] = breakdown.get(t, 0) + 1
        if breakdown:
            parts = ", ".join(f"{t}={n}" for t, n in sorted(breakdown.items()))
            lines.append(f"types: {parts}")

        # Bounding box from primitive positions (cheap, no meshing required).
        positions = [
            p.get("transform", {}).get("position")
            for p in prims
            if isinstance(p.get("transform", {}).get("position"), (list, tuple))
        ]
        if positions:
            mins = [min(pos[i] for pos in positions) for i in range(3)]
            maxs = [max(pos[i] for pos in positions) for i in range(3)]
            lines.append(
                "position bbox: "
                f"min=({mins[0]:.3f}, {mins[1]:.3f}, {mins[2]:.3f}) "
                f"max=({maxs[0]:.3f}, {maxs[1]:.3f}, {maxs[2]:.3f})"
            )

        return "\n".join(lines)

    # -- render ---------------------------------------------------------- #
    def _render(self, doc: dict) -> Any:
        """Best-effort rasterised preview as a ComfyUI IMAGE tensor.

        Returns a [1, H, W, 3] float tensor (torch if available, else a numpy
        array, which ComfyUI also accepts). Always succeeds: any rendering
        failure yields a neutral grey placeholder.
        """
        size = self.PREVIEW_SIZE
        try:
            import numpy as np
        except Exception:
            # Without numpy we cannot build an IMAGE tensor at all; signal clearly.
            raise RuntimeError("CubeGB Preview requires numpy to build the preview image.")

        rgb: Any = None
        try:
            baker = _import_baker()
            scene = baker.bake_scene(doc)
            # trimesh's scene.save_image() rasterises via pyglet/headless GL.
            png_bytes = scene.save_image(resolution=(size, size), visible=True)
            if png_bytes:
                from io import BytesIO

                from PIL import Image

                img = Image.open(BytesIO(png_bytes)).convert("RGB")
                rgb = np.asarray(img, dtype=np.float32) / 255.0
        except Exception:
            # Headless GL / pyglet unavailable, empty scene, etc. -> placeholder.
            rgb = None

        if rgb is None:
            # Neutral grey placeholder so downstream Preview Image nodes still work.
            rgb = np.full((size, size, 3), 0.5, dtype=np.float32)

        batched = rgb[None, ...]  # [1, H, W, 3]

        # Prefer a torch tensor (ComfyUI's canonical IMAGE type) when torch is
        # present; otherwise the numpy array is accepted by SaveImage/PreviewImage.
        try:
            import torch

            return torch.from_numpy(np.ascontiguousarray(batched))
        except Exception:
            return batched
