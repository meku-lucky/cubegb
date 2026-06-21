"""CubeGB image-recognition pipeline (Phase 4-5).

Turns a single RGB image into an editable parametric-primitive ``.cgb``
blockout by chaining three stages:

1. **Segmentation** (:mod:`recognition.segment`) — Segment Anything (SAM) splits
   the image into a handful of salient object regions (boolean masks).
2. **Depth + back-projection** (:mod:`recognition.depth`) — Depth Anything V2
   (MiDaS fallback) estimates a dense depth map, which is unprojected per mask
   into a 3D point cloud in the ``.cgb`` world frame (Y-up, right-handed, m).
3. **Fitting** (:mod:`recognition.fit`) — each per-segment cloud is fit to the
   best parametric primitive (cube / cylinder / cone / sphere), pose-normalised
   against the world axes, and written out as a ``.cgb`` document.

The heavy ML stack (``torch``, ``segment_anything``, ``open3d``, ``opencv``,
``scipy``) is **not** imported at module-import time. Every dependency is loaded
lazily inside the function/method that needs it, so ``import recognition`` (and
``import recognition.fit``) always succeeds even when none of the heavy wheels
are installed. Missing deps surface as a clear, actionable :class:`RuntimeError`
the moment you actually try to run the pipeline.

The public surface is exposed lazily through ``__getattr__`` for the same
reason — touching :data:`Segmenter` only imports ``segment.py`` (and therefore
``numpy``) on demand.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "Segmenter",
    "Mask",
    "DepthEstimator",
    "default_intrinsics",
    "backproject",
    "image_to_cgb",
]


def __getattr__(name: str) -> Any:  # PEP 562 lazy module attributes
    """Resolve public symbols on first access without eager heavy imports."""
    if name in ("Segmenter", "Mask"):
        from . import segment

        return getattr(segment, name)
    if name in ("DepthEstimator", "default_intrinsics", "backproject"):
        from . import depth

        return getattr(depth, name)
    if name == "image_to_cgb":
        from . import fit

        return getattr(fit, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
