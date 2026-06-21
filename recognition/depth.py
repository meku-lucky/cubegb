"""Phase 4 — monocular depth estimation and camera back-projection.

This module turns a single RGB image into a dense depth map and then unprojects
masked pixels into per-segment 3D point clouds expressed in the **.cgb world
frame** (Y-up, right-handed, metres).

Two pieces:

* :class:`DepthEstimator` — wraps **Depth Anything V2** (preferred) with a
  **MiDaS** fallback, both behind lazy imports. Returns a float32 ``(H, W)``
  *relative* depth map (larger = nearer or farther depending on the model;
  we normalise to "metric-ish nearer-is-larger" so the rest of the pipeline is
  model-agnostic — see :meth:`DepthEstimator.estimate`).

* :func:`backproject` — pinhole unprojection of masked pixels to ``Nx3`` points,
  plus :func:`default_intrinsics` to synthesise a plausible camera (a single
  photo carries no real intrinsics), and :func:`save_point_cloud_ply` to dump a
  cloud for debugging.

Coordinate convention (NORMATIVE for this package)
--------------------------------------------------
Image/pixel space is right-down with depth into the screen. The ``.cgb`` world
is **Y-up, right-handed** (see ``docs/cgb-format.md``). We map::

    image +x  (right)        -> world +x   (right)
    image +y  (down)         -> world -y   (so world +y is up)
    camera +z (into scene)   -> world -z   (camera looks down -z, OpenGL style)

Concretely, after pinhole unprojection into camera space
``Xc = (u - cx)/fx * d``, ``Yc = (v - cy)/fy * d``, ``Zc = d`` (d = depth, +into
scene), we emit world points ``(Xc, -Yc, -Zc)``. This yields an upright object
(its top in the image ends up at larger world +y) facing roughly toward +z,
consistent with the baker/viewer's Y-up frame.

Scale ambiguity
---------------
Monocular depth is scale-ambiguous and (for relative models) only ordinal. We
therefore **normalise** every reconstruction: the per-image cloud is recentred
and uniformly scaled so the whole object fits inside a ``target_size`` (default
~1.5 m) bounding box. Absolute scale is meaningless from one photo — this just
keeps the blockout at a sane, human-ish metric size.

Single-view caveat
-------------------
A single image only reveals **front-facing visible surfaces** — the back of every
object is missing, so raw clouds are shells, not solids. :mod:`recognition.fit`
compensates with symmetry/thickness heuristics so fitted boxes are not
paper-thin. Nothing here invents the hidden geometry.

Model licences
--------------
* **SAM** (segmentation): Apache-2.0.
* **Depth Anything V2**: the *small* and *base* checkpoints are commonly released
  under permissive terms, but **the large variant and some releases carry more
  restrictive licences** (e.g. CC-BY-NC). Confirm the licence of the specific
  checkpoint you download before any commercial use.
* **MiDaS**: MIT (a safe permissive fallback).

Checkpoint guidance
-------------------
Depth Anything V2 is usually loaded one of two ways (we try both, in order):

1. ``transformers`` pipeline — e.g. model id
   ``"depth-anything/Depth-Anything-V2-Small-hf"`` (also ``...-Base-hf``,
   ``...-Large-hf``). Pass that id (or a local snapshot dir) as ``checkpoint``.
2. The upstream repo's ``DepthAnythingV2`` class with a local ``.pth`` checkpoint
   (``depth_anything_v2_{vits,vitb,vitl}.pth``) — set ``checkpoint`` to that path
   and ``encoder`` to match.

If neither Depth Anything path is available, the MiDaS fallback loads via
``torch.hub`` (``intel-isl/MiDaS``, ``DPT_Hybrid`` by default).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .segment import _INSTALL_HINT


# --------------------------------------------------------------------------- #
# Depth estimation
# --------------------------------------------------------------------------- #
class DepthEstimator:
    """Monocular depth: Depth Anything V2 (preferred) → MiDaS fallback.

    Parameters
    ----------
    checkpoint:
        Either a HuggingFace model id / local snapshot dir for the
        ``transformers`` path, or a local ``.pth`` for the upstream-repo path.
        ``None`` uses a sensible default per backend.
    backend:
        ``"auto"`` (try Depth Anything V2, then MiDaS), ``"depth_anything_v2"``,
        or ``"midas"``.
    encoder:
        Depth Anything V2 encoder size for the upstream-repo path
        (``"vits"`` / ``"vitb"`` / ``"vitl"``). Ignored by the transformers path.
    device:
        Torch device string; ``None`` auto-selects CUDA/MPS/CPU.

    The model loads lazily on first :meth:`estimate`.
    """

    _DA_V2_DEFAULT_HF = "depth-anything/Depth-Anything-V2-Small-hf"
    _MIDAS_DEFAULT = "DPT_Hybrid"

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        *,
        backend: str = "auto",
        encoder: str = "vits",
        device: Optional[str] = None,
    ) -> None:
        if backend not in ("auto", "depth_anything_v2", "midas"):
            raise ValueError(
                "backend must be 'auto', 'depth_anything_v2', or 'midas', "
                f"got {backend!r}"
            )
        self.checkpoint = checkpoint
        self.backend = backend
        self.encoder = encoder
        self.device = device

        self._infer = None  # callable(image_rgb) -> (H, W) float32 depth
        self._active_backend: Optional[str] = None

    # ------------------------------------------------------------------ #
    def _pick_device(self) -> str:
        if self.device is not None:
            return self.device
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(_INSTALL_HINT) from exc
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # ------------------------------------------------------------------ #
    def load(self) -> "DepthEstimator":
        """Eagerly resolve and load a depth backend. Idempotent; returns self."""
        if self._infer is not None:
            return self

        order: tuple[str, ...]
        if self.backend == "auto":
            order = ("depth_anything_v2", "midas")
        else:
            order = (self.backend,)

        errors: list[str] = []
        for name in order:
            try:
                if name == "depth_anything_v2":
                    self._infer = self._load_depth_anything_v2()
                else:
                    self._infer = self._load_midas()
                self._active_backend = name
                return self
            except Exception as exc:  # noqa: BLE001 - aggregate and report all
                errors.append(f"  [{name}] {exc}")

        raise RuntimeError(
            "Could not load any depth backend.\n"
            + "\n".join(errors)
            + "\n\n"
            + _INSTALL_HINT
        )

    # ------------------------------------------------------------------ #
    def _load_depth_anything_v2(self):
        """Return an inference callable for Depth Anything V2.

        Tries the ``transformers`` pipeline first (most robust, downloads
        weights automatically), then the upstream repo's ``DepthAnythingV2``
        class with a local ``.pth``.
        """
        # --- Path 1: HuggingFace transformers pipeline -------------------- #
        try:
            from transformers import pipeline  # type: ignore

            model_id = self.checkpoint or self._DA_V2_DEFAULT_HF
            device = self._pick_device()
            # transformers maps "cuda"/"mps"/"cpu" via device= int/str.
            pipe = pipeline(task="depth-estimation", model=model_id, device=device)

            def _infer(image_rgb: np.ndarray) -> np.ndarray:
                from PIL import Image  # type: ignore

                pil = Image.fromarray(np.asarray(image_rgb).astype(np.uint8))
                out = pipe(pil)
                depth = np.asarray(out["depth"], dtype=np.float32)
                return depth

            return _infer
        except ImportError:
            pass  # transformers not installed; try the upstream-repo path

        # --- Path 2: upstream repo class + local .pth --------------------- #
        try:
            import torch
            from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Depth Anything V2 unavailable: install `transformers` for the "
                "pipeline path, or vendor the `depth_anything_v2` package and "
                "pass a local .pth checkpoint. See docs/recognition.md."
            ) from exc

        if self.checkpoint is None or not Path(self.checkpoint).is_file():
            raise RuntimeError(
                "Depth Anything V2 upstream path needs a local .pth checkpoint "
                "(depth_anything_v2_vits/vitb/vitl.pth); none found at "
                f"{self.checkpoint!r}."
            )

        cfgs = {
            "vits": dict(encoder="vits", features=64, out_channels=[48, 96, 192, 384]),
            "vitb": dict(encoder="vitb", features=128, out_channels=[96, 192, 384, 768]),
            "vitl": dict(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024]),
        }
        if self.encoder not in cfgs:
            raise ValueError(f"encoder must be one of {list(cfgs)}, got {self.encoder!r}")

        device = self._pick_device()
        model = DepthAnythingV2(**cfgs[self.encoder])
        model.load_state_dict(torch.load(self.checkpoint, map_location="cpu"))
        model = model.to(device).eval()

        def _infer(image_rgb: np.ndarray) -> np.ndarray:
            # DepthAnythingV2.infer_image takes a BGR uint8 array (OpenCV order).
            import cv2  # type: ignore

            bgr = cv2.cvtColor(np.asarray(image_rgb).astype(np.uint8), cv2.COLOR_RGB2BGR)
            depth = model.infer_image(bgr)  # (H, W) float, larger = nearer
            return np.asarray(depth, dtype=np.float32)

        return _infer

    # ------------------------------------------------------------------ #
    def _load_midas(self):
        """Return an inference callable for the MiDaS fallback (MIT)."""
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(_INSTALL_HINT) from exc

        model_name = self.checkpoint or self._MIDAS_DEFAULT
        device = self._pick_device()

        midas = torch.hub.load("intel-isl/MiDaS", model_name)
        midas.to(device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if "DPT" in model_name:
            transform = transforms.dpt_transform
        else:
            transform = transforms.small_transform

        def _infer(image_rgb: np.ndarray) -> np.ndarray:
            img = np.asarray(image_rgb).astype(np.uint8)
            sample = transform(img).to(device)
            with torch.no_grad():
                pred = midas(sample)
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            # MiDaS returns inverse-depth: larger = nearer (same sense we want).
            return pred.detach().cpu().numpy().astype(np.float32)

        return _infer

    # ------------------------------------------------------------------ #
    def estimate(self, image_rgb: np.ndarray) -> np.ndarray:
        """Estimate a ``(H, W)`` float32 depth map from an RGB image.

        Output convention: **larger value = nearer the camera**. Both Depth
        Anything V2 and MiDaS natively emit inverse-depth-like maps with that
        sense, so we keep it. Values are *relative* (not metric); the absolute
        scale is fixed later in :func:`backproject` / normalisation.
        """
        image_rgb = np.asarray(image_rgb)
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(
                f"image_rgb must be (H, W, 3) RGB, got shape {image_rgb.shape}"
            )
        self.load()
        assert self._infer is not None
        depth = np.asarray(self._infer(image_rgb), dtype=np.float32)
        if depth.shape != image_rgb.shape[:2]:
            raise RuntimeError(
                f"depth map shape {depth.shape} does not match image "
                f"{image_rgb.shape[:2]}"
            )
        return depth


# --------------------------------------------------------------------------- #
# Camera model
# --------------------------------------------------------------------------- #
def default_intrinsics(width: int, height: int, fov_deg: float = 55.0) -> np.ndarray:
    """Synthesise a plausible pinhole intrinsics matrix for a single photo.

    A lone image carries no real intrinsics, so we assume a typical horizontal
    field of view (default 55°, ~a normal lens) and a principal point at the
    image centre. The focal length in pixels is derived from the image width::

        fx = fy = (width / 2) / tan(fov / 2)

    Returns the standard ``3x3`` matrix::

        [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size: {width}x{height}")
    fov = np.deg2rad(float(fov_deg))
    fx = (width / 2.0) / np.tan(fov / 2.0)
    fy = fx  # square pixels
    cx = width / 2.0
    cy = height / 2.0
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def backproject(
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    *,
    target_size: float = 1.5,
    depth_gamma: float = 1.0,
) -> np.ndarray:
    """Unproject masked pixels into an ``Nx3`` world-frame point cloud.

    Parameters
    ----------
    depth:
        ``(H, W)`` float32 depth map, **larger = nearer** (the convention from
        :meth:`DepthEstimator.estimate`).
    mask:
        ``(H, W)`` boolean region to unproject.
    intrinsics:
        ``3x3`` pinhole matrix from :func:`default_intrinsics`.
    target_size:
        The reconstructed cloud is uniformly scaled so its largest bounding-box
        extent equals this many metres (monocular scale is arbitrary).
    depth_gamma:
        Optional exponent applied to normalised depth before unprojection to
        gently tune relief; ``1.0`` leaves it unchanged.

    Returns
    -------
    ``(N, 3)`` float64 points in the ``.cgb`` world frame (Y-up, right-handed,
    metres), recentred on the cloud centroid.

    Notes
    -----
    The depth model gives *inverse-depth-like* values (larger = nearer). We
    convert to a forward "distance into the scene" so geometry comes out the
    right way round: nearer pixels (large depth value) get a *smaller* Z. We map
    the per-mask depth range into a unit ``[0, 1]`` interval, then use
    ``z_cam = 1 - d_norm`` as relative distance. Absolute scale is fixed by the
    ``target_size`` normalisation at the end, so the only thing that matters is
    relative relief across the cloud.
    """
    depth = np.asarray(depth, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if depth.shape != mask.shape:
        raise ValueError(
            f"depth {depth.shape} and mask {mask.shape} must have the same shape"
        )

    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    d_raw = depth[ys, xs]

    # Normalise the *inverse-depth* values within this segment to [0, 1] and
    # flip to a forward distance (nearer pixels => smaller z into the scene).
    d_min, d_max = float(d_raw.min()), float(d_raw.max())
    if d_max - d_min < 1e-9:
        # Flat depth (e.g. a billboard) — give it a hair of relief so it is not
        # a degenerate zero-thickness plane.
        d_norm = np.zeros_like(d_raw)
    else:
        d_norm = (d_raw - d_min) / (d_max - d_min)
    if depth_gamma != 1.0:
        d_norm = np.power(d_norm, depth_gamma)
    z_cam = 1.0 - d_norm  # forward distance into the scene, in [0, 1]

    # Use a unit nominal camera distance so the perspective spread is sensible;
    # the final metric scale is set by target_size normalisation regardless.
    z_cam = z_cam + 1.0  # shift off zero so pixels are in front of the camera

    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])

    # Pinhole unprojection into camera space (camera +z into the scene).
    x_cam = (xs - cx) / fx * z_cam
    y_cam = (ys - cy) / fy * z_cam

    # Camera -> world axis mapping (see module docstring):
    #   image +x -> world +x ; image +y (down) -> world -y ; cam +z -> world -z
    world = np.stack([x_cam, -y_cam, -z_cam], axis=1)  # (N, 3)

    # Recenter on the centroid and normalise to a sane metric size. Monocular
    # depth is scale-ambiguous, so we just fit the object into target_size.
    world -= world.mean(axis=0, keepdims=True)
    extent = world.max(axis=0) - world.min(axis=0)
    longest = float(extent.max())
    if longest > 1e-9:
        world *= target_size / longest
    return world


# --------------------------------------------------------------------------- #
# Debug helper
# --------------------------------------------------------------------------- #
def save_point_cloud_ply(
    points: np.ndarray,
    path: str,
    *,
    colors: Optional[np.ndarray] = None,
) -> str:
    """Write an ``Nx3`` cloud to a ``.ply`` file for debugging (via open3d).

    ``colors`` is an optional ``Nx3`` array in ``0..1`` (RGB). open3d is lazily
    imported so this module still imports without it.
    """
    try:
        import open3d as o3d  # type: ignore
    except ImportError as exc:  # pragma: no cover - env dependent
        raise RuntimeError(_INSTALL_HINT) from exc

    points = np.asarray(points, dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        colors = np.asarray(colors, dtype=np.float64)
        if colors.shape == points.shape:
            pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out), pcd)
    return str(out)
