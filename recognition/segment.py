"""Phase 4 — image segmentation with Segment Anything (SAM).

This module wraps Meta's `Segment Anything <https://github.com/facebookresearch/segment-anything>`_
(Apache-2.0) behind a small :class:`Segmenter` facade. Given an RGB image it
produces a handful of salient object regions as boolean masks; downstream stages
(:mod:`recognition.depth`, :mod:`recognition.fit`) turn each region into a 3D
point cloud and finally into a ``.cgb`` primitive.

We use ``SamAutomaticMaskGenerator`` (no prompts) so the pipeline is fully
automatic on an arbitrary single image. SAM tends to over-segment, so we filter
the raw proposals down to the most useful blockout-sized regions: tiny specks
and the full-frame background mask are dropped, and only the ``max_masks`` most
salient regions are kept.

Heavy imports (``torch``, ``segment_anything``) are performed lazily inside the
methods that need them so that merely importing this module never fails when the
recognition extras are absent. ``numpy`` is the only hard dependency, matching
the rest of the recognition package.

Checkpoints
-----------
SAM weights are downloaded separately (they are not on PyPI). Pick a backbone
and grab the matching checkpoint:

==========  =========================  ============================
model_type  checkpoint file            notes
==========  =========================  ============================
``vit_h``   ``sam_vit_h_4b8939.pth``   best quality, ~2.6 GB, slow
``vit_l``   ``sam_vit_l_0b3195.pth``   balanced, ~1.2 GB
``vit_b``   ``sam_vit_b_01ec64.pth``   fastest, ~375 MB
==========  =========================  ============================

Links: https://github.com/facebookresearch/segment-anything#model-checkpoints
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Install hint reused by every "missing heavy dependency" error in the package.
_INSTALL_HINT = (
    "The recognition pipeline needs the heavy ML extras. Install them with\n"
    "    pip install -r requirements-recognition.txt\n"
    "and download a SAM checkpoint (e.g. sam_vit_h_4b8939.pth) from\n"
    "    https://github.com/facebookresearch/segment-anything#model-checkpoints"
)

# Valid SAM backbones and the checkpoint filename each one expects.
SAM_MODEL_TYPES = ("vit_h", "vit_l", "vit_b")


@dataclass
class Mask:
    """One segmented region.

    Attributes
    ----------
    mask:
        Boolean array of shape ``(H, W)`` — ``True`` where the region covers the
        image. This is the field the rest of the pipeline consumes.
    area:
        Pixel count of the region (``int(mask.sum())``).
    bbox:
        Axis-aligned bounding box in XYWH pixel coordinates
        ``(x_min, y_min, width, height)`` (SAM's native convention).
    predicted_iou:
        SAM's self-reported quality score for the mask (``0..1``); useful for
        ranking. ``-1`` when unknown.
    point_coords:
        Optional sampling point(s) SAM used to seed the mask, kept for debugging.
    """

    mask: np.ndarray
    area: int
    bbox: tuple[int, int, int, int]
    predicted_iou: float = -1.0
    point_coords: list = field(default_factory=list)

    @property
    def centroid(self) -> tuple[float, float]:
        """(x, y) pixel centroid of the region (handy for ordering/labels)."""
        ys, xs = np.nonzero(self.mask)
        if xs.size == 0:
            return (0.0, 0.0)
        return (float(xs.mean()), float(ys.mean()))


class Segmenter:
    """Automatic SAM segmenter.

    Parameters
    ----------
    checkpoint:
        Path to a downloaded SAM ``.pth`` checkpoint.
    model_type:
        SAM backbone, one of ``vit_h`` / ``vit_l`` / ``vit_b``. Must match the
        checkpoint.
    device:
        Torch device string (``"cuda"``, ``"mps"``, ``"cpu"``). ``None`` picks
        CUDA when available, otherwise CPU.
    points_per_side:
        Density of the automatic prompt grid. Lower = fewer, coarser masks,
        which suits blockout. Forwarded to ``SamAutomaticMaskGenerator``.
    pred_iou_thresh / stability_score_thresh:
        SAM's internal quality gates; the defaults match SAM's own defaults.

    The expensive model load happens lazily on first :meth:`segment` (or via an
    explicit :meth:`load`) so constructing a ``Segmenter`` is cheap and never
    touches torch.
    """

    def __init__(
        self,
        checkpoint: str,
        *,
        model_type: str = "vit_h",
        device: Optional[str] = None,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.92,
    ) -> None:
        if model_type not in SAM_MODEL_TYPES:
            raise ValueError(
                f"model_type must be one of {SAM_MODEL_TYPES!r}, got {model_type!r}"
            )
        self.checkpoint = str(checkpoint)
        self.model_type = model_type
        self.device = device
        self.points_per_side = int(points_per_side)
        self.pred_iou_thresh = float(pred_iou_thresh)
        self.stability_score_thresh = float(stability_score_thresh)

        # Populated lazily by load().
        self._generator = None
        self._resolved_device: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Model loading (lazy, heavy)
    # ------------------------------------------------------------------ #
    def _pick_device(self) -> str:
        """Resolve the torch device for SAM.

        SAM's ``SamAutomaticMaskGenerator`` builds ``float64`` point tensors,
        which Apple's MPS backend cannot handle ("Cannot convert a MPS Tensor to
        float64"). So we never run SAM on MPS: an explicit ``device="mps"`` (or
        an auto-pick that lands on MPS) falls back to CPU with a warning. The
        depth model is unaffected and may still use MPS.
        """
        requested = self.device
        if requested is None:
            try:
                import torch
            except ImportError as exc:  # pragma: no cover - env dependent
                raise RuntimeError(_INSTALL_HINT) from exc
            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                requested = "mps"
            else:
                return "cpu"

        if requested == "mps":
            warnings.warn(
                "SAM does not support the MPS backend (float64 limitation); "
                "running SAM on CPU instead.",
                RuntimeWarning,
                stacklevel=2,
            )
            return "cpu"
        return requested

    def load(self) -> "Segmenter":
        """Eagerly load the SAM model and build the mask generator.

        Idempotent; returns ``self`` so it can be chained. Raises a clear
        :class:`RuntimeError` if the heavy deps or checkpoint are missing.
        """
        if self._generator is not None:
            return self

        try:
            import torch  # noqa: F401  (imported for device handling/side effects)
            from segment_anything import (
                SamAutomaticMaskGenerator,
                sam_model_registry,
            )
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(_INSTALL_HINT) from exc

        ckpt = Path(self.checkpoint)
        if not ckpt.is_file():
            raise RuntimeError(
                f"SAM checkpoint not found: {ckpt}\n"
                f"Download the '{self.model_type}' checkpoint from\n"
                "    https://github.com/facebookresearch/segment-anything#model-checkpoints"
            )

        self._resolved_device = self._pick_device()
        sam = sam_model_registry[self.model_type](checkpoint=str(ckpt))
        sam.to(device=self._resolved_device)

        # A coarser grid than SAM's default keeps proposals blockout-sized and
        # speeds up CPU inference; min_mask_region_area prunes specks early.
        self._generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=self.points_per_side,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
            min_mask_region_area=64,
        )
        return self

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def segment(
        self,
        image_rgb: np.ndarray,
        *,
        max_masks: int = 12,
        min_area_frac: float = 0.002,
        max_area_frac: float = 0.95,
    ) -> list[Mask]:
        """Segment an RGB image into a filtered list of :class:`Mask`.

        Parameters
        ----------
        image_rgb:
            ``(H, W, 3)`` ``uint8`` RGB image (note: **RGB**, not OpenCV's BGR).
        max_masks:
            Keep at most this many regions, ranked by salience
            (``predicted_iou * area``). Blockout wants a handful of big parts.
        min_area_frac / max_area_frac:
            Drop masks smaller than ``min_area_frac`` of the frame (specks) or
            larger than ``max_area_frac`` (the full-image background plate).

        Returns the surviving masks sorted from most to least salient.
        """
        image_rgb = np.asarray(image_rgb)
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(
                f"image_rgb must be (H, W, 3) RGB, got shape {image_rgb.shape}"
            )

        self.load()
        assert self._generator is not None  # for type-checkers

        raw = self._generator.generate(image_rgb)
        return self._filter(
            raw,
            image_shape=image_rgb.shape[:2],
            max_masks=max_masks,
            min_area_frac=min_area_frac,
            max_area_frac=max_area_frac,
        )

    # ------------------------------------------------------------------ #
    # Proposal filtering
    # ------------------------------------------------------------------ #
    @staticmethod
    def _filter(
        raw: list[dict],
        *,
        image_shape: tuple[int, int],
        max_masks: int,
        min_area_frac: float,
        max_area_frac: float,
    ) -> list[Mask]:
        """Convert SAM's raw dict proposals into filtered, ranked :class:`Mask`."""
        h, w = image_shape
        frame_area = float(h * w)
        min_area = min_area_frac * frame_area
        max_area = max_area_frac * frame_area

        kept: list[Mask] = []
        for item in raw:
            seg = np.asarray(item["segmentation"], dtype=bool)
            area = float(item.get("area", int(seg.sum())))

            # Drop specks and the near-full-frame background plate.
            if area < min_area or area > max_area:
                continue

            bbox = item.get("bbox", _bbox_from_mask(seg))
            kept.append(
                Mask(
                    mask=seg,
                    area=int(area),
                    bbox=tuple(int(v) for v in bbox),  # XYWH
                    predicted_iou=float(item.get("predicted_iou", -1.0)),
                    point_coords=list(item.get("point_coords", [])),
                )
            )

        # Rank by salience: confident *and* sizeable regions first.
        def salience(m: Mask) -> float:
            iou = m.predicted_iou if m.predicted_iou >= 0 else 1.0
            return iou * m.area

        kept.sort(key=salience, reverse=True)
        return kept[:max_masks]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    """XYWH pixel bbox of a boolean mask (fallback when SAM omits ``bbox``)."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return (0, 0, 0, 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def load_image_rgb(path: str) -> np.ndarray:
    """Load an image from disk as an ``(H, W, 3)`` ``uint8`` **RGB** array.

    Tries OpenCV first (BGR→RGB), falls back to Pillow. Both are listed in
    ``requirements-recognition.txt``; the import is lazy so this module still
    imports without them.
    """
    try:
        import cv2  # type: ignore

        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Could not read image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except ImportError:
        pass

    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:  # pragma: no cover - env dependent
        raise RuntimeError(_INSTALL_HINT) from exc

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))
