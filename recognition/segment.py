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
        points_per_side: int = 16,
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
        min_area_frac: float = 0.01,
        max_area_frac: float = 0.95,
        border_bg_frac: float = 0.55,
        nms_overlap: float = 0.5,
        part_area_frac: float = 0.02,
    ) -> list[Mask]:
        """Segment an RGB image into a filtered, **part-level** list of :class:`Mask`.

        SAM over-segments: it emits the background plate, the whole-object mask,
        near-duplicates, and sub-fragments of a single part. For a blockout we
        want a handful of *mid-level parts* (seat, back, legs — not the whole
        chair, not wood-grain specks), so raw proposals are post-processed:

        1. **Specks / full-frame** dropped by area (``min_area_frac`` /
           ``max_area_frac``).
        2. **Background plate** dropped when a mask hugs the frame border
           (border coverage ``> border_bg_frac``) — see :func:`_border_touch_frac`.
        3. **Redundant parents** dropped: a mask decomposed by ``>=2`` *sizeable*
           parts (each ``> part_area_frac``) covering most of it is removed in
           favour of those parts (drops the whole-object mask, keeps the parts).
        4. **Overlap NMS**: process by salience and skip a mask that overlaps an
           already-kept part by more than ``nms_overlap`` (``|A∩B|/min``) — this
           discards nested sub-fragments and duplicates while keeping
           side-by-side parts.

        Parameters
        ----------
        image_rgb:
            ``(H, W, 3)`` ``uint8`` RGB image (note: **RGB**, not OpenCV's BGR).
        max_masks:
            Keep at most this many parts, ranked by salience.
        min_area_frac / max_area_frac:
            Area gates as a fraction of the frame.
        border_bg_frac:
            Drop a mask covering more than this fraction of the image border.
        nms_overlap:
            Skip a mask whose ``|A∩B|/min(|A|,|B|)`` with a kept part exceeds
            this (nested fragment or duplicate of an already-kept part).
        part_area_frac:
            Minimum area (frame fraction) for a mask to count as a "part" when
            deciding whether a larger mask is a redundant parent.

        Returns the surviving part masks sorted from most to least salient.
        """
        image_rgb = np.asarray(image_rgb)
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(
                f"image_rgb must be (H, W, 3) RGB, got shape {image_rgb.shape}"
            )

        self.load()
        assert self._generator is not None  # for type-checkers

        raw = self._generator.generate(image_rgb)
        return self._select_parts(
            raw,
            image_shape=image_rgb.shape[:2],
            max_masks=max_masks,
            min_area_frac=min_area_frac,
            max_area_frac=max_area_frac,
            border_bg_frac=border_bg_frac,
            nms_overlap=nms_overlap,
            part_area_frac=part_area_frac,
        )

    # ------------------------------------------------------------------ #
    # Proposal filtering -> part selection
    # ------------------------------------------------------------------ #
    @staticmethod
    def _select_parts(
        raw: list[dict],
        *,
        image_shape: tuple[int, int],
        max_masks: int,
        min_area_frac: float,
        max_area_frac: float,
        border_bg_frac: float,
        nms_overlap: float,
        part_area_frac: float,
    ) -> list[Mask]:
        """Turn SAM's raw proposals into filtered, merged, ranked part masks."""
        h, w = image_shape
        frame_area = float(h * w)
        min_area = min_area_frac * frame_area
        max_area = max_area_frac * frame_area
        part_area = part_area_frac * frame_area

        # --- 1+2. area gate + background-plate removal ------------------ #
        cand: list[Mask] = []
        for item in raw:
            seg = np.asarray(item["segmentation"], dtype=bool)
            area = float(item.get("area", int(seg.sum())))
            if area < min_area or area > max_area:
                continue
            # Background: a sizeable mask that wraps the frame border.
            if area > 0.30 * frame_area and _border_touch_frac(seg) > border_bg_frac:
                continue
            bbox = item.get("bbox", _bbox_from_mask(seg))
            cand.append(
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

        cand.sort(key=salience, reverse=True)

        # --- 3. drop redundant "parent" masks (prefer parts) ----------- #
        # SAM emits a whole-object mask *and* its parts. For a blockout we want
        # the parts, so drop a mask that is mostly decomposed by >=2 *sizeable*
        # parts nested inside it. Requiring sizeable children (not specks) keeps
        # mid-level masks like a seat from being dissolved by texture fragments.
        def child_frac_in(parent: Mask, child: Mask) -> float:
            inter = int(np.logical_and(parent.mask, child.mask).sum())
            return inter / float(child.area) if child.area else 0.0

        survivors: list[Mask] = []
        for p in cand:
            children = [
                c for c in cand
                if c is not p and part_area <= c.area < 0.9 * p.area
                and child_frac_in(p, c) > 0.80
            ]
            covered = 0
            if children:
                union = np.zeros_like(p.mask)
                for c in children:
                    union |= c.mask
                covered = int(np.logical_and(union, p.mask).sum())
            if len(children) >= 2 and covered > 0.60 * p.area:
                continue  # redundant parent — its parts carry the detail
            survivors.append(p)

        # --- 4. overlap NMS: keep salient parts, skip nested fragments -- #
        kept: list[Mask] = []
        for m in survivors:
            if any(_intersection_over_min(m.mask, k.mask) > nms_overlap for k in kept):
                continue  # nested sub-fragment or duplicate of a kept part
            kept.append(m)

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


def _border_touch_frac(mask: np.ndarray) -> float:
    """Fraction of the image's border pixels covered by ``mask``.

    A background plate hugs the frame, so it covers most of the perimeter;
    a centred object touches little of it. Used to drop the background mask
    that ``SamAutomaticMaskGenerator`` often emits.
    """
    if mask.size == 0:
        return 0.0
    border = np.concatenate([
        mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1],
    ])
    return float(border.mean())


def _intersection_over_min(a: np.ndarray, b: np.ndarray) -> float:
    """``|A∩B| / min(|A|, |B|)`` — high when one mask nests inside the other.

    Better than IoU for catching a small sub-fragment of a larger part (IoU
    stays low when the parts differ a lot in size, but this ratio approaches 1).
    """
    aa, bb = int(a.sum()), int(b.sum())
    if aa == 0 or bb == 0:
        return 0.0
    inter = int(np.logical_and(a, b).sum())
    return inter / float(min(aa, bb))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two boolean masks."""
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / float(union) if union else 0.0


def _solidity(mask: np.ndarray) -> float:
    """``area / convex-hull area`` of a mask (1.0 = convex). ``-1`` if cv2 absent.

    A single solid (e.g. the three shaded faces of one box) has a near-convex
    silhouette; distinct parts glued together (seat + leg) leave concavities, so
    their union's solidity drops. Used to merge faces of one object without
    fusing genuinely separate parts.
    """
    try:
        import cv2  # type: ignore
    except ImportError:  # pragma: no cover - env dependent
        return -1.0
    m = np.ascontiguousarray(mask.astype(np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    area = float(sum(cv2.contourArea(c) for c in cnts))
    hull = cv2.convexHull(np.concatenate(cnts, axis=0))
    hull_area = float(cv2.contourArea(hull))
    return area / hull_area if hull_area > 1e-6 else 0.0


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
