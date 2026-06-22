"""Multi-view (2x2 sheet) reconstruction — the optional "precision mode".

A single concept image gives only a front shell, so the back/thickness are
guessed. When the user also supplies a **2x2 multi-view sheet** (one image,
four cells at known angles) we instead *measure* the shape by space-carving a
voxel occupancy from the silhouettes, then fit the same parametric primitives to
it. Output is still a normal ``.cgb`` so the viewer/baker/Blender are unchanged.

Fixed sheet layout (agreed convention)::

    +---------+---------+
    | front   | side    |     top-left  = front  (camera looks along -Z)
    +---------+---------+     top-right = side   (right side, looks along -X)
    | back    | top     |     bot-left  = back   (looks along +Z)
    +---------+---------+     bot-right = top    (looks down -Y)

A cell may be **blank** (a view the user does not have): blank cells are detected
and simply skipped — they add no carving constraint, so accuracy scales with how
many faces are filled (never worse for a missing one). Pure numpy + Pillow/OpenCV.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Cell (row, col) -> view name, per the fixed layout above.
SHEET_LAYOUT = {(0, 0): "front", (0, 1): "side", (1, 0): "back", (1, 1): "top"}


@dataclass
class ViewCell:
    """One cropped view: its RGB image, foreground silhouette, and view name."""

    name: str                 # front | side | back | top
    rgb: np.ndarray           # (h, w, 3) uint8
    silhouette: np.ndarray    # (h, w) bool — True = object
    blank: bool               # True when no object was found (skip in carving)


def _bg_color(rgb: np.ndarray, border: int = 6) -> np.ndarray:
    """Estimate the flat background colour from the cell's border pixels."""
    h, w = rgb.shape[:2]
    edges = np.concatenate([
        rgb[:border].reshape(-1, 3), rgb[-border:].reshape(-1, 3),
        rgb[:, :border].reshape(-1, 3), rgb[:, -border:].reshape(-1, 3),
    ])
    return np.median(edges, axis=0)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes by flood-filling background from the border.

    Any background pixel not reachable from the image border is enclosed by the
    object, so it belongs to the object. Robust to low-contrast interiors (a
    steel blade whose centre matches the plate) as long as the outline is found.
    """
    try:
        import cv2  # type: ignore
    except ImportError:  # pragma: no cover
        return mask
    m = mask.astype(np.uint8)
    # Pad with a 0 ring so flood-fill reaches background connected to ANY border
    # (a thin object spanning the frame splits the background into regions; the
    # ring reconnects them so only truly enclosed holes remain unreached).
    padded = np.pad(m, 1, mode="constant", constant_values=0)
    ff = padded.copy()
    cv2.floodFill(ff, None, (0, 0), 1)        # fill all border-connected bg
    holes = (ff[1:-1, 1:-1] == 0)             # interior unreached = real holes
    return (m.astype(bool) | holes)


def _silhouette_bg(rgb: np.ndarray, *, tol: int = 20) -> np.ndarray:
    """Silhouette by subtracting the flat background plate.

    Reliable for clean concept-art cells (object on a near-uniform plate), and —
    unlike unioning SAM masks — it cannot accidentally swallow a stray background
    band. A morphological close bridges thin low-contrast gaps (a steel blade
    that nearly matches the plate), then enclosed holes are filled.
    """
    bg = _bg_color(rgb)
    dist = np.abs(rgb.astype(np.int16) - bg).sum(axis=2)
    mask = dist > tol
    try:
        import cv2  # type: ignore
        m = mask.astype(np.uint8)
        # drop tiny specks first, then bridge gaps within the object
        ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ko)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kc)
        # keep components above a small area (the object may be a few pieces)
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(m)
        if n > 1:
            big = stats[1:, cv2.CC_STAT_AREA].max()
            keep = {i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 0.05 * big}
            m = np.isin(lbl, list(keep)).astype(np.uint8)
        mask = m.astype(bool)
    except ImportError:  # pragma: no cover
        pass
    return _fill_holes(mask)


def _silhouette_sam(rgb: np.ndarray, segmenter, *, border_thresh: float = 0.30) -> np.ndarray:
    """Robust silhouette = union of SAM masks that are NOT the background plate.

    Uses the raw automatic masks (not the part-extraction pipeline) and drops any
    mask hugging the cell border (the background), so a low-contrast object on a
    grey plate (a steel blade) is captured cleanly. Falls back to background
    subtraction if SAM is unavailable or yields nothing.
    """
    from .segment import _border_touch_frac
    try:
        segmenter.load()
        raw = segmenter._generator.generate(np.ascontiguousarray(rgb))
    except Exception:
        return _silhouette_bg(rgb)
    if not raw:
        return _silhouette_bg(rgb)
    h, w = rgb.shape[:2]
    frame = float(h * w)
    sil = np.zeros((h, w), dtype=bool)
    for item in raw:
        seg = np.asarray(item["segmentation"], dtype=bool)
        frac = seg.sum() / frame
        if frac > 0.95:
            continue  # full-frame
        # Background = a mask that owns an image corner (concept art centres the
        # object with a margin, so only the plate reaches the corners). This is
        # robust even when SAM splits the plate into several sub-50% pieces.
        if seg[0, 0] or seg[0, -1] or seg[-1, 0] or seg[-1, -1]:
            continue
        # Also drop anything still hugging the border heavily (safety net).
        if frac > 0.50 and _border_touch_frac(seg) > border_thresh:
            continue
        sil |= seg
    if not sil.any():
        return _silhouette_bg(rgb)
    return _fill_holes(sil)


def split_sheet(
    image_rgb: np.ndarray,
    *,
    segmenter=None,
    min_fg_frac: float = 0.005,
) -> list[ViewCell]:
    """Split a 2x2 sheet into four :class:`ViewCell` (front/side/back/top).

    If ``segmenter`` (a :class:`recognition.segment.Segmenter`) is given, each
    cell's silhouette is the union of its SAM object masks (robust for
    low-contrast art); otherwise a background-subtraction silhouette is used. A
    cell whose silhouette covers less than ``min_fg_frac`` of the cell is marked
    ``blank`` (an absent view) and excluded from carving.
    """
    img = np.asarray(image_rgb)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) RGB, got {img.shape}")
    H, W = img.shape[:2]
    hh, hw = H // 2, W // 2
    cells: list[ViewCell] = []
    for (r, c), name in SHEET_LAYOUT.items():
        sub = np.ascontiguousarray(img[r * hh:(r + 1) * hh, c * hw:(c + 1) * hw])
        sil = _silhouette_sam(sub, segmenter) if segmenter is not None else _silhouette_bg(sub)
        blank = sil.mean() < min_fg_frac
        cells.append(ViewCell(name=name, rgb=sub, silhouette=sil, blank=blank))
    return cells


# --------------------------------------------------------------------------- #
# Space carving — silhouettes -> voxel occupancy
# --------------------------------------------------------------------------- #
# Each view maps a normalised world point (x, y, z) in [0,1]^3 (X right, Y up,
# Z toward the front camera) to a normalised image coord (u right, v down).
# All upright views share v = 1 - y; they differ only in the horizontal axis.
def _project(name: str, x: np.ndarray, y: np.ndarray, z: np.ndarray):
    if name == "front":
        return x, 1.0 - y
    if name == "back":
        return 1.0 - x, 1.0 - y
    if name == "side":            # right side: image-right = world +Z
        return z, 1.0 - y
    if name == "top":             # looking down: image-down = world +Z (front)
        return x, z
    raise ValueError(f"unknown view {name!r}")


def carve_occupancy(cells: list[ViewCell], *, res: int = 64) -> np.ndarray:
    """Space-carve an ``(res, res, res)`` boolean occupancy from view silhouettes.

    A voxel is occupied iff it projects **inside the silhouette of every
    available (non-blank) view**. Blank views add no constraint (so accuracy
    scales with how many faces are filled). Indexed ``[ix, iy, iz]`` for world
    (x, y, z); empty grid if no usable view.
    """
    used = [c for c in cells if not c.blank]
    occ = np.ones((res, res, res), dtype=bool)
    if not used:
        return np.zeros_like(occ)

    # Normalised voxel-centre coordinates along each axis.
    coord = (np.arange(res) + 0.5) / res
    X, Y, Z = np.meshgrid(coord, coord, coord, indexing="ij")  # each (res,res,res)

    for c in used:
        u, v = _project(c.name, X, Y, Z)
        H, W = c.silhouette.shape
        ui = np.clip((u * W).astype(np.int32), 0, W - 1)
        vi = np.clip((v * H).astype(np.int32), 0, H - 1)
        inside = c.silhouette[vi, ui]
        occ &= inside     # intersection of all view cones (visual hull)
    return occ


def occupancy_to_world_points(occ: np.ndarray) -> np.ndarray:
    """Occupied voxel centres as an ``(N, 3)`` world cloud, centred in [-0.5,0.5]."""
    idx = np.argwhere(occ)
    if idx.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    res = occ.shape[0]
    return (idx + 0.5) / res - 0.5


# --------------------------------------------------------------------------- #
# Greedy axis-aligned box decomposition (voxel solid -> tiling cubes)
# --------------------------------------------------------------------------- #
def _grow_box(occ: np.ndarray, seed: tuple[int, int, int]) -> tuple:
    """Grow a maximal all-occupied axis-aligned box around ``seed``.

    Expands each of the 6 faces by one slab at a time while the slab stays fully
    occupied, cycling until no face can grow. Returns ``(lo, hi)`` inclusive
    index bounds.
    """
    sx, sy, sz = seed
    lo = [sx, sy, sz]
    hi = [sx, sy, sz]
    R = occ.shape[0]
    grew = True
    while grew:
        grew = False
        for axis in range(3):
            # try low face
            if lo[axis] > 0:
                lo[axis] -= 1
                if _slab_full(occ, lo, hi, axis, lo[axis]):
                    grew = True
                else:
                    lo[axis] += 1
            # try high face
            if hi[axis] < R - 1:
                hi[axis] += 1
                if _slab_full(occ, lo, hi, axis, hi[axis]):
                    grew = True
                else:
                    hi[axis] -= 1
    return tuple(lo), tuple(hi)


def _slab_full(occ, lo, hi, axis, plane) -> bool:
    """Is the box face-slab at ``plane`` (perpendicular to ``axis``) fully set?"""
    rng = [slice(lo[a], hi[a] + 1) for a in range(3)]
    rng[axis] = slice(plane, plane + 1)
    return bool(occ[tuple(rng)].all())


def occupancy_to_boxes(
    occ: np.ndarray,
    *,
    max_boxes: int = 16,
    min_cover: float = 0.95,
    min_box_frac: float = 0.0005,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cover an occupancy grid with a few maximal axis-aligned boxes.

    Greedy: seed at the *deepest* still-uncovered voxel (max distance-to-empty),
    grow a maximal occupied box there, mark it covered, repeat until ``min_cover``
    of the solid is explained or ``max_boxes`` is reached. Boxes may overlap
    (fine for a solid blockout) and **tile the carved shape with no gaps or
    floating parts** — far more faithful than fitting separate primitives.

    Returns ``[(centre, size)]`` in normalised world units (cube in [-0.5,0.5]).
    """
    try:
        from scipy import ndimage
        edt = ndimage.distance_transform_edt
    except Exception:  # pragma: no cover
        edt = None

    R = occ.shape[0]
    total = int(occ.sum())
    if total == 0:
        return []
    remaining = occ.copy()
    covered = 0
    boxes = []
    min_box_vox = max(1, int(min_box_frac * total))

    while remaining.any() and len(boxes) < max_boxes:
        if edt is not None:
            dt = edt(remaining)
            seed = np.unravel_index(int(np.argmax(dt)), dt.shape)
        else:
            seed = tuple(np.argwhere(remaining)[0])
        lo, hi = _grow_box(occ, seed)
        sel = (slice(lo[0], hi[0] + 1), slice(lo[1], hi[1] + 1), slice(lo[2], hi[2] + 1))
        vol = (hi[0] - lo[0] + 1) * (hi[1] - lo[1] + 1) * (hi[2] - lo[2] + 1)
        new = int(remaining[sel].sum())
        if vol < min_box_vox and new == 0:
            remaining[seed] = False
            continue
        remaining[sel] = False
        covered += new
        centre = (np.array(lo) + np.array(hi) + 1) / 2.0 / R - 0.5
        size = (np.array(hi) - np.array(lo) + 1) / R
        boxes.append((centre, size))
        if covered / total >= min_cover:
            break
    return boxes


# --------------------------------------------------------------------------- #
# Orchestration: 2x2 sheet -> .cgb (precision mode)
# --------------------------------------------------------------------------- #
def image_to_cgb_multiview(
    sheet_path: str,
    out_path: str,
    *,
    sam_checkpoint: str,
    device: Optional[str] = None,
    sam_model_type: str = "vit_h",
    res: int = 64,
    max_segments: int = 12,
    prior_weight: float = 0.6,
    target_size: float = 1.5,
    ground: bool = True,
    method: str = "boxes",
    max_boxes: int = 16,
) -> dict:
    """Reconstruct a ``.cgb`` from a 2x2 multi-view sheet via space carving.

    Pipeline: split sheet -> per-view silhouettes -> carve voxel occupancy ->
    segment the **front** cell into parts -> assign each part the carved voxels
    under its front silhouette -> fit a primitive (accurate 3D extent from the
    carving + 2D type prior) -> global scale + ground -> validated ``.cgb``.

    Reuses the single-view fitter, so type/ground/assembly behave identically;
    only the 3D evidence (a measured solid, not a guessed front shell) differs.
    """
    import cgb
    from PIL import Image

    from .segment import Segmenter, load_image_rgb
    from .shape2d import classify as classify_shape
    from .fit import fit_primitive, build_document, FitResult, _lowest_y

    sheet = load_image_rgb(sheet_path)
    cells = split_sheet(sheet)               # bg-subtraction silhouettes
    front = next(c for c in cells if c.name == "front")
    used = [c.name for c in cells if not c.blank]

    occ = carve_occupancy(cells, res=res)
    if not occ.any():
        raise RuntimeError("Space carving produced an empty volume — check the sheet.")

    H, W = front.silhouette.shape
    world = occupancy_to_world_points(occ)         # centred voxel cloud
    centroid = world.mean(0)
    ext = world.max(0) - world.min(0)
    scale = target_size / max(float(ext.max()), 1e-9)

    def front_color(cx: float, cy: float) -> tuple:
        ui = int(np.clip((cx + 0.5) * W, 0, W - 1))
        vi = int(np.clip((1.0 - (cy + 0.5)) * H, 0, H - 1))
        c = front.rgb[vi, ui].astype(float) / 255.0
        return (float(c[0]), float(c[1]), float(c[2]))

    fits: list = []
    if method == "boxes":
        # Tile the carved SOLID with axis-aligned boxes (no gaps / no floating
        # parts) — the faithful path. Each box becomes an axis-aligned cube.
        for centre, size in occupancy_to_boxes(occ, max_boxes=max_boxes):
            cs = (centre - centroid) * scale
            sz = np.maximum(size * scale, 1e-3)
            fits.append(FitResult(
                "cube",
                (float(cs[0]), float(cs[1]), float(cs[2])),
                (0.0, 0.0, 0.0),
                {"size": [float(sz[0]), float(sz[1]), float(sz[2])]},
                0.0,
                front_color(float(centre[0]), float(centre[1])),
            ))
    else:
        # "parts": segment the front cell and fit one primitive per part.
        segmenter = Segmenter(sam_checkpoint, model_type=sam_model_type, device=device)
        parts = segmenter.segment(front.rgb, max_masks=max_segments)
        ix, iy, iz = np.nonzero(occ)
        ui = np.clip(((ix + 0.5) / res) * W, 0, W - 1).astype(np.int32)
        vi = np.clip((1.0 - (iy + 0.5) / res) * H, 0, H - 1).astype(np.int32)
        world_s = (world - centroid) * scale
        assigned = np.zeros(len(world), dtype=bool)
        for m in sorted(parts, key=lambda p: p.area, reverse=True):
            sel = m.mask[vi, ui] & ~assigned
            if sel.sum() < 16:
                continue
            assigned |= sel
            prior = classify_shape(m.mask).get("prior")
            fit = fit_primitive(world_s[sel], color=_mean_color(front.rgb, m.mask),
                                shape_prior=prior, prior_weight=prior_weight)
            if fit is not None:
                fits.append(fit)
        if not fits:
            fits = [f for f in [fit_primitive(world_s)] if f is not None]

    if not fits:
        raise RuntimeError("No primitives could be fit from the carved volume.")

    if ground:
        y_min = min(_lowest_y(f) for f in fits)
        if np.isfinite(y_min):
            fits = [FitResult(f.prim_type,
                              (f.position[0], f.position[1] - y_min, f.position[2]),
                              f.rotation_euler, f.params, f.residual, f.color)
                    for f in fits]

    doc = build_document(fits, source_image=str(sheet_path))
    cgb.save(doc, out_path)
    return {
        "out_path": str(out_path), "views_used": used,
        "voxels": int(occ.sum()), "n_primitives": len(fits),
        "primitives": [{"type": f.prim_type} for f in fits],
    }


def _mean_color(rgb: np.ndarray, mask: np.ndarray):
    pix = np.asarray(rgb)[np.asarray(mask, dtype=bool)]
    if pix.size == 0:
        return (0.7, 0.7, 0.72)
    m = pix.reshape(-1, pix.shape[-1]).mean(0) / 255.0
    return (float(m[0]), float(m[1]), float(m[2]))
