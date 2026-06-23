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


def align_views(cells: list["ViewCell"], *, target_fill: float = 0.9) -> None:
    """Centre and commonly-scale each view's silhouette in place.

    The 2x2 views of a sheet are often drawn off-centre or at slightly different
    scales, so their projection cones miss each other and space carving collapses
    to almost nothing. This re-centres each silhouette in its cell and applies a
    **single shared scale** (so the largest view fills ``target_fill`` of the
    cell) — fixing position/scale mismatch while preserving relative proportions.
    Blank cells are left untouched. No-op without OpenCV.
    """
    try:
        import cv2  # type: ignore
    except ImportError:  # pragma: no cover
        return

    boxes = []
    for c in cells:
        if c.blank or not c.silhouette.any():
            boxes.append(None)
            continue
        ys, xs = np.nonzero(c.silhouette)
        boxes.append((int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())))

    exts = [max(b[1] - b[0] + 1, b[3] - b[2] + 1) for b in boxes if b]
    if not exts:
        return
    H, W = cells[0].silhouette.shape
    scale = (target_fill * min(H, W)) / max(exts)

    for c, b in zip(cells, boxes):
        if b is None:
            continue
        crop = c.silhouette[b[0]:b[1] + 1, b[2]:b[3] + 1].astype(np.uint8)
        nh = max(1, min(H, int(round(crop.shape[0] * scale))))
        nw = max(1, min(W, int(round(crop.shape[1] * scale))))
        r = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_NEAREST).astype(bool)
        oy, ox = (H - nh) // 2, (W - nw) // 2
        out = np.zeros((H, W), dtype=bool)
        out[oy:oy + nh, ox:ox + nw] = r
        c.silhouette = out

        # Align the RGB identically so colour sampling stays registered with the
        # carved silhouette (else front/multi-view colour is offset).
        rgb_crop = c.rgb[b[0]:b[1] + 1, b[2]:b[3] + 1]
        rgb_r = cv2.resize(rgb_crop, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.tile(_bg_color(c.rgb).astype(np.uint8), (H, W, 1))
        canvas[oy:oy + nh, ox:ox + nw] = rgb_r
        c.rgb = canvas


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
    if name == "side":            # side profile faces image-right = world +Z (front)
        return z, 1.0 - y
    if name == "top":             # looking down: image-up (v=0) = world +Z (front)
        return x, 1.0 - z
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


def _downsample_max(occ: np.ndarray, factor: int) -> np.ndarray:
    """Block max-pool an occupancy grid by an integer ``factor`` (occupied if any)."""
    if factor <= 1:
        return occ
    R = occ.shape[0]
    pad = (-R) % factor
    if pad:
        occ = np.pad(occ, ((0, pad), (0, pad), (0, pad)))
    n = occ.shape[0] // factor
    return occ.reshape(n, factor, n, factor, n, factor).any(axis=(1, 3, 5))


def _surface_voxels(occ: np.ndarray) -> np.ndarray:
    """Occupied voxels with at least one empty 6-neighbour (a hollow shell)."""
    nbr_all = np.ones_like(occ)
    for axis in range(3):
        for shift in (1, -1):
            nbr_all &= np.roll(occ, shift, axis=axis)
            # voxels rolled across the border are not real neighbours → empty
            sl = [slice(None)] * 3
            sl[axis] = (0 if shift == 1 else -1)
            nbr_all[tuple(sl)] = False
    return occ & ~nbr_all


def _voxel_view(vox: np.ndarray, ix: int, iy: int, iz: int) -> str:
    """Pick the 2x2-sheet view that looks onto this surface voxel's exposed face.

    Priority front(+Z) > side(+X) > top(+Y) > back(-Z); -X falls back to side and
    everything else to front (the sheet has no left/bottom view).
    """
    vr = vox.shape[0]

    def empty(ax: int, d: int) -> bool:
        j = [ix, iy, iz]
        j[ax] += d
        if j[ax] < 0 or j[ax] >= vr:
            return True
        return not vox[j[0], j[1], j[2]]

    if empty(2, 1):
        return "front"
    if empty(0, 1):
        return "side"
    if empty(1, 1):
        return "top"
    if empty(2, -1):
        return "back"
    if empty(0, -1):
        return "side"
    return "front"


def occupancy_to_voxel_doc(
    occ: np.ndarray, centroid, scale, y_min: float = 0.0, *,
    view_res: int = 44, max_cubes: int = 4000, source_image=None,
    color=(0.45, 0.55, 0.7), color_fn=None, object_fn=None, mv_color_fn=None,
) -> dict:
    """Build a ``.cgb`` of cubes visualising the carved voxel solid (debug view).

    Downsamples to ``view_res`` and keeps only surface voxels (a hollow shell) so
    the cube count stays viewer-friendly, then renders each as a cube in the SAME
    world frame as the fitted primitives (shared ``centroid`` / ``scale`` /
    ``y_min``) so the two line up when shown side by side.

    ``color_fn(cx, cy) -> (r, g, b)`` colours each voxel by sampling a view (e.g.
    the front cell) at the voxel's normalised world ``(x, y)``; if ``None`` a flat
    ``color`` is used.
    """
    import cgb

    R = occ.shape[0]
    factor = max(1, int(np.ceil(R / view_res)))
    vox = _downsample_max(occ, factor)
    vr = vox.shape[0]
    surf = _surface_voxels(vox)
    idx = np.argwhere(surf)
    if len(idx) > max_cubes:                      # thin out evenly if still dense
        keep = np.linspace(0, len(idx) - 1, max_cubes).astype(int)
        idx = idx[keep]

    doc = cgb.new_document(source_image=str(source_image) if source_image else None)
    doc["metadata"]["generator"] = "CubeGB voxel debug"
    size = float(scale / vr) * 1.03               # slight overlap hides seams
    flat = [float(c) for c in color]
    def _hex(rgb):
        return "#%02x%02x%02x" % tuple(int(np.clip(c, 0, 1) * 255) for c in rgb)

    for k, (ix, iy, iz) in enumerate(idx):
        cn = (np.array([ix, iy, iz]) + 0.5) / vr - 0.5
        p = (cn - centroid) * scale
        front = color_fn(float(cn[0]), float(cn[1])) if color_fn else flat
        # material.color = multi-view colour (sampled from the view facing this
        # voxel's exposed face); the front-only colour is kept in `name` (hex) so
        # the UI can show a front-vs-multiview comparison.
        if mv_color_fn is not None:
            view = _voxel_view(vox, ix, iy, iz)
            mv = mv_color_fn(view, (ix + 0.5) / vr, (iy + 0.5) / vr, (iz + 0.5) / vr)
            col = [float(c) for c in (mv if mv is not None else front)]
        else:
            col = [float(c) for c in front]
        # Object group id (from front SAM segments) → material.name (obj3 / bg).
        oid = object_fn(float(cn[0]), float(cn[1])) if object_fn else None
        mname = f"obj{oid}" if (oid is not None and oid >= 0) else "bg"
        cgb.add_primitive(doc, cgb.cube(
            f"v{k}", [size, size, size], name=_hex(front),
            transform=cgb.make_transform(position=[float(p[0]), float(p[1] - y_min), float(p[2])]),
            color=col, material_name=mname))
    return doc


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
    res: int = 128,
    fit_res: int = 128,
    max_segments: int = 12,
    prior_weight: float = 0.6,
    target_size: float = 1.5,
    ground: bool = True,
    align: bool = True,
    method: str = "primitives",
    max_boxes: int = 24,
    voxel_out_path: Optional[str] = None,
    segment_objects: bool = True,
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
    if align:
        align_views(cells)                   # fix off-centre / mismatched-scale views
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

    # Multi-view colour: sample the RGB of the view facing a voxel's exposed face
    # (front/side/back/top), so sides and the back get their own colour instead
    # of the front colour smeared across them. Falls back to the front view.
    cells_by_name = {c.name: c for c in cells}

    def view_rgb(viewname, x01, y01, z01):
        c = cells_by_name.get(viewname)
        if c is None or c.blank:
            viewname, c = "front", front
        u, v = _project(viewname, x01, y01, z01)
        Hc, Wc = c.rgb.shape[:2]
        ui = int(np.clip(u * Wc, 0, Wc - 1))
        vi = int(np.clip(v * Hc, 0, Hc - 1))
        px = c.rgb[vi, ui].astype(float) / 255.0
        return (float(px[0]), float(px[1]), float(px[2]))

    # Object grouping: SAM-segment the front view and tag each pixel with an
    # object id, so voxels can record which object they belong to (material.name).
    # This accumulates structure for future per-object simplification.
    object_fn = None
    n_objects = 0
    if voxel_out_path is not None and segment_objects:
        try:
            seg = Segmenter(sam_checkpoint, model_type=sam_model_type, device=device)
            obj_masks = seg.segment(front.rgb, max_masks=max_segments)
            idmap = -np.ones((H, W), dtype=np.int32)
            for oid, m in enumerate(sorted(obj_masks, key=lambda p: p.area, reverse=True)):
                idmap[np.asarray(m.mask, bool) & (idmap < 0)] = oid
            n_objects = len(obj_masks)

            def object_fn(cx: float, cy: float, _m=idmap) -> int:
                ui = int(np.clip((cx + 0.5) * W, 0, W - 1))
                vi = int(np.clip((1.0 - (cy + 0.5)) * H, 0, H - 1))
                return int(_m[vi, ui])
        except Exception:
            object_fn = None  # segmentation is best-effort; never block generation

    fits: list = []
    if method == "primitives":
        # Recursive shape abstraction: explain the carved solid with VARIED
        # volumetric primitives (cube/cylinder/cone/sphere), choosing each part's
        # type by IoU and partitioning the voxels (little overlap). See
        # recognition.primfit.
        from .primfit import decompose_occupancy
        # Decouple: fit primitives on a (possibly downsampled) grid so a high
        # carving res — used for the voxel debug view — doesn't make the
        # decomposition crawl (256+ would take minutes at full res).
        occ_fit = occ
        if fit_res and res > fit_res:
            occ_fit = _downsample_max(occ, int(np.ceil(res / fit_res)))

        for vp in decompose_occupancy(occ_fit, max_prims=max_boxes):
            fits.append(_voxprim_to_fit(vp, centroid, scale, front_color))
    elif method == "boxes":
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

    y_min = 0.0
    if ground:
        y_min = min(_lowest_y(f) for f in fits)
        if not np.isfinite(y_min):
            y_min = 0.0
        fits = [FitResult(f.prim_type,
                          (f.position[0], f.position[1] - y_min, f.position[2]),
                          f.rotation_euler, f.params, f.residual, f.color)
                for f in fits]

    doc = build_document(fits, source_image=str(sheet_path))
    cgb.save(doc, out_path)

    summary = {
        "out_path": str(out_path), "views_used": used, "res": int(res),
        "fit_res": int(min(res, fit_res)) if fit_res else int(res),
        "voxels": int(occ.sum()), "n_primitives": len(fits),
        "n_objects": int(n_objects),
        "primitives": [{"type": f.prim_type} for f in fits],
    }

    # Optional: also emit the carved voxel solid as a viewable .cgb (debug view),
    # carrying per-voxel front colour AND object id (material.name). The display
    # resolution scales with the carve res but is capped for the viewer.
    if voxel_out_path is not None:
        vdoc = occupancy_to_voxel_doc(
            occ, centroid, scale, y_min, source_image=sheet_path,
            color_fn=front_color, mv_color_fn=view_rgb, object_fn=object_fn,
            view_res=min(int(res), 72), max_cubes=16000)
        cgb.save(vdoc, voxel_out_path)
        summary["voxel_out_path"] = str(voxel_out_path)
        summary["voxel_cubes"] = len(vdoc["primitives"])
    return summary


import math

# CGB cylinder/cone point along +Y locally; rotate +Y onto the world axis.
_CYL_EULER = {0: (0.0, 0.0, -math.pi / 2), 1: (0.0, 0.0, 0.0), 2: (math.pi / 2, 0.0, 0.0)}
# Cone apex direction (axis, apex_high) -> euler mapping +Y onto that direction.
_CONE_EULER = {
    (0, True): (0.0, 0.0, -math.pi / 2), (0, False): (0.0, 0.0, math.pi / 2),
    (1, True): (0.0, 0.0, 0.0), (1, False): (math.pi, 0.0, 0.0),
    (2, True): (math.pi / 2, 0.0, 0.0), (2, False): (-math.pi / 2, 0.0, 0.0),
}


def _voxprim_to_fit(vp, centroid, scale, front_color):
    """Convert a :class:`recognition.primfit.VoxPrim` to a ``fit.FitResult``."""
    from .fit import FitResult

    cs = (np.asarray(vp.center) - centroid) * scale
    pos = (float(cs[0]), float(cs[1]), float(cs[2]))
    color = front_color(float(vp.center[0]), float(vp.center[1]))

    if vp.type == "cube":
        sz = np.maximum(np.asarray(vp.size) * scale, 1e-3)
        return FitResult("cube", pos, (0.0, 0.0, 0.0),
                         {"size": [float(sz[0]), float(sz[1]), float(sz[2])]}, vp.iou, color)
    if vp.type == "sphere":
        return FitResult("sphere", pos, (0.0, 0.0, 0.0),
                         {"radius": float(max(vp.radius * scale, 1e-3)), "segments": 16}, vp.iou, color)
    if vp.type == "cylinder":
        return FitResult("cylinder", pos, _CYL_EULER[vp.axis],
                         {"radius": float(max(vp.radius * scale, 1e-3)),
                          "height": float(max(vp.height * scale, 1e-3)), "segments": 16}, vp.iou, color)
    if vp.type == "cone":
        return FitResult("cone", pos, _CONE_EULER[(vp.axis, vp.apex_high)],
                         {"radius": float(max(vp.radius * scale, 1e-3)),
                          "height": float(max(vp.height * scale, 1e-3)), "segments": 16}, vp.iou, color)
    raise ValueError(vp.type)


def _mean_color(rgb: np.ndarray, mask: np.ndarray):
    pix = np.asarray(rgb)[np.asarray(mask, dtype=bool)]
    if pix.size == 0:
        return (0.7, 0.7, 0.72)
    m = pix.reshape(-1, pix.shape[-1]).mean(0) / 255.0
    return (float(m[0]), float(m[1]), float(m[2]))
