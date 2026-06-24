"""Per-part composition — keep part identity, place parts in depth.

The honest finding from ``mesh_fit`` was that abstracting a whole composite shape
loses semantics (a character collapses into geometric blobs). The fix that stays
within *direction 1* (existing libraries, no LLM / training) is **per-part**:

    SAM (parts) → reconstruct/abstract each part on its own → place in a shared
    world frame by image position + depth → compose, keeping each part's identity
    and colour.

The unifying abstraction is that **each part contributes an occupancy grid plus a
2D image position, a depth, and a colour**. Where the occupancy comes from is
pluggable:

* :func:`part_from_silhouette` — the existing dome silhouette-extrude
  (``object_recon.reconstruct_object``); works today, no extra model.
* :func:`part_from_mesh` — voxelise a per-part mesh from an image-to-3D model via
  the :mod:`recognition.mesh_fit` bridge (real volume per part).

:func:`compose_parts` then fits primitives to each part and places it. The new
piece over ``image_to_cgb_selected`` is **per-part depth (``z``)** — the missing
dimension that kept earlier multi-part composition flat.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# --------------------------------------------------------------------------- #
# A "part" = occupancy + 2D bbox (in image px) + relative depth + colour
# --------------------------------------------------------------------------- #
def _bbox_of(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())


def part_from_silhouette(
    mask: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    *,
    obj_id: int = 0,
    z: Optional[float] = None,
    res: int = 72,
    depth_frac: float = 0.35,
) -> dict:
    """Build a part from a 2D silhouette via dome extrude (no extra model)."""
    from recognition.object_recon import reconstruct_object, _colmap_mean

    occ, colmap = reconstruct_object(mask, rgb, res=res, depth_frac=depth_frac, dome=True)
    x0, x1, y0, y1 = _bbox_of(mask)
    return {
        "id": int(obj_id), "occ": occ, "bbox": (x0, x1, y0, y1),
        "z": z, "color": _colmap_mean(colmap), "hw": mask.shape[:2],
    }


def part_from_mesh(
    mesh,
    mask: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    *,
    obj_id: int = 0,
    z: Optional[float] = None,
    res: int = 64,
    up: str = "y",
) -> dict:
    """Build a part by voxelising a per-part image-to-3D mesh (the bridge)."""
    from recognition.mesh_fit import mesh_to_occupancy, _as_mesh, _mean_color

    occ = mesh_to_occupancy(mesh, res=res, up=up)
    x0, x1, y0, y1 = _bbox_of(mask)
    color = _mean_color(_as_mesh(mesh)) or (0.7, 0.6, 0.45)
    return {
        "id": int(obj_id), "occ": occ, "bbox": (x0, x1, y0, y1),
        "z": z, "color": color, "hw": mask.shape[:2],
    }


# --------------------------------------------------------------------------- #
# Composition
# --------------------------------------------------------------------------- #
def compose_parts(
    parts: list,
    *,
    target_size: float = 1.5,
    depth_span: float = 0.6,
    ground: bool = True,
    per_object_prims: int = 8,
    source_image: Optional[str] = None,
) -> dict:
    """Fit primitives to each part and place them in one shared world frame.

    Each part is positioned by its 2D image bbox centre (x, y) and its relative
    depth ``z`` in ``[0, 1]`` (0 = back, 1 = front) mapped through ``depth_span``.
    A part with ``z is None`` sits on the ``z = 0`` plane (flat fallback). Per-part
    colour and identity are preserved. Returns a validated ``.cgb`` document.
    """
    import cgb

    from recognition.fit import build_document, FitResult
    from recognition.oriented_fit import fit_oriented_primitives
    from recognition.object_recon import _min_y_of

    all_fits: list = []
    for part in parts:
        occ = part["occ"]
        if occ is None or not occ.any():
            continue
        R = occ.shape[0]
        norm = (np.argwhere(occ) + 0.5) / R - 0.5
        centroid = norm.mean(0)
        ext = float((norm.max(0) - norm.min(0)).max())

        x0, x1, y0, y1 = part["bbox"]
        H, W = part["hw"]
        maxwh = max(H, W)
        obj_px = max(x1 - x0, y1 - y0) + 1
        scale = (obj_px / maxwh * target_size) / max(ext, 1e-9)

        cx = ((x0 + x1) / 2.0 - W / 2.0) / maxwh * target_size
        cy = (H / 2.0 - (y0 + y1) / 2.0) / maxwh * target_size
        z = part.get("z")
        cz = ((float(z) - 0.5) * depth_span * target_size) if z is not None else 0.0
        offset = np.array([cx, cy, cz])

        color = part.get("color") or (0.7, 0.6, 0.45)
        pts_world = (norm - centroid) * scale + offset
        all_fits.extend(
            fit_oriented_primitives(pts_world, max_prims=per_object_prims, color=color)
        )

    if not all_fits:
        raise RuntimeError("No primitives could be fit from the given parts.")

    # Centre the group in x/z; ground on y.
    pos = np.array([f.position for f in all_fits])
    shift = np.array([pos[:, 0].mean(), 0.0, pos[:, 2].mean()])
    y_min = min(_min_y_of(f) for f in all_fits)
    shift[1] = y_min if (ground and np.isfinite(y_min)) else 0.0

    placed = [
        FitResult(
            f.prim_type,
            (f.position[0] - shift[0], f.position[1] - shift[1], f.position[2] - shift[2]),
            f.rotation_euler, f.params, f.residual, f.color,
        )
        for f in all_fits
    ]
    doc = build_document(placed, source_image=source_image)
    cgb.validate(doc)
    return doc


# --------------------------------------------------------------------------- #
# Image entry (SAM parts + Depth Anything per-part z) — needs the models
# --------------------------------------------------------------------------- #
def image_to_cgb_composed(
    image_path: str,
    out_path: str,
    *,
    sam_checkpoint: str,
    depth_checkpoint: Optional[str] = None,
    device: Optional[str] = None,
    sam_model_type: str = "vit_h",
    max_objects: int = 12,
    res: int = 72,
    depth_frac: float = 0.35,
    target_size: float = 1.5,
    depth_span: float = 0.6,
    per_object_prims: int = 6,
    ground: bool = True,
) -> dict:
    """SAM parts + Depth Anything per-part depth → composed, depth-placed ``.cgb``.

    Each SAM part is reconstructed in isolation (dome extrude), coloured by its
    mean colour, and placed by its 2D bbox plus its **mean monocular depth** so
    parts separate front-to-back instead of collapsing onto one plane. (Monocular
    depth on flat art is coarse; swap in :func:`part_from_mesh` with a per-part
    image-to-3D mesh for true volume.) Requires the recognition stack + weights.
    """
    import cgb

    from recognition.segment import Segmenter, load_image_rgb
    from recognition.object_recon import partition_objects

    img = load_image_rgb(image_path)
    H, W = img.shape[:2]
    masks = Segmenter(sam_checkpoint, model_type=sam_model_type, device=device).segment(
        img, max_masks=max_objects
    )
    objects = partition_objects(masks, H, W)
    if not objects:
        raise RuntimeError("No parts segmented.")

    depth = _mean_depths(img, objects, depth_checkpoint, device)

    parts = [
        part_from_silhouette(sel, img, obj_id=int(oid), z=depth.get(int(oid)),
                             res=res, depth_frac=depth_frac)
        for oid, sel in objects
    ]
    doc = compose_parts(parts, target_size=target_size, depth_span=depth_span,
                        ground=ground, per_object_prims=per_object_prims,
                        source_image=str(image_path))
    cgb.save(doc, out_path)
    return {
        "out_path": str(out_path), "n_parts": len(parts),
        "n_primitives": len(doc["primitives"]),
        "depth_used": bool(depth),
    }


def _mean_depths(img, objects, depth_checkpoint, device) -> dict:
    """Per-part mean monocular depth normalised to [0,1] (1 = nearest). {} on failure."""
    try:
        from recognition.depth import DepthEstimator
        dmap = DepthEstimator(depth_checkpoint, device=device).estimate(img)
        dmap = np.asarray(dmap, dtype=float)
        lo, hi = float(dmap.min()), float(dmap.max())
        norm = (dmap - lo) / (hi - lo) if hi > lo else np.zeros_like(dmap)
        return {int(oid): float(norm[sel].mean()) for oid, sel in objects if sel.any()}
    except Exception:  # noqa: BLE001 - depth is optional; fall back to flat z=None
        return {}
