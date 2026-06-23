"""Per-object reconstruction: turn ONE clean silhouette into a voxel + primitives.

The whole-scene carver merges touching parts (a held shield fuses into the
body). The fix is to reconstruct **one object at a time**: take a single clean
mask (a SAM automatic mask, or — better — a point/box-prompted SAM mask of just
that object) and build a solid for *only* that object.

Technique (single-view, silhouette-extrude):

1. Crop + scale the mask into a cubic voxel grid.
2. Extrude the 2D silhouette along depth (Z). With ``dome=True`` the thickness
   tapers from the silhouette's medial axis to its edge (a distance transform),
   giving a rounded, convex solid instead of a flat slab — right for shields,
   helmets, pauldrons, torsos, etc.
3. Optionally carry per-voxel colour sampled from the source image.

This is much cleaner per object than back-projecting a noisy monocular depth
cloud, and it is the building block for an object-by-object scene pipeline
(reconstruct each SAM object, then place each by its image position + depth).

Only numpy is required here; OpenCV is used for resizing / distance transform.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def reconstruct_object(
    mask: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    *,
    res: int = 72,
    depth_frac: float = 0.35,
    dome: bool = True,
    fill: float = 0.82,
) -> tuple[np.ndarray, dict]:
    """Reconstruct a single object's voxel solid from its 2D silhouette.

    Parameters
    ----------
    mask : (H, W) bool — the object's silhouette in the source image.
    rgb  : (H, W, 3) uint8 — optional source image, for per-voxel colour.
    res  : cubic voxel grid resolution.
    depth_frac : full depth extent as a fraction of ``res`` (object thickness).
    dome : taper thickness from the medial axis to the edge (rounded solid).
    fill : fraction of the grid the object's larger side fills.

    Returns ``(occ, colmap)`` — a boolean ``(res, res, res)`` grid (indexed
    ``[x, y, z]``, Y up, Z depth) and a ``{(ix, iy, iz): (r, g, b)}`` colour map
    (empty if ``rgb`` is None).
    """
    import cv2  # type: ignore

    mask = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return np.zeros((res, res, res), dtype=bool), {}
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1

    m = mask[y0:y1, x0:x1].astype(np.uint8)
    h, w = m.shape
    s = (fill * res) / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    m = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_NEAREST).astype(bool)

    sub = None
    if rgb is not None:
        sub = cv2.resize(np.asarray(rgb)[y0:y1, x0:x1], (nw, nh), interpolation=cv2.INTER_AREA)

    # Dome thickness from the normalised distance-to-edge (centre = thickest).
    if dome:
        dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
        dt = dt / (dt.max() or 1.0)
    else:
        dt = m.astype(float)

    occ = np.zeros((res, res, res), dtype=bool)
    colmap: dict = {}
    oy, ox = (res - nh) // 2, (res - nw) // 2
    zc = res // 2
    th = res * depth_frac / 2.0

    vv, uu = np.nonzero(m)
    for v, u in zip(vv, uu):
        ix = ox + u
        iy = res - 1 - (oy + v)        # image-down v → world-up y
        hz = th * np.sqrt(max(1e-3, dt[v, u])) if dome else th
        z0 = max(0, int(round(zc - hz)))
        z1 = min(res - 1, int(round(zc + hz)))
        occ[ix, iy, z0:z1 + 1] = True
        if sub is not None:
            c = sub[v, u].astype(float) / 255.0
            rgb_t = (float(c[0]), float(c[1]), float(c[2]))
            for iz in range(z0, z1 + 1):
                colmap[(ix, iy, iz)] = rgb_t
    return occ, colmap


def object_to_documents(
    occ: np.ndarray,
    colmap: Optional[dict] = None,
    *,
    max_prims: int = 6,
    target_size: float = 1.0,
    ground: bool = False,
):
    """Fit primitives to a reconstructed object and build ``.cgb`` documents.

    Returns ``(prim_doc, voxel_doc)`` — the fitted primitives and a coloured
    voxel debug doc — both in a shared, centred world frame.
    """
    import cgb

    from .primfit import decompose_occupancy
    from .multiview import _voxprim_to_fit
    from .fit import build_document, FitResult, _lowest_y

    pts = np.argwhere(occ).astype(float)
    if pts.size == 0:
        raise ValueError("empty occupancy")
    R = occ.shape[0]
    world = (pts + 0.5) / R - 0.5
    centroid = world.mean(0)
    ext = world.max(0) - world.min(0)
    scale = target_size / max(float(ext.max()), 1e-9)

    fits = [
        _voxprim_to_fit(vp, centroid, scale, lambda cx, cy: (0.72, 0.6, 0.42))
        for vp in decompose_occupancy(occ, max_prims=max_prims)
    ]
    if ground and fits:
        y_min = min(_lowest_y(f) for f in fits)
        if np.isfinite(y_min):
            fits = [FitResult(f.prim_type,
                              (f.position[0], f.position[1] - y_min, f.position[2]),
                              f.rotation_euler, f.params, f.residual, f.color) for f in fits]
    prim_doc = build_document(fits)

    voxel_doc = cgb.new_document()
    voxel_doc["metadata"]["generator"] = "CubeGB object voxel"
    size = float(scale / R) * 1.03
    for k, (ix, iy, iz) in enumerate(pts.astype(int)):
        cn = (np.array([ix, iy, iz]) + 0.5) / R - 0.5
        p = (cn - centroid) * scale
        col = list((colmap or {}).get((int(ix), int(iy), int(iz)), (0.6, 0.5, 0.4)))
        cgb.add_primitive(voxel_doc, cgb.cube(
            f"v{k}", [size, size, size],
            transform=cgb.make_transform(position=[float(p[0]), float(p[1]), float(p[2])]),
            color=col))
    return prim_doc, voxel_doc
