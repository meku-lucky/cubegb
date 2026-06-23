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


def partition_objects(masks, H: int, W: int, *, min_area_frac: float = 0.004) -> list:
    """Turn overlapping SAM masks into a non-overlapping object label map.

    SAM returns a hierarchy (a whole-figure mask plus its parts). Assigning each
    pixel to the **smallest** mask that covers it makes parts win over the whole,
    yielding a clean partition. Returns ``[(label_id, bool_mask)]`` for objects
    above ``min_area_frac`` of the frame.
    """
    order = sorted(range(len(masks)), key=lambda i: int(masks[i].area))
    labelmap = -np.ones((H, W), dtype=np.int32)
    for i in order:
        m = np.asarray(masks[i].mask, bool) & (labelmap < 0)
        labelmap[m] = i
    out = []
    min_area = min_area_frac * H * W
    for i in order:
        sel = labelmap == i
        if sel.sum() >= min_area:
            out.append((i, sel))
    return out


def image_to_cgb_objects(
    image_path: str,
    out_path: str,
    *,
    sam_checkpoint: str,
    depth_checkpoint: Optional[str] = None,
    device: Optional[str] = None,
    sam_model_type: str = "vit_h",
    max_objects: int = 12,
    res: int = 96,
    depth_frac: float = 0.32,
    depth_span: float = 0.55,
    target_size: float = 1.5,
    ground: bool = True,
    per_object_prims: int = 5,
    voxel_out_path: Optional[str] = None,
    masks=None,
    select_ids=None,
) -> dict:
    """Object-by-object reconstruction from a single image. **Experimental.**

    Segment the image into non-overlapping objects, reconstruct each by
    silhouette-extrude into a **shared** world grid (placed in depth by the depth
    map), then fit primitives **per object** so distinct parts (cat / sword /
    shield / armour / cape) stay clean and separated instead of fusing.

    Caveat: monocular depth on flat/stylised concept art barely separates parts
    in z, so the whole assembled scene comes out as a near-flat relief and thin
    body parts fit as stray cylinders. Per-object reconstruction shines for an
    **isolated** object (see :func:`reconstruct_object` — a prompted shield comes
    out clean); good full-scene composition needs real per-object depth, e.g.
    each object's silhouette across a multi-view sheet (future work).
    """
    import cgb

    from .segment import Segmenter, load_image_rgb
    from .depth import DepthEstimator
    from .fit import build_document, FitResult, _lowest_y

    img = load_image_rgb(image_path)
    H, W = img.shape[:2]

    if masks is None:
        masks = Segmenter(sam_checkpoint, model_type=sam_model_type, device=device).segment(
            img, max_masks=max_objects)
    objects = partition_objects(masks, H, W)
    if select_ids is not None:                      # keep only the chosen objects
        want = set(int(i) for i in select_ids)
        objects = [(i, sel) for (i, sel) in objects if int(i) in want]
    if not objects:
        raise RuntimeError("No objects selected/segmented — pick at least one part.")

    depth = DepthEstimator(depth_checkpoint, device=device).estimate(img)
    dn = depth.astype(np.float64)
    dn = (dn - dn.min()) / (float(np.ptp(dn)) or 1.0)   # 0..1, larger = nearer

    occ, objid, colmap = _place_objects(
        img, objects, dn, res=res, depth_frac=depth_frac, depth_span=depth_span)
    if not occ.any():
        raise RuntimeError("Reconstruction produced an empty volume.")

    world = (np.argwhere(occ) + 0.5) / res - 0.5
    centroid = world.mean(0)
    scale = target_size / max(float((world.max(0) - world.min(0)).max()), 1e-9)

    def color_at(ix, iy, iz):
        return colmap.get((ix, iy, iz), (0.6, 0.55, 0.5))

    # Fit primitives PER object (oriented/OBB) so parts stay separate AND rotated.
    from .oriented_fit import fit_oriented_primitives
    fits: list = []
    obj_summ = []
    for oid, _ in objects:
        sub = occ & (objid == oid)
        n = int(sub.sum())
        if n < 8:
            continue
        ocol = _mean_obj_color(colmap, objid, oid)
        pts_o = ((np.argwhere(sub) + 0.5) / res - 0.5 - centroid) * scale
        new = fit_oriented_primitives(pts_o, max_prims=per_object_prims, color=ocol)
        fits.extend(new)
        obj_summ.append({"object": int(oid), "voxels": n, "primitives": len(new)})

    if not fits:
        raise RuntimeError("No primitives could be fit.")

    if ground:
        y_min = min(_lowest_y(f) for f in fits)
        if np.isfinite(y_min):
            fits = [FitResult(f.prim_type,
                              (f.position[0], f.position[1] - y_min, f.position[2]),
                              f.rotation_euler, f.params, f.residual, f.color) for f in fits]

    doc = build_document(fits, source_image=str(image_path))
    cgb.save(doc, out_path)
    summary = {
        "out_path": str(out_path), "n_objects": len(objects),
        "n_primitives": len(fits), "voxels": int(occ.sum()),
        "objects": obj_summ,
        "primitives": [{"type": f.prim_type} for f in fits],
    }

    if voxel_out_path is not None:
        vdoc = _voxel_doc(occ, objid, colmap, centroid, scale, res, ground=ground, fits=fits)
        cgb.save(vdoc, voxel_out_path)
        summary["voxel_out_path"] = str(voxel_out_path)
        summary["voxel_cubes"] = len(vdoc["primitives"])
    return summary


def _place_objects(img, objects, dn, *, res, depth_frac, depth_span):
    """Rasterise every object's domed silhouette into one shared world grid."""
    import cv2  # type: ignore

    H, W = img.shape[:2]
    maxwh = max(H, W)
    sc = res / maxwh                              # pixels → grid cells
    gw, gh = int(round(W * sc)), int(round(H * sc))
    gx0, gy0 = (res - gw) // 2, (res - gh) // 2

    occ = np.zeros((res, res, res), dtype=bool)
    objid = -np.ones((res, res, res), dtype=np.int32)
    colmap: dict = {}

    img_g = cv2.resize(img, (gw, gh), interpolation=cv2.INTER_AREA)
    dn_g = cv2.resize(dn.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA)

    for oid, sel in objects:
        mg = cv2.resize(sel.astype(np.uint8), (gw, gh), interpolation=cv2.INTER_NEAREST).astype(bool)
        if not mg.any():
            continue
        # Per-object thickness ∝ the object's own in-plane size, so a round shield
        # gets a round depth and a thin sword a thin depth (a global thickness
        # made everything a uniform flat slab).
        ys2, xs2 = np.nonzero(mg)
        obj_size = max(int(xs2.max() - xs2.min()) + 1, int(ys2.max() - ys2.min()) + 1)
        th = depth_frac * 0.5 * obj_size
        zc = res / 2.0 + (float(dn_g[mg].mean()) - 0.5) * depth_span * res
        dt = cv2.distanceTransform(mg.astype(np.uint8), cv2.DIST_L2, 3)
        dt = dt / (dt.max() or 1.0)
        rows, cols = np.nonzero(mg)
        for r, c in zip(rows, cols):
            ix = gx0 + c
            iy = res - 1 - (gy0 + r)
            if not (0 <= ix < res and 0 <= iy < res):
                continue
            hz = th * np.sqrt(max(1e-3, dt[r, c]))
            z0 = max(0, int(round(zc - hz)))
            z1 = min(res - 1, int(round(zc + hz)))
            occ[ix, iy, z0:z1 + 1] = True
            objid[ix, iy, z0:z1 + 1] = oid
            col = tuple(float(v) for v in img_g[r, c].astype(float) / 255.0)
            for iz in range(z0, z1 + 1):
                colmap[(ix, iy, iz)] = col
    return occ, objid, colmap


def _mean_obj_color(colmap, objid, oid):
    cols = [colmap[k] for k in colmap if objid[k] == oid]
    if not cols:
        return (0.6, 0.55, 0.5)
    a = np.mean(cols, axis=0)
    return (float(a[0]), float(a[1]), float(a[2]))


def _voxel_doc(occ, objid, colmap, centroid, scale, res, *, ground, fits):
    """Coloured voxel debug doc, grounded to match the fitted primitives."""
    import cgb

    doc = cgb.new_document()
    doc["metadata"]["generator"] = "CubeGB object voxels"
    size = float(scale / res) * 1.03
    pts = np.argwhere(occ)
    # Ground by the grid's own lowest world-y (≈ the fits' lowest point).
    ys = (((pts[:, 1] + 0.5) / res - 0.5) - centroid[1]) * scale
    yshift = float(ys.min()) if ground else 0.0
    for k, (ix, iy, iz) in enumerate(pts):
        cn = (np.array([ix, iy, iz]) + 0.5) / res - 0.5
        p = (cn - centroid) * scale
        col = list(colmap.get((int(ix), int(iy), int(iz)), (0.6, 0.55, 0.5)))
        cgb.add_primitive(doc, cgb.cube(
            f"v{k}", [size, size, size], material_name=f"obj{int(objid[ix, iy, iz])}",
            transform=cgb.make_transform(position=[float(p[0]), float(p[1] - yshift), float(p[2])]),
            color=col))
    return doc


def image_to_cgb_selected(
    image_path: str,
    out_path: str,
    *,
    sam_checkpoint: str,
    masks=None,
    select_ids=None,
    device: Optional[str] = None,
    sam_model_type: str = "vit_h",
    max_objects: int = 12,
    res: int = 80,
    depth_frac: float = 0.42,
    target_size: float = 1.5,
    per_object_prims: int = 8,
    ground: bool = True,
    voxel_out_path: Optional[str] = None,
) -> dict:
    """Reconstruct the **selected** objects, each in **isolation** (high quality).

    Each picked object is reconstructed on its own (dome silhouette-extrude) and
    oriented-fit — so one part (a shield → clean disc, a sword → blade) comes out
    well instead of being squashed in a shared scene grid. A single selection is
    just that part; multiple are placed by their image position (depth/composition
    deliberately deferred). Returns the same summary shape as the other pipelines.
    """
    import cgb

    from .segment import Segmenter, load_image_rgb
    from .fit import build_document, FitResult
    from .oriented_fit import fit_oriented_primitives

    img = load_image_rgb(image_path)
    H, W = img.shape[:2]
    if masks is None:
        masks = Segmenter(sam_checkpoint, model_type=sam_model_type, device=device).segment(
            img, max_masks=max_objects)
    objects = partition_objects(masks, H, W)
    if select_ids is not None:
        want = set(int(i) for i in select_ids)
        objects = [(i, sel) for (i, sel) in objects if int(i) in want]
    if not objects:
        raise RuntimeError("No objects selected — pick at least one part.")

    maxwh = max(H, W)
    multi = len(objects) > 1
    all_fits: list = []
    voxels: list = []                                   # (pos(3,), size, color, oid)
    obj_summ = []
    for oid, sel in objects:
        occ, colmap = reconstruct_object(sel, img, res=res, depth_frac=depth_frac, dome=True)
        if not occ.any():
            continue
        R = occ.shape[0]
        idx = np.argwhere(occ)
        norm = (idx + 0.5) / R - 0.5
        centroid = norm.mean(0)
        ext = float((norm.max(0) - norm.min(0)).max())
        ys, xs = np.nonzero(sel)
        obj_px = max(int(xs.max() - xs.min()), int(ys.max() - ys.min())) + 1
        s = (obj_px / maxwh * target_size) / max(ext, 1e-9)     # normalised → world
        if multi:                                                # place by image position
            cx = ((xs.min() + xs.max()) / 2 - W / 2) / maxwh * target_size
            cy = (H / 2 - (ys.min() + ys.max()) / 2) / maxwh * target_size
            offset = np.array([cx, cy, 0.0])
        else:
            offset = np.zeros(3)

        ocol = _colmap_mean(colmap)
        pts_world = (norm - centroid) * s + offset
        new = fit_oriented_primitives(pts_world, max_prims=per_object_prims, color=ocol)
        all_fits.extend(new)
        obj_summ.append({"object": int(oid), "voxels": int(occ.sum()), "primitives": len(new)})

        from .multiview import _surface_voxels       # hollow shell keeps the cube
        vsize = float(s / R) * 1.03                   # count viewer-friendly
        for (ix, iy, iz) in np.argwhere(_surface_voxels(occ)):
            cn = (np.array([ix, iy, iz]) + 0.5) / R - 0.5
            p = (cn - centroid) * s + offset
            voxels.append((p, vsize, list(colmap.get((int(ix), int(iy), int(iz)), (0.6, 0.55, 0.5))), int(oid)))

    if not all_fits:
        raise RuntimeError("No primitives could be fit.")

    # Centre the group in x/z and ground it on y.
    pos = np.array([f.position for f in all_fits])
    shift = np.array([pos[:, 0].mean(), 0.0, pos[:, 2].mean()])
    y_min = min(_min_y_of(f) for f in all_fits)
    shift[1] = y_min if (ground and np.isfinite(y_min)) else 0.0

    all_fits = [FitResult(f.prim_type,
                          (f.position[0] - shift[0], f.position[1] - shift[1], f.position[2] - shift[2]),
                          f.rotation_euler, f.params, f.residual, f.color) for f in all_fits]
    doc = build_document(all_fits, source_image=str(image_path))
    cgb.save(doc, out_path)

    summary = {
        "out_path": str(out_path), "n_objects": len(obj_summ),
        "n_primitives": len(all_fits), "voxels": sum(o["voxels"] for o in obj_summ),
        "objects": obj_summ, "primitives": [{"type": f.prim_type} for f in all_fits],
    }
    if voxel_out_path is not None:
        vdoc = cgb.new_document()
        vdoc["metadata"]["generator"] = "CubeGB selected voxels"
        for k, (p, sz, col, oid) in enumerate(voxels):
            cgb.add_primitive(vdoc, cgb.cube(
                f"v{k}", [sz, sz, sz], material_name=f"obj{oid}",
                transform=cgb.make_transform(position=[float(p[0] - shift[0]),
                                                       float(p[1] - shift[1]),
                                                       float(p[2] - shift[2])]),
                color=col))
        cgb.save(vdoc, voxel_out_path)
        summary["voxel_out_path"] = str(voxel_out_path)
        summary["voxel_cubes"] = len(vdoc["primitives"])
    return summary


def _colmap_mean(colmap):
    if not colmap:
        return (0.7, 0.6, 0.45)
    a = np.mean(list(colmap.values()), axis=0)
    return (float(a[0]), float(a[1]), float(a[2]))


def _min_y_of(fit) -> float:
    from .fit import _lowest_y
    return _lowest_y(fit)


def object_to_documents(
    occ: np.ndarray,
    colmap: Optional[dict] = None,
    *,
    max_prims: int = 6,
    target_size: float = 1.0,
    ground: bool = False,
    oriented: bool = True,
):
    """Fit primitives to a reconstructed object and build ``.cgb`` documents.

    With ``oriented`` (default), primitives are PCA-aligned and inverse-
    transformed so they rotate to hug the part; otherwise they are axis-aligned.
    Returns ``(prim_doc, voxel_doc)`` in a shared, centred world frame.
    """
    import cgb

    from .fit import build_document, FitResult, _lowest_y

    pts = np.argwhere(occ).astype(float)
    if pts.size == 0:
        raise ValueError("empty occupancy")
    R = occ.shape[0]
    world = (pts + 0.5) / R - 0.5
    centroid = world.mean(0)
    ext = world.max(0) - world.min(0)
    scale = target_size / max(float(ext.max()), 1e-9)

    color = (0.72, 0.6, 0.42)
    if oriented:
        from .oriented_fit import fit_oriented_primitives
        fits = fit_oriented_primitives((world - centroid) * scale,
                                       max_prims=max_prims, color=color)
    else:
        from .primfit import decompose_occupancy
        from .multiview import _voxprim_to_fit
        fits = [_voxprim_to_fit(vp, centroid, scale, lambda cx, cy: color)
                for vp in decompose_occupancy(occ, max_prims=max_prims)]
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
