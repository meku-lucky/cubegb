"""Voxel solid → varied volumetric primitives (cube / cylinder / cone / sphere).

The multi-view carver (:mod:`recognition.multiview`) produces a boolean
occupancy grid (the visual hull). The simple ``occupancy_to_boxes`` tiling
explains that solid with axis-aligned **cubes only**, which over-segments round
parts into stair-stepped boxes.

This module instead does a **top-down recursive shape abstraction**:

1. Split the solid into connected components.
2. For each region, rasterize candidate *volumetric* primitives — an
   axis-aligned box, a sphere, a cylinder along each world axis, and a cone
   along each axis/direction — and score each by **IoU against the region's
   voxels**. The box score is just the region's fill ratio, so a round leg
   (fill ≈ π/4) loses to a cylinder that matches it (IoU ≈ 1).
3. If the best primitive explains the region well enough (``min_iou``), emit it;
   otherwise split the region in half along its longest axis and recurse.

The result is a small set of varied, non-overlapping primitives (the regions
partition the voxels), which is what a blockout wants. Pure numpy + scipy.

Inspired by classic shape-abstraction / primitive-fitting work (e.g. Schnabel
et al., *Efficient RANSAC for Point-Cloud Shape Detection*, 2007, which likewise
detects planes/cylinders/cones/spheres and removes explained points).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class VoxPrim:
    """A fitted primitive in **normalised world units** (grid maps to [-0.5, 0.5]).

    - ``cube``: uses ``center`` + ``size`` (full extents, axis-aligned).
    - ``sphere``: ``center`` + ``radius``.
    - ``cylinder`` / ``cone``: ``center`` + ``radius`` + ``height`` + ``axis``
      (0/1/2 = world X/Y/Z). ``cone`` tapers to a point toward +axis if
      ``apex_high`` else toward -axis.
    """

    type: str
    center: np.ndarray
    iou: float
    size: Optional[np.ndarray] = None
    radius: Optional[float] = None
    height: Optional[float] = None
    axis: Optional[int] = None
    apex_high: bool = True


# --------------------------------------------------------------------------- #
# Rasterisers — boolean candidate masks over a local (nx,ny,nz) crop.
# Voxel centres are at integer index + 0.5.
# --------------------------------------------------------------------------- #
def _coords(shape):
    g = [np.arange(s) + 0.5 for s in shape]
    return np.meshgrid(g[0], g[1], g[2], indexing="ij")


def _raster_sphere(shape, center, radius):
    X, Y, Z = _coords(shape)
    d2 = (X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2
    return d2 <= radius * radius


def _raster_cylinder(shape, center, radius, axis, lo, hi):
    X, Y, Z = _coords(shape)
    C = [X, Y, Z]
    perp = [i for i in range(3) if i != axis]
    rad2 = (C[perp[0]] - center[perp[0]]) ** 2 + (C[perp[1]] - center[perp[1]]) ** 2
    along = C[axis]
    return (rad2 <= radius * radius) & (along >= lo) & (along <= hi)


def _raster_cone(shape, center, r_base, axis, lo, hi, apex_high):
    X, Y, Z = _coords(shape)
    C = [X, Y, Z]
    perp = [i for i in range(3) if i != axis]
    rad2 = (C[perp[0]] - center[perp[0]]) ** 2 + (C[perp[1]] - center[perp[1]]) ** 2
    along = C[axis]
    h = max(hi - lo, 1e-6)
    # t = 0 at base (full radius) -> 1 at apex (zero radius)
    t = (along - lo) / h if apex_high else (hi - along) / h
    r_at = np.clip(1.0 - t, 0.0, 1.0) * r_base
    return (rad2 <= r_at * r_at) & (along >= lo) & (along <= hi)


def _iou(a, b):
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return inter / union if union else 0.0


# --------------------------------------------------------------------------- #
# Per-region best-primitive fit
# --------------------------------------------------------------------------- #
def _fit_region(mask):
    """Fit the best volumetric primitive to a cropped boolean region ``mask``.

    Returns ``(type, geometry, iou)`` where geometry holds local-crop floats.
    Candidates are volume-matched to the region (so a single rasterise is a fair
    test), then scored by IoU.
    """
    idx = np.argwhere(mask)
    n = len(idx)
    shape = mask.shape
    centroid = idx.mean(0) + 0.5
    lo = idx.min(0).astype(float)
    hi = idx.max(0).astype(float) + 1.0  # exclusive-ish extent
    ext = hi - lo

    candidates = []

    # Box: fill ratio of the tight bbox.
    box_vol = float(np.prod(ext))
    candidates.append(("cube", {"center": (lo + hi) / 2, "size": ext.copy()},
                       n / box_vol if box_vol else 0.0))

    # Sphere: volume-matched radius.
    r_sph = (3.0 * n / (4.0 * np.pi)) ** (1.0 / 3.0)
    sph = _raster_sphere(shape, centroid, r_sph)
    candidates.append(("sphere", {"center": centroid.copy(), "radius": r_sph},
                       _iou(sph, mask)))

    # Cylinder along each axis: area-matched radius, full extent along axis.
    for a in range(3):
        a_lo, a_hi = float(idx[:, a].min()), float(idx[:, a].max()) + 1.0
        h = max(a_hi - a_lo, 1.0)
        r = np.sqrt(n / (np.pi * h))
        cyl = _raster_cylinder(shape, centroid, r, a, a_lo, a_hi)
        candidates.append(("cylinder",
                           {"center": centroid.copy(), "radius": r, "axis": a,
                            "lo": a_lo, "hi": a_hi},
                           _iou(cyl, mask)))

    # Dome / half-cylinder caps: a cylinder whose disk centre sits on a
    # perpendicular bbox edge, so only the top arc lies inside the region. This
    # captures rounded lids / barrel tops / domes (a half-disk cross-section)
    # that a centred cylinder (fill ≈ π/4 as a box) would otherwise miss.
    for a in range(3):
        a_lo, a_hi = float(idx[:, a].min()), float(idx[:, a].max()) + 1.0
        perp = [i for i in range(3) if i != a]
        for cap in perp:
            c_lo, c_hi = float(idx[:, cap].min()), float(idx[:, cap].max()) + 1.0
            r = c_hi - c_lo                      # cap height acts as the radius
            for edge in (c_lo, c_hi):            # flat side at low / high edge
                center2 = centroid.copy()
                center2[cap] = edge
                cyl = _raster_cylinder(shape, center2, r, a, a_lo, a_hi)
                candidates.append(("cylinder",
                                   {"center": center2, "radius": r, "axis": a,
                                    "lo": a_lo, "hi": a_hi},
                                   _iou(cyl, mask)))

    # Cone along each axis and direction: volume-matched base radius.
    for a in range(3):
        a_lo, a_hi = float(idx[:, a].min()), float(idx[:, a].max()) + 1.0
        h = max(a_hi - a_lo, 1.0)
        r0 = np.sqrt(3.0 * n / (np.pi * h))
        for apex_high in (True, False):
            cone = _raster_cone(shape, centroid, r0, a, a_lo, a_hi, apex_high)
            candidates.append(("cone",
                               {"center": centroid.copy(), "radius": r0, "axis": a,
                                "lo": a_lo, "hi": a_hi, "apex_high": apex_high},
                               _iou(cone, mask)))

    best = max(candidates, key=lambda c: c[2])
    return best


# --------------------------------------------------------------------------- #
# Top-down recursive decomposition
# --------------------------------------------------------------------------- #
def decompose_occupancy(
    occ: np.ndarray,
    *,
    min_iou: float = 0.88,
    min_vox: int = 16,
    max_prims: int = 24,
    max_depth: int = 7,
) -> list[VoxPrim]:
    """Decompose a boolean occupancy grid into varied volumetric primitives.

    Returns a list of :class:`VoxPrim` in normalised world units (grid → cube
    centred in [-0.5, 0.5]).
    """
    try:
        from scipy import ndimage
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("primfit needs scipy (scipy.ndimage).") from exc

    R = occ.shape[0]
    labels, n_comp = ndimage.label(occ)
    out: list[VoxPrim] = []

    # Work queue of (global_lo (3,), local boolean mask, depth), largest first.
    stack = []
    for ci in range(1, n_comp + 1):
        comp = labels == ci
        gidx = np.argwhere(comp)
        glo = gidx.min(0)
        ghi = gidx.max(0) + 1
        sub = comp[glo[0]:ghi[0], glo[1]:ghi[1], glo[2]:ghi[2]]
        stack.append((glo.astype(float), sub, 0))

    # Process largest regions first so the primitive budget goes to big parts.
    while stack and len(out) < max_prims:
        stack.sort(key=lambda s: -int(s[1].sum()))
        glo, mask, depth = stack.pop(0)
        n = int(mask.sum())
        if n == 0:
            continue
        ptype, geom, iou = _fit_region(mask)

        budget_left = max_prims - len(out) - len(stack)
        accept = (iou >= min_iou) or (n <= min_vox) or (depth >= max_depth) \
            or budget_left <= 1

        if not accept:
            split = _best_split(mask)
            if split is not None:
                lo_mask, hi_mask = split
                stack.append((glo.copy(), lo_mask, depth + 1))
                stack.append((glo.copy(), hi_mask, depth + 1))
                continue
            # No usable split → accept the region as-is.

        out.append(_to_voxprim(ptype, geom, glo, R, iou))

    return out


def _split_at(mask, axis, cut):
    """Split ``mask`` into (low, high) halves at index ``cut`` along ``axis``."""
    lo_mask = mask.copy()
    hi_mask = mask.copy()
    sl_lo = [slice(None)] * 3
    sl_hi = [slice(None)] * 3
    sl_lo[axis] = slice(cut, None)
    sl_hi[axis] = slice(0, cut)
    lo_mask[tuple(sl_lo)] = False
    hi_mask[tuple(sl_hi)] = False
    return lo_mask, hi_mask


def _best_split(mask, *, min_part: int = 8):
    """Choose a cut plane that best separates the region into two parts.

    For each axis we consider a few candidate cut positions — the interior
    minimum of the cross-section profile (a *neck*), the steepest profile step (a
    part *junction*), and the median (balanced fallback) — and pick the cut whose
    two halves are individually best explained by a single primitive (max total
    ``iou * size``). This 1-step lookahead separates, e.g., a round leg from a
    boxy body instead of slicing through the middle.
    """
    idx = np.argwhere(mask)
    best = None
    best_score = -1.0
    for axis in range(3):
        a_lo, a_hi = int(idx[:, axis].min()), int(idx[:, axis].max())
        L = a_hi - a_lo
        if L < 2:
            continue
        # Cross-section area per slice along this axis.
        other = tuple(i for i in range(3) if i != axis)
        prof = mask.sum(axis=other)  # length = mask.shape[axis]
        interior = range(a_lo + 1, a_hi + 1)  # valid cut positions
        margin = max(1, int(0.12 * L))
        inner = [c for c in interior if a_lo + margin <= c <= a_hi - margin] or list(interior)

        cands = set()
        # neck: smallest cross-section in the interior
        cands.add(min(inner, key=lambda c: prof[c]))
        # junction: steepest step in the profile
        grad = np.abs(np.diff(prof.astype(float)))
        if grad.size:
            cands.add(int(np.clip(np.argmax(grad) + 1, a_lo + 1, a_hi)))
        # balanced fallback
        cands.add(int(np.median(idx[:, axis])) or (a_lo + L // 2))

        for cut in cands:
            if not (a_lo < cut <= a_hi):
                continue
            lo_mask, hi_mask = _split_at(mask, axis, cut)
            n_lo, n_hi = int(lo_mask.sum()), int(hi_mask.sum())
            if n_lo < min_part or n_hi < min_part:
                continue
            _, _, iou_lo = _fit_region(lo_mask)
            _, _, iou_hi = _fit_region(hi_mask)
            score = iou_lo * n_lo + iou_hi * n_hi
            if score > best_score:
                best_score = score
                best = (lo_mask, hi_mask)
    return best


def _to_voxprim(ptype, geom, glo, R, iou) -> VoxPrim:
    """Convert a local-crop fit into a normalised-world :class:`VoxPrim`."""
    def world(v):  # local-crop voxel coord -> normalised world [-0.5,0.5]
        return (np.asarray(v, dtype=float) + glo) / R - 0.5

    if ptype == "cube":
        center = world(geom["center"])
        size = np.asarray(geom["size"], dtype=float) / R
        return VoxPrim("cube", center, iou, size=size)
    if ptype == "sphere":
        center = world(geom["center"])
        return VoxPrim("sphere", center, iou, radius=geom["radius"] / R)
    if ptype in ("cylinder", "cone"):
        # CGB cylinder/cone are centred on their axis midpoint; the region's
        # volume centroid is biased toward a cone's base, so use the geometric
        # mid of the axis extent for the along-axis coordinate.
        lc = np.asarray(geom["center"], dtype=float).copy()
        lc[geom["axis"]] = (geom["lo"] + geom["hi"]) / 2.0
        center = world(lc)
        kw = dict(radius=geom["radius"] / R, height=(geom["hi"] - geom["lo"]) / R,
                  axis=geom["axis"])
        if ptype == "cylinder":
            return VoxPrim("cylinder", center, iou, **kw)
        return VoxPrim("cone", center, iou, apex_high=geom["apex_high"], **kw)
    raise ValueError(ptype)
