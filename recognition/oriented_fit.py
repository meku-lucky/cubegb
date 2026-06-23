"""Oriented (OBB-style) per-object primitive fitting.

Axis-aligned fitting leaves tilted parts (a held sword, an arm) approximated by
upright boxes/cylinders — "stacked, not fitted". This module instead:

1. **PCA-aligns** an object's voxels to their principal axes (an oriented
   bounding box frame), so the part is axis-aligned in that frame.
2. Fits/decomposes primitives there (where axis-aligned fitting is tight), and is
   free to **combine several primitives** for one part.
3. **Inverse-transforms** each fitted primitive back by the PCA rotation, so it
   carries the part's real orientation (a cube comes back *rotated* to hug a
   tilted part).

Operates in metric world coordinates and returns ``fit.FitResult`` objects ready
for ``build_document``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _pca_rotation(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(centroid, R)`` where ``R``'s columns are the principal axes.

    ``R`` is a proper rotation (right-handed, det +1); ``world = R @ aligned + c``.
    """
    c = points.mean(0)
    q = points - c
    cov = q.T @ q
    w, v = np.linalg.eigh(cov)            # ascending eigenvalues
    R = v[:, np.argsort(w)[::-1]]         # columns = axes, largest variance first
    if np.linalg.det(R) < 0:
        R[:, 2] = -R[:, 2]                # make right-handed
    return c, R


def _voxelize(A: np.ndarray, res: int, fill: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Rasterise aligned points ``A`` into a cubic grid (uniform scale).

    Returns ``(occ, bbox_center, span)`` where a normalised primitive coordinate
    ``n`` ∈ [-0.5, 0.5] maps back to aligned-metric as ``bbox_center + n * span``.
    """
    b = (A.min(0) + A.max(0)) / 2.0
    span = float((A.max(0) - b).max() * 2.0) / fill or 1e-9
    n = (A - b) / span                     # ~[-0.5, 0.5]
    idx = np.clip(((n + 0.5) * res).astype(int), 0, res - 1)
    occ = np.zeros((res, res, res), dtype=bool)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    try:                                   # solidify the thin shell left by
        from scipy import ndimage           # re-voxelising rotated points
        occ = ndimage.binary_closing(occ, iterations=1)
        occ = ndimage.binary_fill_holes(occ)
    except Exception:  # pragma: no cover
        pass
    return occ, b, span


def fit_oriented_primitives(
    points: np.ndarray,
    *,
    res: int = 48,
    max_prims: int = 6,
    fill: float = 0.85,
    color=(0.7, 0.6, 0.45),
    min_iou: float = 0.9,
) -> list:
    """Fit oriented primitives to an object's voxel-centre cloud (metric world).

    Returns a list of :class:`recognition.fit.FitResult` carrying the object's
    PCA orientation (so cubes/cylinders are rotated to hug tilted parts).
    """
    from .primfit import decompose_occupancy
    from .fit import FitResult, _euler_xyz_to_matrix, _rotation_to_euler_xyz
    from .multiview import _CYL_EULER, _CONE_EULER

    P = np.asarray(points, dtype=np.float64)
    if len(P) < 4:
        return []
    c, R = _pca_rotation(P)
    A = (P - c) @ R                        # aligned to principal axes
    occ, b, span = _voxelize(A, res, fill)
    if not occ.any():
        return []

    fits = []
    for vp in decompose_occupancy(occ, max_prims=max_prims, min_iou=min_iou):
        # primitive geometry in the ALIGNED metric frame
        a_center = b + np.asarray(vp.center) * span
        if vp.type == "cube":
            Ma = np.eye(3)
            params = {"size": [float(s) for s in np.asarray(vp.size) * span]}
        elif vp.type == "sphere":
            Ma = np.eye(3)
            params = {"radius": float(vp.radius * span), "segments": 16}
        elif vp.type == "cylinder":
            Ma = _euler_xyz_to_matrix(_CYL_EULER[vp.axis])
            params = {"radius": float(vp.radius * span),
                      "height": float(vp.height * span), "segments": 16}
        else:  # cone
            Ma = _euler_xyz_to_matrix(_CONE_EULER[(vp.axis, vp.apex_high)])
            params = {"radius": float(vp.radius * span),
                      "height": float(vp.height * span), "segments": 16}

        # inverse-transform back to world: compose the PCA rotation
        Mw = R @ Ma
        pw = R @ a_center + c
        euler = _rotation_to_euler_xyz(Mw)
        fits.append(FitResult(vp.type, (float(pw[0]), float(pw[1]), float(pw[2])),
                              (float(euler[0]), float(euler[1]), float(euler[2])),
                              params, float(vp.iou), color))
    return fits
