"""Voxel → varied-primitive abstraction tests (recognition.primfit).

Uses synthetic occupancy grids (perfect ground truth) so it needs only numpy +
scipy — no torch / SAM / model weights. Skips if scipy is unavailable.
"""

import numpy as np
import pytest

pytest.importorskip("scipy")

from recognition.primfit import decompose_occupancy, VoxPrim  # noqa: E402

R = 48


def _grid():
    return np.zeros((R, R, R), bool)


def _coords():
    g = np.arange(R) + 0.5
    return np.meshgrid(g, g, g, indexing="ij")


def test_box_to_single_cube():
    occ = _grid()
    occ[12:36, 8:40, 16:32] = True
    prims = decompose_occupancy(occ, min_iou=0.9)
    assert [p.type for p in prims] == ["cube"]
    assert prims[0].iou > 0.95


def test_sphere_to_single_sphere():
    X, Y, Z = _coords()
    occ = ((X - R / 2) ** 2 + (Y - R / 2) ** 2 + (Z - R / 2) ** 2) <= 14 ** 2
    prims = decompose_occupancy(occ, min_iou=0.9)
    assert [p.type for p in prims] == ["sphere"]
    assert prims[0].iou > 0.95


def test_cylinder_to_single_cylinder():
    X, Y, Z = _coords()
    occ = (((X - R / 2) ** 2 + (Z - R / 2) ** 2) <= 10 ** 2) & (Y >= 8) & (Y <= 40)
    prims = decompose_occupancy(occ, min_iou=0.9)
    assert [p.type for p in prims] == ["cylinder"]
    assert prims[0].axis == 1  # along Y
    assert prims[0].iou > 0.95


def test_cone_to_single_cone():
    X, Y, Z = _coords()
    t = np.clip((Y - 8) / 32, 0, 1)
    rr = (1 - t) * 12
    occ = (((X - R / 2) ** 2 + (Z - R / 2) ** 2) <= rr ** 2) & (Y >= 8) & (Y <= 40)
    prims = decompose_occupancy(occ, min_iou=0.9)
    assert [p.type for p in prims] == ["cone"]
    assert prims[0].iou > 0.9


def test_table_becomes_box_plus_round_legs():
    """A boxy top on four round legs → one cube + four cylinders (the goal)."""
    X, Y, Z = _coords()
    occ = _grid()
    occ[8:40, 34:40, 12:36] = True  # table top (box)
    for lx, lz in [(14, 18), (14, 32), (36, 18), (36, 32)]:
        occ |= (((X - lx) ** 2 + (Z - lz) ** 2) <= 3 ** 2) & (Y >= 8) & (Y <= 34)
    prims = decompose_occupancy(occ, min_iou=0.9, max_prims=16)
    types = sorted(p.type for p in prims)
    assert types.count("cube") == 1
    assert types.count("cylinder") == 4
    assert len(prims) == 5


def _rasterize(vp: VoxPrim) -> np.ndarray:
    from recognition.primfit import _raster_sphere, _raster_cylinder, _raster_cone
    c = (np.asarray(vp.center) + 0.5) * R
    if vp.type == "cube":
        s = np.asarray(vp.size) * R
        lo = np.clip(np.floor(c - s / 2).astype(int), 0, R)
        hi = np.clip(np.ceil(c + s / 2).astype(int), 0, R)
        m = np.zeros((R, R, R), bool)
        m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
        return m
    if vp.type == "sphere":
        return _raster_sphere((R, R, R), c, vp.radius * R)
    a, h = vp.axis, vp.height * R
    lo, hi = c[a] - h / 2, c[a] + h / 2
    if vp.type == "cylinder":
        return _raster_cylinder((R, R, R), c, vp.radius * R, a, lo, hi)
    return _raster_cone((R, R, R), c, vp.radius * R, a, lo, hi, vp.apex_high)


def test_partition_has_negligible_overlap():
    """Decomposition partitions voxels: total volume ≈ union (little redundancy)."""
    X, Y, Z = _coords()
    occ = _grid()
    occ[10:38, 26:40, 16:32] = True
    occ |= (((X - R / 2) ** 2 + (Z - R / 2) ** 2) <= 5 ** 2) & (Y >= 6) & (Y <= 26)
    prims = decompose_occupancy(occ, min_iou=0.9, max_prims=16)
    masks = [_rasterize(p) for p in prims]
    union = np.zeros((R, R, R), bool)
    summ = 0
    for m in masks:
        union |= m
        summ += int(m.sum())
    overlap = 1 - union.sum() / max(summ, 1)
    assert overlap < 0.05  # boxes-method overlap here is ~0.5+
