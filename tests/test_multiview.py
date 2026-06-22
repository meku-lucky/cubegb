"""Unit tests for multi-view space carving (recognition.multiview).

Deterministic and SAM-free: synthetic sheets/silhouettes only.
"""
import numpy as np
import pytest

from recognition.multiview import (
    SHEET_LAYOUT,
    ViewCell,
    carve_occupancy,
    occupancy_to_boxes,
    occupancy_to_world_points,
    split_sheet,
    _project,
)


def _sheet_with_blank_bottom_right(n=240):
    """2x2 sheet: TL/TR/BL have a centred square, BR is blank plate."""
    img = np.full((n, n, 3), 210, np.uint8)  # grey plate
    h = n // 2
    for (r, c) in [(0, 0), (0, 1), (1, 0)]:
        y0, x0 = r * h + h // 4, c * h + h // 4
        img[y0:y0 + h // 2, x0:x0 + h // 2] = (40, 60, 200)  # filled square
    return img


def test_split_sheet_names_and_blank():
    cells = split_sheet(_sheet_with_blank_bottom_right())
    names = [c.name for c in cells]
    assert names == [SHEET_LAYOUT[k] for k in SHEET_LAYOUT]
    by = {c.name: c for c in cells}
    assert by["top"].blank          # bottom-right left empty
    assert not by["front"].blank
    assert by["front"].silhouette.mean() > 0.05


def _square_cell(name, frac=0.5, n=64):
    sil = np.zeros((n, n), bool)
    lo, hi = int(n * (0.5 - frac / 2)), int(n * (0.5 + frac / 2))
    sil[lo:hi, lo:hi] = True
    return ViewCell(name=name, rgb=np.zeros((n, n, 3), np.uint8), silhouette=sil, blank=False)


def test_carve_box_from_front_and_side():
    cells = [_square_cell("front"), _square_cell("side")]
    occ = carve_occupancy(cells, res=64)
    assert occ.any()
    ix, iy, iz = np.nonzero(occ)
    # a 0.5-wide square in front (x,y) and side (z,y) -> ~0.5 box in all axes
    for lo, hi in [(ix.min(), ix.max()), (iy.min(), iy.max()), (iz.min(), iz.max())]:
        assert 0.4 < (hi - lo + 1) / 64 < 0.65


def test_blank_views_add_no_constraint():
    # Only a front square; everything behind it should remain occupied (a bar).
    occ = carve_occupancy([_square_cell("front")], res=32)
    # full depth in z (no side/top to carve it)
    iz = np.nonzero(occ)[2]
    assert (iz.max() - iz.min() + 1) == 32


def test_no_views_gives_empty():
    occ = carve_occupancy([ViewCell("front", np.zeros((8, 8, 3), np.uint8),
                                    np.zeros((8, 8), bool), blank=True)], res=16)
    assert not occ.any()


def test_project_front_back_mirror():
    x = np.array([0.25]); y = np.array([0.5]); z = np.array([0.5])
    uf, vf = _project("front", x, y, z)
    ub, vb = _project("back", x, y, z)
    assert uf[0] == pytest.approx(0.25)
    assert ub[0] == pytest.approx(0.75)        # horizontally mirrored
    assert vf[0] == pytest.approx(vb[0])        # same height


def test_boxes_cover_a_solid_block():
    occ = np.zeros((32, 32, 32), bool)
    occ[8:24, 8:24, 8:24] = True
    boxes = occupancy_to_boxes(occ, max_boxes=8)
    assert 1 <= len(boxes) <= 8
    # one box should essentially cover the whole 0.5-cube block
    centre, size = boxes[0]
    assert abs(centre).max() < 0.05               # centred block -> centred box
    assert min(size) > 0.45                        # ~16/32 in each axis


def test_boxes_tile_an_l_shape():
    occ = np.zeros((24, 24, 24), bool)
    occ[4:20, 4:10, 4:20] = True                   # one arm
    occ[4:10, 4:20, 4:20] = True                   # perpendicular arm
    boxes = occupancy_to_boxes(occ, max_boxes=8, min_cover=0.95)
    assert len(boxes) >= 2                          # L needs >1 axis-aligned box
    assert all(min(s) > 0 for _, s in boxes)


def test_boxes_empty_occupancy():
    assert occupancy_to_boxes(np.zeros((8, 8, 8), bool)) == []


def test_occupancy_points_centered():
    occ = np.zeros((16, 16, 16), bool)
    occ[4:12, 4:12, 4:12] = True
    pts = occupancy_to_world_points(occ)
    assert pts.shape[1] == 3
    assert abs(pts.mean(axis=0)).max() < 1e-9   # symmetric block -> centred at 0
