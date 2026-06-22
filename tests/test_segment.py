"""Unit tests for SAM proposal post-processing (recognition.segment).

Exercises the pure-numpy mask helpers and the ``_select_parts`` pipeline with
hand-built proposals — no torch / SAM checkpoint required.
"""
import numpy as np
import pytest

from recognition.segment import (
    Segmenter,
    _border_touch_frac,
    _intersection_over_min,
    _iou,
)

H = W = 100


def _rect(x0, y0, x1, y1):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


def _proposal(mask, iou=0.95):
    return {
        "segmentation": mask,
        "area": int(mask.sum()),
        "predicted_iou": iou,
        "bbox": [0, 0, 1, 1],
    }


def test_border_touch_frac():
    full = np.ones((H, W), bool)
    assert _border_touch_frac(full) == pytest.approx(1.0)
    center = _rect(40, 40, 60, 60)
    assert _border_touch_frac(center) == pytest.approx(0.0)


def test_iou_and_intersection_over_min():
    a = _rect(0, 0, 50, 50)
    b = _rect(0, 0, 50, 50)
    assert _iou(a, b) == pytest.approx(1.0)
    small = _rect(0, 0, 25, 50)          # fully inside a
    assert _intersection_over_min(a, small) == pytest.approx(1.0)
    assert _iou(a, small) < 1.0


def _select(proposals, **kw):
    defaults = dict(
        image_shape=(H, W), max_masks=12, min_area_frac=0.005,
        max_area_frac=0.95, border_bg_frac=0.55, nms_overlap=0.5,
        part_area_frac=0.02,
    )
    defaults.update(kw)
    return Segmenter._select_parts(proposals, **defaults)


def test_background_plate_dropped():
    bg = np.ones((H, W), bool)            # hugs the whole border
    obj = _rect(30, 30, 70, 70)
    kept = _select([_proposal(bg), _proposal(obj)])
    masks = [k.mask for k in kept]
    assert len(kept) == 1
    assert masks[0][50, 50]               # the object survived
    assert not masks[0][0, 0]             # not the background plate


def test_duplicates_suppressed():
    obj = _rect(20, 20, 80, 80)
    kept = _select([_proposal(obj, 0.95), _proposal(obj.copy(), 0.90)])
    assert len(kept) == 1


def test_redundant_parent_dropped_in_favour_of_parts():
    # A whole-object mask tiled by two sizeable, non-overlapping parts.
    left = _rect(20, 20, 48, 80)
    right = _rect(52, 20, 80, 80)
    whole = _rect(20, 20, 80, 80)
    kept = _select([_proposal(whole), _proposal(left), _proposal(right)])
    # The parent is dropped; the two parts are kept.
    assert len(kept) == 2
    areas = sorted(k.area for k in kept)
    assert areas[0] < whole.sum() and areas[1] < whole.sum()


def test_nested_fragment_skipped_by_nms():
    obj = _rect(20, 20, 80, 80)
    fragment = _rect(30, 30, 50, 50)      # nested inside obj, no sibling parts
    kept = _select([_proposal(obj), _proposal(fragment)])
    assert len(kept) == 1
    assert kept[0].area == int(obj.sum())


def test_max_masks_cap():
    objs = [_proposal(_rect(i * 8 + 2, 2, i * 8 + 8, 90)) for i in range(10)]
    kept = _select(objs, max_masks=4)
    assert len(kept) == 4
