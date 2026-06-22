"""Unit tests for the 2D silhouette type classifier (recognition.shape2d).

These build clean synthetic silhouette masks (no SAM / torch needed) and assert
the classifier names the right primitive. cv2 is required, so the module is
skipped when the recognition extras are not installed.
"""
import numpy as np
import pytest

pytest.importorskip("cv2")
import cv2  # noqa: E402

from recognition import shape2d  # noqa: E402

H = W = 256


def _disk(r=90):
    m = np.zeros((H, W), np.uint8)
    cv2.circle(m, (W // 2, H // 2), r, 1, -1)
    return m.astype(bool)


def _square(s=150):
    m = np.zeros((H, W), np.uint8)
    x0 = (W - s) // 2
    cv2.rectangle(m, (x0, x0), (x0 + s, x0 + s), 1, -1)
    return m.astype(bool)


def _capsule(w=70, h=180):
    """A vertical rounded rectangle (cylinder side-view silhouette)."""
    m = np.zeros((H, W), np.uint8)
    cx, cy = W // 2, H // 2
    cv2.rectangle(m, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), 1, -1)
    cv2.circle(m, (cx, cy - h // 2), w // 2, 1, -1)
    cv2.circle(m, (cx, cy + h // 2), w // 2, 1, -1)
    return m.astype(bool)


def _triangle():
    m = np.zeros((H, W), np.uint8)
    pts = np.array([[W // 2, 30], [40, H - 30], [W - 40, H - 30]], np.int32)
    cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def test_circle_is_sphere():
    assert shape2d.classify(_disk())["type"] == "sphere"


def test_square_is_cube():
    assert shape2d.classify(_square())["type"] == "cube"


def test_capsule_is_cylinder():
    assert shape2d.classify(_capsule())["type"] == "cylinder"


def test_triangle_is_cone():
    assert shape2d.classify(_triangle())["type"] == "cone"


def test_prior_is_normalised_distribution():
    res = shape2d.classify(_disk())
    prior = res["prior"]
    assert set(prior) == set(shape2d.TYPES)
    assert abs(sum(prior.values()) - 1.0) < 1e-6
    assert all(0.0 <= v <= 1.0 for v in prior.values())
    assert 0.0 <= res["confidence"] <= 1.0


def test_degenerate_mask_falls_back_to_flat_prior():
    empty = np.zeros((H, W), bool)
    res = shape2d.classify(empty)
    assert res["type"] == "cube"  # safe default
    assert abs(sum(res["prior"].values()) - 1.0) < 1e-6
