"""Unit tests for type selection in recognition.fit.fit_primitive.

Pure numpy + the core cgb package — no SAM / depth model needed. Verifies that
the 2D shape prior can override the raw lowest-residual pick (the mechanism that
lets a side-on cylinder, which fits a box just as well in noisy monocular depth,
still be typed a cylinder).
"""
import numpy as np

from recognition.fit import fit_primitive


def _box_cloud(n=2000, sx=1.0, sy=1.0, sz=0.4, seed=0):
    """A filled axis-aligned box point cloud (ambiguous: fits cube or cylinder)."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-0.5, 0.5, size=(n, 3)) * np.array([sx, sy, sz])
    return pts


def test_no_prior_picks_lowest_residual_cube():
    pts = _box_cloud()
    fit = fit_primitive(pts)
    assert fit is not None
    assert fit.prim_type == "cube"


def test_strong_cylinder_prior_overrides_to_cylinder():
    pts = _box_cloud()
    prior = {"cube": 0.02, "cylinder": 0.94, "cone": 0.02, "sphere": 0.02}
    fit = fit_primitive(pts, shape_prior=prior, prior_weight=0.6)
    assert fit is not None
    assert fit.prim_type == "cylinder"


def test_prior_weight_zero_ignores_prior():
    pts = _box_cloud()
    prior = {"cube": 0.02, "cylinder": 0.94, "cone": 0.02, "sphere": 0.02}
    fit = fit_primitive(pts, shape_prior=prior, prior_weight=0.0)
    assert fit is not None
    assert fit.prim_type == "cube"  # residual wins when prior weight is 0


def test_too_few_points_returns_none():
    assert fit_primitive(np.zeros((5, 3))) is None
