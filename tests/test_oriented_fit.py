"""Oriented (OBB) primitive fitting tests (recognition.oriented_fit).

Synthetic point clouds (no SAM / weights). Skips without scipy.
"""

import numpy as np
import pytest

pytest.importorskip("scipy")

from recognition.oriented_fit import fit_oriented_primitives


def _rot_y(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])


def test_tilted_cylinder_recovers_axis_and_rotation():
    """A cylinder tilted 40° about Y comes back as a cylinder carrying rotation."""
    rng = np.random.RandomState(0)
    p = rng.rand(12000, 3)
    p = p[((p[:, 0] - 0.5) ** 2 + (p[:, 1] - 0.5) ** 2) <= 0.25]
    p = (p - 0.5) * np.array([0.6, 0.6, 2.5])          # long along Z
    p = p @ _rot_y(40).T + np.array([2.0, 1.0, -1.0])

    fits = fit_oriented_primitives(p, res=44, max_prims=2)
    assert fits
    cyl = max(fits, key=lambda f: f.params.get("height", 0) if f.prim_type == "cylinder" else 0)
    assert cyl.prim_type == "cylinder"
    assert cyl.params["height"] > 1.5                  # captured the long axis
    assert max(abs(r) for r in cyl.rotation_euler) > 0.2  # actually rotated


def test_empty_input_is_safe():
    assert fit_oriented_primitives(np.zeros((2, 3))) == []


def test_primitive_lands_near_the_cloud():
    rng = np.random.RandomState(1)
    p = (rng.rand(6000, 3) - 0.5) * np.array([1.5, 1.0, 0.6]) + np.array([5.0, 2.0, 3.0])
    fits = fit_oriented_primitives(p, res=40, max_prims=3)
    assert fits
    centre = np.mean([f.position for f in fits], axis=0)
    assert np.allclose(centre, [5.0, 2.0, 3.0], atol=0.6)
