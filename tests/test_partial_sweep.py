"""Partial-sweep (Deformation & Boolean, Priority 1) tests.

Covers the full vertical slice: builders -> validation -> baked mesh geometry.
The key invariants:

* A full (default) sweep is byte-for-byte unchanged from the legacy path.
* A partial sweep produces the right *fraction* of the full solid's volume.
* ``sweep_caps`` controls whether the wedge is a closed (watertight) solid.
* Validation rejects an empty / inverted arc.
"""

import math

import pytest

import cgb
from bake.baker import primitive_to_mesh


# --------------------------------------------------------------------------- #
# Builders + validation
# --------------------------------------------------------------------------- #
def test_full_sweep_omits_params_and_validates():
    """No sweep args -> no sweep keys in params (backward compatible)."""
    cyl = cgb.cylinder("c", 0.5, 1.0)
    assert "sweep_start" not in cyl["params"]
    assert "sweep_end" not in cyl["params"]
    doc = cgb.new_document()
    cgb.add_primitive(doc, cyl)
    cgb.validate(doc)


def test_partial_sweep_params_roundtrip():
    cyl = cgb.cylinder("c", 0.5, 1.0, sweep_start=0, sweep_end=180, sweep_caps=True)
    assert cyl["params"]["sweep_start"] == 0.0
    assert cyl["params"]["sweep_end"] == 180.0
    assert cyl["params"]["sweep_caps"] is True
    doc = cgb.new_document()
    cgb.add_primitive(doc, cyl)
    cgb.validate(doc)  # schema + semantics


def test_inverted_sweep_rejected():
    doc = cgb.new_document()
    cgb.add_primitive(doc, cgb.cylinder("c", 0.5, 1.0, sweep_start=200, sweep_end=100))
    with pytest.raises(cgb.ValidationError):
        cgb.validate(doc)


def test_out_of_range_sweep_rejected():
    doc = cgb.new_document()
    cgb.add_primitive(doc, cgb.cylinder("c", 0.5, 1.0, sweep_start=0, sweep_end=400))
    with pytest.raises(cgb.ValidationError):
        cgb.validate(doc)


# --------------------------------------------------------------------------- #
# Baked geometry
# --------------------------------------------------------------------------- #
def test_full_cylinder_geometry_unchanged():
    """A 0..360 sweep falls back to the legacy trimesh path verbatim."""
    plain = primitive_to_mesh(cgb.cylinder("c", 0.5, 1.0, 24))
    swept_full = primitive_to_mesh(
        cgb.cylinder("c", 0.5, 1.0, 24, sweep_start=0, sweep_end=360)
    )
    assert len(plain.faces) == len(swept_full.faces)
    assert math.isclose(plain.volume, swept_full.volume, rel_tol=1e-9)


def test_half_cylinder_is_half_volume():
    full = primitive_to_mesh(cgb.cylinder("c", 0.5, 1.0, 64))
    half = primitive_to_mesh(
        cgb.cylinder("c", 0.5, 1.0, 64, sweep_start=0, sweep_end=180, sweep_caps=True)
    )
    assert half.is_watertight
    assert math.isclose(half.volume, full.volume / 2.0, rel_tol=0.02)


def test_quarter_cone_is_quarter_volume():
    full = primitive_to_mesh(cgb.cone("c", 0.5, 1.0, 64))
    quarter = primitive_to_mesh(
        cgb.cone("c", 0.5, 1.0, 64, sweep_start=0, sweep_end=90, sweep_caps=True)
    )
    assert quarter.is_watertight
    assert math.isclose(quarter.volume, full.volume / 4.0, rel_tol=0.03)


def test_uncapped_wedge_is_open_shell():
    capped = primitive_to_mesh(
        cgb.cylinder("c", 0.5, 1.0, 24, sweep_start=0, sweep_end=120, sweep_caps=True)
    )
    open_ = primitive_to_mesh(
        cgb.cylinder("c", 0.5, 1.0, 24, sweep_start=0, sweep_end=120, sweep_caps=False)
    )
    assert capped.is_watertight
    assert not open_.is_watertight


def test_sweep_arc_direction_matches_threejs_convention():
    """x = r*sin(theta), z = r*cos(theta): a 0..90 arc lives in the +x/+z quadrant."""
    mesh = primitive_to_mesh(
        cgb.cylinder("c", 1.0, 0.2, 32, sweep_start=0, sweep_end=90, sweep_caps=False)
    )
    v = mesh.vertices
    # Every rim point of the arc sits in x>=-eps and z>=-eps (the 0..90 quadrant).
    on_rim = (v[:, 0] ** 2 + v[:, 2] ** 2) > 0.5  # radius ~1, excludes axis verts
    assert (v[on_rim, 0] >= -1e-6).all()
    assert (v[on_rim, 2] >= -1e-6).all()
