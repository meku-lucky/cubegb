"""Deformation (Priority 2) tests — taper along +Y.

Invariants:
* No ``deform`` key -> mesh identical to the undeformed primitive.
* ``taper`` scales the cross-section linearly along Y; a tapered cylinder has the
  exact frustum volume, and the -Y end is left unchanged (ratio 1).
* The deform validates and survives a save/load round-trip.
"""

import math

import pytest

import cgb
from bake.baker import primitive_to_mesh


def test_no_deform_is_identical():
    plain = primitive_to_mesh(cgb.cylinder("c", 0.5, 1.0, 24))
    again = primitive_to_mesh(cgb.cylinder("c", 0.5, 1.0, 24))
    assert plain.volume == again.volume
    assert len(plain.faces) == len(again.faces)


def test_taper_cylinder_is_frustum_volume():
    full = primitive_to_mesh(cgb.cylinder("c", 0.5, 1.0, 96))
    tap = primitive_to_mesh(cgb.cylinder("c", 0.5, 1.0, 96, deform=cgb.taper(0.5, 0.5)))
    # Frustum r1=0.5 (-Y end, unchanged), r2=0.25 (+Y end): V = pi*h/3*(r1^2+r1 r2+r2^2)
    expected = math.pi * 1.0 / 3.0 * (0.25 + 0.125 + 0.0625)
    assert tap.is_watertight
    assert tap.volume < full.volume
    assert math.isclose(tap.volume, expected, rel_tol=0.01)


def test_taper_keeps_minus_y_end_fixed():
    """The -Y end keeps scale 1; only the +Y end is scaled."""
    mesh = primitive_to_mesh(cgb.cube("b", [0.2, 1.0, 0.2], deform=cgb.taper(0.1, 0.1)))
    v = mesh.vertices
    bot = v[v[:, 1] < -0.4]
    top = v[v[:, 1] > 0.4]
    # bottom keeps full 0.2 extent; top shrinks to ~0.02.
    assert math.isclose(float(bot[:, 0].max() - bot[:, 0].min()), 0.2, abs_tol=1e-6)
    assert float(top[:, 0].max() - top[:, 0].min()) < 0.05


def test_blade_taper_preserves_thickness():
    """A blade tapers width (x) to a tip but keeps thickness (z)."""
    blade = primitive_to_mesh(cgb.cube("b", [0.1, 1.0, 0.03], deform=cgb.taper(0.05, 1.0)))
    v = blade.vertices
    top = v[v[:, 1] > 0.4]
    assert float(top[:, 0].max() - top[:, 0].min()) < 0.02  # width pinched
    assert math.isclose(float(top[:, 2].max() - top[:, 2].min()), 0.03, abs_tol=1e-6)


def test_taper_validates_and_roundtrips(tmp_path):
    doc = cgb.new_document()
    cgb.add_primitive(doc, cgb.cylinder("c", 0.4, 1.0, deform=cgb.taper(1.6, 1.6)))
    cgb.validate(doc)
    p = tmp_path / "t.cgb"
    cgb.save(doc, p)
    reloaded = cgb.load(p)
    assert reloaded["primitives"][0]["deform"]["taper"] == [1.6, 1.6]
    cgb.validate(reloaded)


def test_nonpositive_taper_rejected():
    doc = cgb.new_document()
    doc["primitives"].append(
        {
            "id": "c",
            "name": "c",
            "type": "cylinder",
            "transform": cgb.make_transform(),
            "params": {"radius": 0.5, "height": 1.0, "segments": 16},
            "deform": {"taper": [0.0, 1.0]},  # 0 is invalid (must be > 0)
            "parent": None,
        }
    )
    with pytest.raises(cgb.ValidationError):
        cgb.validate(doc)


def test_deformed_knight_sample_bakes():
    """The showcase sample loads, validates, and bakes watertight parts."""
    doc = cgb.load("samples/cat_knight_deformed.cgb")
    cgb.validate(doc)
    assert len(doc["primitives"]) == 31
    # The tapered sword blade is present and deformed.
    blade = next(p for p in doc["primitives"] if p["id"] == "sword_blade")
    assert "taper" in blade["deform"]
