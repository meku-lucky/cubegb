"""Boolean / CSG (Priority 3) tests — declarative ``operations`` baked once.

The real mesh boolean needs the manifold3d backend; skip cleanly if it is not
installed (it is a documented requirement in requirements.txt).
"""

import math

import pytest

pytest.importorskip("manifold3d", reason="boolean backend (manifold3d) not installed")

import cgb
from bake.baker import bake_scene


def _block_with_hole():
    doc = cgb.new_document()
    cgb.add_primitive(doc, cgb.cube("block", [1.0, 1.0, 1.0], color=(0.6, 0.6, 0.65)))
    cgb.add_primitive(doc, cgb.cylinder("hole", 0.25, 1.4, color=(0.8, 0.2, 0.2)))
    cgb.add_operation(doc, cgb.difference("block", "hole"))
    return doc


def test_difference_drills_a_hole():
    doc = _block_with_hole()
    cgb.validate(doc)
    scene = bake_scene(doc)
    # the cutter is consumed — only the target remains as geometry
    assert set(scene.geometry.keys()) == {"block"}
    block = scene.geometry["block"]
    assert block.is_watertight
    expected = 1.0 - math.pi * 0.25 ** 2 * 1.0
    assert block.volume < 1.0
    assert math.isclose(block.volume, expected, rel_tol=0.02)


def test_union_merges_volume():
    doc = cgb.new_document()
    cgb.add_primitive(doc, cgb.cube("a", [1.0, 1.0, 1.0]))
    cgb.add_primitive(doc, cgb.cube("b", [1.0, 1.0, 1.0], transform=cgb.make_transform([0.5, 0, 0])))
    cgb.add_operation(doc, cgb.union("a", "b"))
    cgb.validate(doc)
    scene = bake_scene(doc)
    assert set(scene.geometry.keys()) == {"a"}
    merged = scene.geometry["a"]
    assert merged.is_watertight
    # union of two unit cubes overlapping by half = 1.5 volume
    assert math.isclose(merged.volume, 1.5, rel_tol=0.02)


def test_intersection_keeps_overlap():
    doc = cgb.new_document()
    cgb.add_primitive(doc, cgb.cube("a", [1.0, 1.0, 1.0]))
    cgb.add_primitive(doc, cgb.cube("b", [1.0, 1.0, 1.0], transform=cgb.make_transform([0.5, 0, 0])))
    cgb.add_operation(doc, cgb.intersection("a", "b"))
    cgb.validate(doc)
    merged = bake_scene(doc).geometry["a"]
    assert merged.is_watertight
    assert math.isclose(merged.volume, 0.5, rel_tol=0.02)


def test_multi_cutter_difference():
    doc = cgb.new_document()
    cgb.add_primitive(doc, cgb.cube("block", [1.0, 1.0, 1.0]))
    cgb.add_primitive(doc, cgb.cylinder("h1", 0.15, 1.4, transform=cgb.make_transform([0.3, 0, 0])))
    cgb.add_primitive(doc, cgb.cylinder("h2", 0.15, 1.4, transform=cgb.make_transform([-0.3, 0, 0])))
    cgb.add_operation(doc, cgb.difference("block", "h1", "h2"))
    cgb.validate(doc)
    scene = bake_scene(doc)
    assert set(scene.geometry.keys()) == {"block"}
    assert scene.geometry["block"].is_watertight
    assert scene.geometry["block"].volume < 1.0


def test_unknown_operand_rejected():
    doc = cgb.new_document()
    cgb.add_primitive(doc, cgb.cube("a", [1, 1, 1]))
    cgb.add_operation(doc, cgb.difference("a", "ghost"))
    with pytest.raises(cgb.ValidationError):
        cgb.validate(doc)


def test_target_cannot_be_its_own_operand():
    doc = cgb.new_document()
    cgb.add_primitive(doc, cgb.cube("a", [1, 1, 1]))
    cgb.add_operation(doc, cgb.difference("a", "a"))
    with pytest.raises(cgb.ValidationError):
        cgb.validate(doc)


def test_keyhole_lock_sample_bakes_to_a_hole():
    doc = cgb.load("samples/keyhole_lock.cgb")
    cgb.validate(doc)
    scene = bake_scene(doc)
    # 5 primitives, 4 of them cutters -> only the plate is emitted
    assert set(scene.geometry.keys()) == {"plate"}
    assert scene.geometry["plate"].is_watertight
