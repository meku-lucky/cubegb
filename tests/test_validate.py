"""Schema and semantic validation tests (Phase 0 checkpoint)."""

from pathlib import Path

import pytest

import cgb
from cgb.validate import ValidationError

SAMPLES = sorted((Path(__file__).resolve().parents[1] / "samples").glob("*.cgb"))


@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.name)
def test_samples_validate(path):
    cgb.validate(cgb.load(path))


def test_rejects_bad_format():
    doc = cgb.new_document()
    doc["format"] = "not-cgb"
    with pytest.raises(ValidationError):
        cgb.validate(doc)


def test_rejects_unknown_primitive_type():
    doc = cgb.new_document()
    doc["primitives"].append({
        "id": "x", "type": "torus",
        "transform": cgb.make_transform(),
        "params": {"radius": 1.0}, "parent": None,
    })
    with pytest.raises(ValidationError):
        cgb.validate(doc)


def test_rejects_wrong_params_for_type():
    doc = cgb.new_document()
    # cube requires `size`, not `radius`.
    doc["primitives"].append({
        "id": "x", "type": "cube",
        "transform": cgb.make_transform(),
        "params": {"radius": 1.0}, "parent": None,
    })
    with pytest.raises(ValidationError):
        cgb.validate(doc)


def test_rejects_duplicate_ids():
    doc = cgb.new_document()
    cgb.add_primitive(doc, cgb.cube("dup", size=[1, 1, 1]))
    cgb.add_primitive(doc, cgb.cube("dup", size=[1, 1, 1]))
    with pytest.raises(ValidationError, match="Duplicate"):
        cgb.validate(doc)


def test_rejects_dangling_parent():
    doc = cgb.new_document()
    cgb.add_primitive(doc, cgb.cube("a", size=[1, 1, 1], parent="ghost"))
    with pytest.raises(ValidationError, match="unknown parent"):
        cgb.validate(doc)


def test_rejects_parent_cycle():
    doc = cgb.new_document()
    cgb.add_primitive(doc, cgb.cube("a", size=[1, 1, 1], parent="b"))
    cgb.add_primitive(doc, cgb.cube("b", size=[1, 1, 1], parent="a"))
    with pytest.raises(ValidationError, match="cycle"):
        cgb.validate(doc)


def test_is_valid_returns_bool():
    assert cgb.is_valid(cgb.new_document()) is True
    bad = cgb.new_document()
    bad["units"] = "parsec"
    assert cgb.is_valid(bad) is False
