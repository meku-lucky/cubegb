"""Round-trip and builder tests for the .cgb IO layer (Phase 0 checkpoint)."""

import json
from pathlib import Path

import pytest

import cgb

SAMPLES = sorted((Path(__file__).resolve().parents[1] / "samples").glob("*.cgb"))


def test_samples_exist():
    assert len(SAMPLES) >= 2, "Expected at least two hand-authored sample .cgb files"


@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.name)
def test_roundtrip_preserves_data(path, tmp_path):
    """load -> save -> load must preserve the document exactly."""
    original = cgb.load(path)
    out = tmp_path / path.name
    cgb.save(original, out)
    reloaded = cgb.load(out)
    assert reloaded == original


@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.name)
def test_dumps_is_stable(path):
    """Serializing twice yields identical text (deterministic, git-friendly)."""
    doc = cgb.load(path)
    assert cgb.dumps(doc) == cgb.dumps(cgb.loads(cgb.dumps(doc)))


def test_builders_produce_valid_document():
    doc = cgb.new_document(source_image="chair.jpg")
    cgb.add_primitive(doc, cgb.cube("seat", size=[0.5, 0.08, 0.5],
                                    transform=cgb.make_transform(position=[0, 0.4, 0]),
                                    color=[0.6, 0.4, 0.2], material_name="wood"))
    cgb.add_primitive(doc, cgb.cylinder("leg", radius=0.03, height=0.4))
    cgb.add_primitive(doc, cgb.sphere("knob", radius=0.05))
    cgb.add_primitive(doc, cgb.cone("tip", radius=0.1, height=0.2))

    # Must be JSON-serializable and schema-valid.
    json.dumps(doc)
    cgb.validate(doc)
    assert len(doc["primitives"]) == 4
    assert doc["primitives"][0]["params"]["size"] == [0.5, 0.08, 0.5]
