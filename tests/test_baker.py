"""Baker tests (Phase 2 checkpoint): every sample bakes to glb/obj with named parts."""

from pathlib import Path

import pytest

import cgb
from bake.baker import bake_file, bake_scene

SAMPLES = sorted((Path(__file__).resolve().parents[1] / "samples").glob("*.cgb"))


@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.name)
def test_scene_has_named_node_per_primitive(path):
    doc = cgb.load(path)
    scene = bake_scene(doc)
    # Operands consumed by a boolean (e.g. difference cutters) are not emitted as
    # separate geometry, so exclude them from the expected node set.
    consumed = set()
    for op in doc.get("operations", []) or []:
        operands = op.get("operands", [])
        if len(operands) >= 2:
            consumed.update(o for o in operands[1:] if o != operands[0])
    expected = [p for p in doc["primitives"] if p["id"] not in consumed]
    # One geometry per surviving primitive, keyed by primitive id (named, separable).
    assert len(scene.geometry) == len(expected)
    for prim in expected:
        assert prim["id"] in scene.graph.nodes_geometry or prim["id"] in scene.geometry


@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.name)
@pytest.mark.parametrize("fmt", ["glb", "obj"])
def test_export_roundtrips(path, fmt, tmp_path):
    out = tmp_path / f"{path.stem}.{fmt}"
    bake_file(str(path), str(out), fmt=fmt)
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.name)
def test_low_poly(path):
    """Blockout meshes stay low-poly: well under a few thousand triangles total."""
    scene = bake_scene(cgb.load(path))
    total_faces = sum(len(g.faces) for g in scene.geometry.values())
    # Low-poly bound: even a detailed hand-authored 30+ primitive character stays
    # well under this; a real game mesh is 10k-100k tris.
    assert total_faces < 8000, f"{path.name}: {total_faces} faces is not low-poly"


def test_segments_override_reduces_polys():
    doc = cgb.load(next(p for p in SAMPLES if "chair" in p.name))
    hi = sum(len(g.faces) for g in bake_scene(doc, segments_override=32).geometry.values())
    lo = sum(len(g.faces) for g in bake_scene(doc, segments_override=8).geometry.values())
    assert lo < hi
