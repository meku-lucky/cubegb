"""Tests for the dense-mesh -> primitive bridge (``recognition.mesh_fit``).

The bridge is the model-agnostic core of the image-to-3D path: any dense mesh
(from TripoSR / InstantMesh / ...) -> occupancy -> existing primitive fitting ->
``.cgb``. These tests use synthetic trimesh shapes (no heavy model needed).
"""

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

import cgb
from recognition.mesh_fit import mesh_to_occupancy, mesh_to_document


def _robot():
    """A composed object: box body + cylinder neck + sphere head, stacked on +Y."""
    body = trimesh.creation.box(extents=[0.6, 0.8, 0.4]); body.apply_translation([0, 0.4, 0])
    neck = trimesh.creation.cylinder(radius=0.08, height=0.2); neck.apply_translation([0, 0.9, 0])
    head = trimesh.creation.icosphere(radius=0.25); head.apply_translation([0, 1.25, 0])
    return trimesh.util.concatenate([body, neck, head])


def test_occupancy_is_centered_grid():
    # a sphere normalised to [-0.5, 0.5] leaves the grid corners empty (round) but
    # fills the centre — a good centering/shape check.
    occ = mesh_to_occupancy(trimesh.creation.icosphere(radius=0.5), res=32)
    assert occ.shape == (32, 32, 32)
    assert occ.dtype == bool
    assert occ.any()
    assert not occ[0, 0, 0] and not occ[-1, -1, -1]  # corners empty
    assert bool(occ[16, 16, 16])  # centre filled


def test_occupancy_solid_interior():
    """fill() should solidify the shell, not just the surface."""
    occ = mesh_to_occupancy(trimesh.creation.box(extents=[1, 1, 1]), res=24)
    # the very center voxel must be filled (interior), proving it is solid
    assert bool(occ[12, 12, 12])


def test_mesh_to_document_validates_and_has_primitives():
    doc = mesh_to_document(_robot(), res=48, max_prims=8, target_size=1.5)
    cgb.validate(doc)
    assert len(doc["primitives"]) >= 2
    assert all(p["type"] in ("cube", "sphere", "cylinder", "cone") for p in doc["primitives"])


def test_sphere_head_is_recovered():
    """A clear spherical part should surface at least one sphere primitive."""
    doc = mesh_to_document(_robot(), res=64, max_prims=8, target_size=1.5)
    types = {p["type"] for p in doc["primitives"]}
    assert "sphere" in types


def test_color_is_applied_from_mesh():
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    mesh.visual.face_colors = np.tile([200, 40, 40, 255], (len(mesh.faces), 1))
    doc = mesh_to_document(mesh, res=32, max_prims=4)
    colors = [tuple(p["material"]["color"]) for p in doc["primitives"] if "material" in p]
    assert colors, "expected per-primitive colours from the mesh"
    r, g, b = colors[0]
    assert r > g and r > b  # red-dominant, matching the mesh


def test_empty_mesh_raises():
    empty = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=int))
    with pytest.raises(Exception):
        mesh_to_document(empty, res=16)
