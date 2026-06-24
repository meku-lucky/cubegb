"""Dense 3D mesh -> editable primitive ``.cgb`` (the "abstraction bridge").

This is the model-agnostic core of *direction 1*: take a dense mesh produced by
**any** single-image-to-3D model (TripoSR, InstantMesh, TRELLIS, Wonder3D, ...),
voxelise it into an occupancy grid, and run CubeGB's existing primitive
abstraction (``recognition.object_recon.object_to_documents``) to get a small,
editable blockout.

The value-add is exactly the part image-to-3D models do *not* give you: instead of
a dense, hard-to-edit mesh, you get a handful of named parametric primitives. The
3D model solves the "single image is too flat" problem (real volume from one
view); this bridge turns that volume into CubeGB primitives.

The image-to-3D model is a swappable *front-end* (see :func:`mesh_to_cgb` — feed
it a ``.glb``/``.obj`` from whatever generator). Nothing here depends on a
specific model or on an LLM.

CLI::

    python -m recognition.mesh_fit model.glb --out out.cgb --res 64 --max-prims 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import trimesh

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MeshLike = Union[str, Path, trimesh.Trimesh, trimesh.Scene]


# --------------------------------------------------------------------------- #
# Mesh -> occupancy
# --------------------------------------------------------------------------- #
def _as_mesh(mesh: MeshLike) -> trimesh.Trimesh:
    """Coerce a path / Scene / Trimesh into a single concatenated ``Trimesh``."""
    if isinstance(mesh, (str, Path)):
        mesh = trimesh.load(str(mesh), force="scene")
    if isinstance(mesh, trimesh.Scene):
        geoms = list(mesh.dump().geometry.values()) if hasattr(mesh.dump(), "geometry") else mesh.dump()
        if isinstance(geoms, list):
            mesh = trimesh.util.concatenate(geoms)
        else:
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Could not coerce {type(mesh)!r} into a Trimesh")
    return mesh


def mesh_to_occupancy(mesh: MeshLike, *, res: int = 64, up: str = "y") -> np.ndarray:
    """Voxelise a mesh into a centered ``(res, res, res)`` bool occupancy grid.

    The grid matches the fitter's convention: index ``i`` maps to world
    ``(i + 0.5) / res - 0.5``, i.e. the object is centered in ``[-0.5, 0.5]`` with
    its longest extent normalised to 1. ``up`` rotates a ``z``-up mesh to the
    ``y``-up CubeGB frame (many image-to-3D models emit z-up or -y-up).
    """
    mesh = _as_mesh(mesh).copy()

    if up == "z":  # z-up -> y-up: rotate -90 deg about X
        mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2.0, [1, 0, 0]))
    elif up == "-y":  # flip
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))

    mesh.apply_translation(-mesh.bounding_box.centroid)
    longest = float(max(mesh.extents.max(), 1e-9))
    mesh.apply_scale(1.0 / longest)  # coords now span ~[-0.5, 0.5]

    vg = mesh.voxelized(pitch=1.0 / res)
    try:
        vg = vg.fill()  # solidify the interior (image-to-3D meshes are shells)
    except Exception:  # noqa: BLE001 - fill is best-effort on open meshes
        pass

    pts = np.asarray(vg.points, dtype=float)
    occ = np.zeros((res, res, res), dtype=bool)
    if pts.size:
        idx = np.clip(np.floor((pts + 0.5) * res).astype(int), 0, res - 1)
        occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return occ


def _mean_color(mesh: trimesh.Trimesh) -> Optional[tuple]:
    """A single representative RGB (0..1) from the mesh's vertex/face colours."""
    visual = getattr(mesh, "visual", None)
    for attr in ("face_colors", "vertex_colors"):
        cols = getattr(visual, attr, None) if visual is not None else None
        if cols is not None and len(cols):
            m = np.asarray(cols)[:, :3].mean(axis=0) / 255.0
            return (float(m[0]), float(m[1]), float(m[2]))
    return None


# --------------------------------------------------------------------------- #
# Mesh -> .cgb
# --------------------------------------------------------------------------- #
def mesh_to_document(
    mesh: MeshLike,
    *,
    res: int = 64,
    up: str = "y",
    max_prims: int = 12,
    target_size: float = 1.5,
    ground: bool = True,
    oriented: bool = True,
    color: bool = True,
) -> dict:
    """Abstract a dense mesh into a primitive ``.cgb`` document (dict)."""
    from recognition.object_recon import object_to_documents

    src = _as_mesh(mesh)
    occ = mesh_to_occupancy(src, res=res, up=up)
    if not occ.any():
        raise ValueError("voxelisation produced an empty occupancy grid")

    prim_doc, _voxel_doc = object_to_documents(
        occ, max_prims=max_prims, target_size=target_size, ground=ground, oriented=oriented
    )

    if color:
        mc = _mean_color(src)
        if mc is not None:
            for prim in prim_doc.get("primitives", []):
                prim.setdefault("material", {})["color"] = list(mc)

    prim_doc.setdefault("metadata", {})["generator"] = "CubeGB mesh_fit (image-to-3D bridge)"
    return prim_doc


def mesh_to_cgb(mesh: MeshLike, out_path: str, **kw) -> dict:
    """Abstract a dense mesh and save the resulting ``.cgb``. Returns the doc."""
    import cgb

    doc = mesh_to_document(mesh, **kw)
    cgb.validate(doc)
    cgb.save(doc, out_path)
    return doc


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(
        prog="recognition.mesh_fit",
        description="Abstract a dense 3D mesh (from any image-to-3D model) into editable .cgb primitives.",
    )
    p.add_argument("mesh", help="Input mesh (.glb/.gltf/.obj/.ply)")
    p.add_argument("--out", "-o", help="Output .cgb (default: input with .cgb)")
    p.add_argument("--res", type=int, default=64, help="Voxel resolution (default 64)")
    p.add_argument("--up", choices=["y", "z", "-y"], default="y", help="Up axis of the input mesh")
    p.add_argument("--max-prims", type=int, default=12, help="Max fitted primitives")
    p.add_argument("--target-size", type=float, default=1.5, help="Output world size (m)")
    p.add_argument("--axis-aligned", action="store_true", help="Disable PCA-oriented fitting")
    p.add_argument("--no-ground", action="store_true", help="Do not seat the result on y=0")
    args = p.parse_args(argv)

    out = args.out or str(Path(args.mesh).with_suffix(".cgb"))
    try:
        doc = mesh_to_cgb(
            args.mesh, out, res=args.res, up=args.up, max_prims=args.max_prims,
            target_size=args.target_size, ground=not args.no_ground,
            oriented=not args.axis_aligned,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"abstracted {args.mesh} -> {out} ({len(doc['primitives'])} primitives)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
