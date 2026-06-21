"""Bake a ``.cgb`` document into a standard low-poly mesh (glTF/GLB/OBJ).

The ``.cgb`` is the source of truth; meshes are *derived*. Each primitive becomes
its own named node so the hierarchy, transforms, and per-primitive materials are
preserved in the export (open it in Blender and every part is a separate object).

Geometry conventions (centered, Y-up) match ``docs/cgb-format.md`` so the baked
mesh, the web viewer, and the Blender add-on all agree.

CLI::

    python -m bake.baker input.cgb --format glb --out out.glb
    python -m bake.baker input.cgb --format obj --out out.obj --segments 24
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

# Allow `python -m bake.baker` and `python bake/baker.py` alike.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cgb  # noqa: E402

DEFAULT_COLOR = (0.7, 0.7, 0.72)


# --------------------------------------------------------------------------- #
# Primitive geometry (local space, centered, axis = +Y for cylinder/cone)
# --------------------------------------------------------------------------- #
def _align_z_to_y() -> np.ndarray:
    """Rotation mapping +Z (trimesh primitive axis) to +Y (our convention)."""
    return trimesh.transformations.rotation_matrix(-np.pi / 2.0, [1.0, 0.0, 0.0])


def primitive_to_mesh(primitive: dict, *, segments_override: Optional[int] = None) -> trimesh.Trimesh:
    """Build a centered, Y-up :class:`trimesh.Trimesh` for one primitive's local geometry.

    The primitive's ``transform`` is *not* applied here; see :func:`_apply_transform`.
    """
    ptype = primitive["type"]
    params = primitive.get("params", {})

    def seg(default: int = cgb.DEFAULT_SEGMENTS) -> int:
        if segments_override is not None:
            return max(3, int(segments_override))
        return max(3, int(params.get("segments", default)))

    if ptype == "cube":
        mesh = trimesh.creation.box(extents=[float(s) for s in params["size"]])

    elif ptype == "sphere":
        r = float(params["radius"])
        n = seg()
        mesh = trimesh.creation.uv_sphere(radius=r, count=[max(3, n // 2), n])

    elif ptype == "cylinder":
        r, h = float(params["radius"]), float(params["height"])
        mesh = trimesh.creation.cylinder(radius=r, height=h, sections=seg())
        mesh.apply_transform(_align_z_to_y())

    elif ptype == "cone":
        r, h = float(params["radius"]), float(params["height"])
        mesh = trimesh.creation.cone(radius=r, height=h, sections=seg())
        # trimesh cone: base at z=0, apex at z=h. Center on the origin first.
        mesh.apply_translation([0.0, 0.0, -h / 2.0])
        mesh.apply_transform(_align_z_to_y())

    else:
        raise ValueError(f"Unknown primitive type: {ptype!r}")

    return mesh


def _apply_transform(mesh: trimesh.Trimesh, transform: dict) -> trimesh.Trimesh:
    """Apply scale -> rotate (Euler XYZ radians) -> translate, in world space."""
    scale = transform.get("scale", [1.0, 1.0, 1.0])
    rot = transform.get("rotation_euler", [0.0, 0.0, 0.0])
    pos = transform.get("position", [0.0, 0.0, 0.0])

    S = np.diag([float(scale[0]), float(scale[1]), float(scale[2]), 1.0])
    R = trimesh.transformations.euler_matrix(float(rot[0]), float(rot[1]), float(rot[2]), "sxyz")
    T = trimesh.transformations.translation_matrix([float(pos[0]), float(pos[1]), float(pos[2])])

    mesh.apply_transform(T @ R @ S)
    return mesh


def _apply_color(mesh: trimesh.Trimesh, primitive: dict) -> None:
    material = primitive.get("material") or {}
    color = material.get("color", DEFAULT_COLOR)
    rgba = np.array([*[int(round(c * 255)) for c in color[:3]], 255], dtype=np.uint8)
    mesh.visual.face_colors = np.tile(rgba, (len(mesh.faces), 1))


# --------------------------------------------------------------------------- #
# Scene assembly
# --------------------------------------------------------------------------- #
def bake_scene(doc: dict, *, segments_override: Optional[int] = None) -> trimesh.Scene:
    """Bake a validated ``.cgb`` document into a :class:`trimesh.Scene`.

    Each primitive is added as a separately named node so parts stay editable in
    downstream tools. Positions are world-space (v0.1 hierarchy is logical only).
    """
    scene = trimesh.Scene()
    used: dict[str, int] = {}

    for prim in doc.get("primitives", []):
        mesh = primitive_to_mesh(prim, segments_override=segments_override)
        _apply_transform(mesh, prim.get("transform", {}))
        _apply_color(mesh, prim)

        # Node names must be unique in the scene graph; ids already are, but a
        # missing/duplicate name should not collide.
        name = prim.get("name") or prim["id"]
        if name in used:
            used[name] += 1
            name = f"{name}.{used[name]:03d}"
        else:
            used[name] = 0
        scene.add_geometry(mesh, node_name=prim["id"], geom_name=name)

    return scene


def bake_file(
    in_path: str,
    out_path: str,
    *,
    fmt: Optional[str] = None,
    segments_override: Optional[int] = None,
    validate: bool = True,
) -> str:
    """Bake a ``.cgb`` file to ``out_path``. Returns the export path."""
    doc = cgb.load(in_path)
    if validate:
        cgb.validate(doc)

    scene = bake_scene(doc, segments_override=segments_override)

    out = Path(out_path)
    file_type = (fmt or out.suffix.lstrip(".") or "glb").lower()
    out.parent.mkdir(parents=True, exist_ok=True)
    scene.export(out, file_type=file_type)
    return str(out)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bake.baker",
        description="Bake a .cgb document into a glTF/GLB/OBJ mesh (low-poly).",
    )
    parser.add_argument("input", help="Path to the input .cgb file")
    parser.add_argument("--out", "-o", help="Output mesh path (default: input with new extension)")
    parser.add_argument(
        "--format", "-f", dest="fmt", choices=["glb", "gltf", "obj"],
        help="Export format (default: inferred from --out, else glb)",
    )
    parser.add_argument(
        "--segments", type=int, default=None,
        help="Override segment count for all curved primitives (low-poly default is per-primitive, 16)",
    )
    parser.add_argument("--no-validate", action="store_true", help="Skip .cgb validation")
    args = parser.parse_args(argv)

    fmt = args.fmt
    if args.out:
        out_path = args.out
    else:
        fmt = fmt or "glb"
        out_path = str(Path(args.input).with_suffix(f".{fmt}"))

    try:
        result = bake_file(
            args.input, out_path, fmt=fmt,
            segments_override=args.segments, validate=not args.no_validate,
        )
    except (cgb.ValidationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    scene = trimesh.load(result)
    n_nodes = len(getattr(scene, "geometry", {})) if hasattr(scene, "geometry") else 1
    print(f"baked {args.input} -> {result} ({n_nodes} objects)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
