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
import math
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


_FULL_SWEEP = 360.0
_SWEEP_EPS = 1e-6


def _sweep_params(params: dict) -> Optional[tuple]:
    """Return ``(start_rad, length_rad, caps)`` if a *partial* sweep is requested.

    Returns ``None`` for a full 0..360 sweep so callers keep the existing
    trimesh full-cylinder/cone path (backward compatible). ``sweep_caps``
    defaults to ``True`` (a partial primitive bakes as a closed solid wedge).
    """
    if "sweep_start" not in params and "sweep_end" not in params:
        return None
    start = float(params.get("sweep_start", 0.0))
    end = float(params.get("sweep_end", _FULL_SWEEP))
    if abs((end - start) - _FULL_SWEEP) <= _SWEEP_EPS:
        return None  # full circle — nothing partial to do
    caps = bool(params.get("sweep_caps", True))
    return math.radians(start), math.radians(end - start), caps


def _partial_swept_mesh(
    r_bottom: float,
    r_top: float,
    height: float,
    segments: int,
    theta_start: float,
    theta_length: float,
    caps: bool,
) -> trimesh.Trimesh:
    """Build a partial cylinder/cone wedge (axis +Y, centered).

    Vertex placement matches three.js ``CylinderGeometry`` exactly
    (``x = r*sin(theta)``, ``z = r*cos(theta)``) so the baked mesh and the web
    viewer agree on which direction the open arc faces. A cone is the
    ``r_top == 0`` case. ``segments`` is the number of radial facets across the
    arc (same as the viewer's ``radialSegments``).
    """
    n = max(2, int(segments))
    half = height / 2.0
    verts: list[list[float]] = []
    bot_ring: list[int] = []
    top_ring: list[int] = []

    for i in range(n + 1):
        theta = theta_start + (i / n) * theta_length
        sx, cz = math.sin(theta), math.cos(theta)
        bot_ring.append(len(verts))
        verts.append([r_bottom * sx, -half, r_bottom * cz])
        top_ring.append(len(verts))
        verts.append([r_top * sx, half, r_top * cz])

    faces: list[list[int]] = []
    # Curved side surface (outward winding).
    for i in range(n):
        b0, b1 = bot_ring[i], bot_ring[i + 1]
        t0, t1 = top_ring[i], top_ring[i + 1]
        if r_bottom > _SWEEP_EPS:
            faces.append([b0, b1, t1])
        if r_top > _SWEEP_EPS:
            faces.append([b0, t1, t0])
        if r_bottom <= _SWEEP_EPS and r_top > _SWEEP_EPS:
            faces.append([b0, b1, t1])  # cone tip fan (b0 == apex projection)

    # Axial caps (the circular-sector ends along the +Y axis).
    if r_bottom > _SWEEP_EPS:
        c = len(verts)
        verts.append([0.0, -half, 0.0])
        for i in range(n):
            faces.append([c, bot_ring[i + 1], bot_ring[i]])  # faces -Y
    if r_top > _SWEEP_EPS:
        c = len(verts)
        verts.append([0.0, half, 0.0])
        for i in range(n):
            faces.append([c, top_ring[i], top_ring[i + 1]])  # faces +Y

    # Radial end caps (the two flat cuts) — close the cross-section when asked.
    if caps:
        for i, flip in ((0, False), (n, True)):
            ax_b = len(verts)
            verts.append([0.0, -half, 0.0])
            ax_t = len(verts)
            verts.append([0.0, half, 0.0])
            rb, rt = bot_ring[i], top_ring[i]
            quad = [ax_b, rb, rt, ax_t]
            if flip:
                quad = quad[::-1]
            faces.append([quad[0], quad[1], quad[2]])
            faces.append([quad[0], quad[2], quad[3]])

    mesh = trimesh.Trimesh(
        vertices=np.asarray(verts, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=True,
    )
    # A cone's apex collapses its top ring to one point, so radial-cap quads
    # degenerate to triangles; drop the zero-area faces so the wedge stays a
    # clean manifold (else watertight checks and booleans choke on them).
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if caps and mesh.is_watertight:
        mesh.fix_normals()
    return mesh


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
        size = [float(s) for s in params["size"]]
        bevel = float((primitive.get("deform") or {}).get("bevel", 0.0))
        if bevel > 0.0:
            mesh = _beveled_box_mesh(size, bevel)
        else:
            mesh = trimesh.creation.box(extents=size)

    elif ptype == "sphere":
        r = float(params["radius"])
        n = seg()
        mesh = trimesh.creation.uv_sphere(radius=r, count=[max(3, n // 2), n])

    elif ptype == "cylinder":
        r, h = float(params["radius"]), float(params["height"])
        sweep = _sweep_params(params)
        if sweep is not None:
            mesh = _partial_swept_mesh(r, r, h, seg(), *sweep)
        else:
            mesh = trimesh.creation.cylinder(radius=r, height=h, sections=seg())
            mesh.apply_transform(_align_z_to_y())

    elif ptype == "cone":
        r, h = float(params["radius"]), float(params["height"])
        sweep = _sweep_params(params)
        if sweep is not None:
            # Cone = swept wedge with the top radius collapsed to the apex.
            mesh = _partial_swept_mesh(r, 0.0, h, seg(), *sweep)
        else:
            mesh = trimesh.creation.cone(radius=r, height=h, sections=seg())
            # trimesh cone: base at z=0, apex at z=h. Center on the origin first.
            mesh.apply_translation([0.0, 0.0, -h / 2.0])
            mesh.apply_transform(_align_z_to_y())

    else:
        raise ValueError(f"Unknown primitive type: {ptype!r}")

    _apply_deform(mesh, primitive)
    return mesh


def _beveled_box_mesh(size, bevel: float) -> trimesh.Trimesh:
    """A low-poly chamfered box (6 faces + 12 edge bevels + 8 corner tris = 44 tris).

    ``bevel`` is a ratio in (0, 0.5]; the cut width is ``bevel * min(size)`` (so
    0.5 fully rounds the shortest edge). The same 24-vertex construction is
    mirrored in the web viewers so the preview matches the baked mesh.
    """
    hx, hy, hz = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
    r = min(float(bevel), 0.5) * min(size)
    r = min(r, hx * 0.999, hy * 0.999, hz * 0.999)

    verts: list = []
    idx: dict = {}
    for sx in (1, -1):
        for sy in (1, -1):
            for sz in (1, -1):
                idx[(sx, sy, sz, "X")] = len(verts)
                verts.append([sx * hx, sy * (hy - r), sz * (hz - r)])
                idx[(sx, sy, sz, "Y")] = len(verts)
                verts.append([sx * (hx - r), sy * hy, sz * (hz - r)])
                idx[(sx, sy, sz, "Z")] = len(verts)
                verts.append([sx * (hx - r), sy * (hy - r), sz * hz])

    faces: list = []

    def quad(a, b, c, d):
        faces.append([a, b, c])
        faces.append([a, c, d])

    for sx in (1, -1):
        quad(idx[(sx, 1, 1, "X")], idx[(sx, 1, -1, "X")], idx[(sx, -1, -1, "X")], idx[(sx, -1, 1, "X")])
    for sy in (1, -1):
        quad(idx[(1, sy, 1, "Y")], idx[(1, sy, -1, "Y")], idx[(-1, sy, -1, "Y")], idx[(-1, sy, 1, "Y")])
    for sz in (1, -1):
        quad(idx[(1, 1, sz, "Z")], idx[(1, -1, sz, "Z")], idx[(-1, -1, sz, "Z")], idx[(-1, 1, sz, "Z")])
    for sx in (1, -1):
        for sy in (1, -1):
            quad(idx[(sx, sy, 1, "X")], idx[(sx, sy, -1, "X")], idx[(sx, sy, -1, "Y")], idx[(sx, sy, 1, "Y")])
    for sx in (1, -1):
        for sz in (1, -1):
            quad(idx[(sx, 1, sz, "X")], idx[(sx, -1, sz, "X")], idx[(sx, -1, sz, "Z")], idx[(sx, 1, sz, "Z")])
    for sy in (1, -1):
        for sz in (1, -1):
            quad(idx[(1, sy, sz, "Y")], idx[(-1, sy, sz, "Y")], idx[(-1, sy, sz, "Z")], idx[(1, sy, sz, "Z")])
    for sx in (1, -1):
        for sy in (1, -1):
            for sz in (1, -1):
                faces.append([idx[(sx, sy, sz, "X")], idx[(sx, sy, sz, "Y")], idx[(sx, sy, sz, "Z")]])

    # Orient every face outward. The shape is convex and centered on the origin,
    # so a face whose normal points away from its own centroid is already
    # outward; otherwise swap two indices. (Mirrored verbatim in the JS viewers,
    # which cannot call trimesh.fix_normals.)
    V = np.asarray(verts, dtype=np.float64)
    oriented: list = []
    for a, b, c in faces:
        n = np.cross(V[b] - V[a], V[c] - V[a])
        if np.dot(n, V[a] + V[b] + V[c]) < 0.0:
            oriented.append([a, c, b])
        else:
            oriented.append([a, b, c])

    mesh = trimesh.Trimesh(
        vertices=V,
        faces=np.asarray(oriented, dtype=np.int64),
        process=True,
    )
    if mesh.is_watertight:
        mesh.fix_normals()
    return mesh


def _apply_deform(mesh: trimesh.Trimesh, primitive: dict) -> None:
    """Apply local-space shape deformations (currently: taper along +Y).

    Operates on the centered local mesh *before* the primitive transform. The
    taper math is intentionally tiny and lives here as the single source of
    truth; the web viewer (``cgb-render.js`` / ``viewer/index.html``) implements
    the identical formula so the preview and the baked mesh agree.
    """
    deform = primitive.get("deform")
    if not deform:
        return

    taper = deform.get("taper")
    if taper:
        tx, tz = float(taper[0]), float(taper[1])
        v = mesh.vertices.copy()
        y = v[:, 1]
        ymin, ymax = float(y.min()), float(y.max())
        h = ymax - ymin
        if h > 1e-9:
            t = (y - ymin) / h  # 0 at the -Y end, 1 at the +Y end
            v[:, 0] *= 1.0 + (tx - 1.0) * t
            v[:, 2] *= 1.0 + (tz - 1.0) * t
            mesh.vertices = v  # reassigning invalidates normal caches


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
