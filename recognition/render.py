"""Tiny offscreen renderer for ``.cgb`` blockouts (documentation / debug figures).

No GPU / OpenGL needed: it bakes the document to a mesh (via :mod:`bake.baker`),
loads it with ``trimesh``, and software-rasterises the triangles with a painter's
algorithm and flat normal shading. Good enough for README figures and quick
visual checks; not a production renderer.

    python -m recognition.render result.cgb --out result.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

if __package__ in (None, ""):  # pragma: no cover - import-path shim
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _view_matrix(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """World->camera rotation for a turntable camera (Y-up)."""
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    # camera forward (looking toward origin) from az/el on a sphere
    cf = np.array([
        -np.cos(el) * np.sin(az),
        -np.sin(el),
        -np.cos(el) * np.cos(az),
    ])
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, -cf); right /= np.linalg.norm(right)
    true_up = np.cross(-cf, right)
    return np.stack([right, true_up, -cf], axis=0)  # rows = cam axes


def render_cgb(
    cgb_path: str,
    out_png: str,
    *,
    size: int = 640,
    azimuth: float = 35.0,
    elevation: float = 22.0,
    bg=(245, 246, 249),
) -> str:
    """Render a ``.cgb`` to a PNG from a 3/4 view. Returns the output path."""
    import trimesh
    from PIL import Image, ImageDraw

    import cgb as _cgb
    from bake import baker

    doc = _cgb.load(cgb_path)
    scene = baker.bake_scene(doc)                       # transforms baked into verts
    geoms = list(scene.geometry.values())
    if not geoms:
        raise RuntimeError("empty .cgb (no primitives to render)")
    mesh = trimesh.util.concatenate(geoms)
    tris = np.asarray(mesh.triangles, dtype=np.float64)  # (F, 3, 3)
    if len(tris) == 0:
        raise RuntimeError("empty mesh")

    # Fit/centre the model, then rotate into camera space.
    centre = tris.reshape(-1, 3).mean(0)
    tris = tris - centre
    R = _view_matrix(azimuth, elevation)
    cam = tris @ R.T                                   # (F,3,3) camera coords

    # Orthographic projection with a margin; flip Y for image coords.
    span = np.abs(cam[..., :2]).max() * 2.0 + 1e-9
    s = size * 0.82 / span
    proj = cam[..., :2] * s
    proj[..., 1] *= -1.0
    proj += size / 2.0
    depth = cam[..., 2].mean(axis=1)                   # per-face mean depth

    # Flat shading from face normals (light over the shoulder).
    e1 = cam[:, 1] - cam[:, 0]
    e2 = cam[:, 2] - cam[:, 0]
    nrm = np.cross(e1, e2)
    nlen = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = nrm / np.where(nlen < 1e-9, 1.0, nlen)
    light = np.array([0.3, 0.5, 0.8]); light /= np.linalg.norm(light)
    shade = 0.30 + 0.70 * np.clip(nrm @ light, 0.0, 1.0)

    img = Image.new("RGB", (size, size), tuple(bg))
    drw = ImageDraw.Draw(img)
    base = np.array([170, 178, 190], dtype=np.float64)  # cool grey clay
    order = np.argsort(depth)                            # far -> near (painter's)
    for f in order:
        if nrm[f, 2] <= 0:                              # back-face cull
            continue
        col = tuple(int(c) for c in np.clip(base * shade[f], 0, 255))
        pts = [(float(proj[f, k, 0]), float(proj[f, k, 1])) for k in range(3)]
        drw.polygon(pts, fill=col, outline=(90, 96, 108))
    img.save(out_png)
    return out_png


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(prog="recognition.render", description="Render a .cgb to PNG.")
    ap.add_argument("cgb")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--azimuth", type=float, default=35.0)
    ap.add_argument("--elevation", type=float, default=22.0)
    a = ap.parse_args(argv)
    print(render_cgb(a.cgb, a.out, size=a.size, azimuth=a.azimuth, elevation=a.elevation))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
