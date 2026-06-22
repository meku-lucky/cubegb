"""Generate shaded synthetic primitive images with KNOWN ground-truth type.

Unlike flat colour fills, these use simple Lambertian-ish shading so the
monocular depth model produces a real gradient — making them faithful tests for
both depth back-projection and 2D shape typing. Each file's stem encodes the
expected primitive type, e.g. ``synth_sphere.png`` -> sphere.

    python test_images/_synth.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
W = H = 512
BG = np.array([235, 236, 240], np.float32)  # light neutral background
LIGHT = np.array([-0.5, -0.6, 0.62])         # toward upper-left, out of screen
LIGHT = LIGHT / np.linalg.norm(LIGHT)


def _shade(base, normal_z, nx, ny):
    """Lambert shade a base RGB given surface normal components."""
    n = np.stack([nx, ny, normal_z], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    lit = np.clip((n * LIGHT).sum(-1), 0.05, 1.0)[..., None]
    return base * (0.35 + 0.65 * lit)


def _canvas():
    img = np.empty((H, W, 3), np.float32)
    img[:] = BG
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return img, xx, yy


def sphere():
    img, xx, yy = _canvas()
    cx, cy, r = W * 0.5, H * 0.5, W * 0.32
    dx, dy = (xx - cx) / r, (yy - cy) / r
    rr = dx * dx + dy * dy
    m = rr <= 1.0
    nz = np.sqrt(np.clip(1.0 - rr, 0, 1))
    shaded = _shade(np.array([200, 90, 80], np.float32), nz, dx, dy)
    img[m] = shaded[m]
    return img


def cylinder():
    img, xx, yy = _canvas()
    cx, r = W * 0.5, W * 0.20
    top, bot = H * 0.16, H * 0.84
    dx = (xx - cx) / r
    body = (np.abs(dx) <= 1.0) & (yy >= top) & (yy <= bot)
    nz = np.sqrt(np.clip(1.0 - dx * dx, 0, 1))  # curvature across width only
    shaded = _shade(np.array([80, 110, 200], np.float32), nz, dx, np.zeros_like(dx))
    img[body] = shaded[body]
    # subtle top ellipse cap for a 3D read
    ex, ey = (xx - cx) / r, (yy - top) / (r * 0.32)
    cap = (ex * ex + ey * ey <= 1.0) & (yy <= top + 2)
    img[cap] = np.array([120, 150, 230], np.float32)
    return img


def cone():
    img, xx, yy = _canvas()
    apex_y, base_y = H * 0.16, H * 0.84
    cx, base_r = W * 0.5, W * 0.26
    t = np.clip((yy - apex_y) / (base_y - apex_y), 0, 1)
    half = base_r * t
    dx = xx - cx
    m = (np.abs(dx) <= np.maximum(half, 1e-3)) & (yy >= apex_y) & (yy <= base_y)
    nx = np.divide(dx, np.maximum(half, 1e-3))
    nz = np.sqrt(np.clip(1.0 - nx * nx, 0, 1))
    shaded = _shade(np.array([90, 170, 90], np.float32), nz, nx, np.full_like(nx, -0.3))
    img[m] = shaded[m]
    return img


def box():
    img, xx, yy = _canvas()
    # Three visible faces of an axis-aligned box, each a flat shaded tone.
    front = np.array([190, 170, 90], np.float32)
    top = front * 1.18
    side = front * 0.72
    # front face quad
    fx0, fx1, fy0, fy1 = W * 0.30, W * 0.66, H * 0.40, H * 0.82
    inside = (xx >= fx0) & (xx <= fx1) & (yy >= fy0) & (yy <= fy1)
    img[inside] = front
    # top parallelogram
    for y in range(int(H * 0.22), int(fy0)):
        off = (fy0 - y) * 0.6
        x0 = int(fx0 + off); x1 = int(fx1 + off)
        img[y, max(0, x0):min(W, x1)] = top
    # right side parallelogram
    for x in range(int(fx1), int(fx1 + W * 0.18)):
        off = (x - fx1) * 1.0
        y0 = int(fy0 - off); y1 = int(fy1 - off)
        img[max(0, y0):min(H, y1), x] = side
    return img


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    for name, fn in [("sphere", sphere), ("cylinder", cylinder),
                     ("cone", cone), ("box", box)]:
        arr = np.clip(fn(), 0, 255).astype(np.uint8)
        out = HERE / f"synth_{name}.png"
        Image.fromarray(arr).save(out)
        print("wrote", out.name)


if __name__ == "__main__":
    main()
