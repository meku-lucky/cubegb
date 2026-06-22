"""Procedurally render concept-art-style HARD-SURFACE props for testing.

We have no photoreal text-to-image generator here, so these are drawn with
numpy/PIL — but with proper shading so the monocular depth model gets a real
gradient, and with clearly separable parts (box body, rounded lid, lock, feet)
that exercise segmentation + 2D type priors. Hard-surface only, matching
CubeGB's scope (organic characters like a schoolgirl/cat are out of scope).

    python test_images/_concept.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent


def _bg(w, h):
    img = np.empty((h, w, 3), np.float32)
    yy = np.linspace(0.0, 1.0, h)[:, None]
    img[:] = (236 - 14 * yy)[..., None] * np.array([1.0, 1.0, 1.02])
    return np.clip(img, 0, 255)


def treasure_chest(w=640, h=560) -> np.ndarray:
    img = _bg(w, h)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    x0, x1 = 150, 500            # body span
    body_top, body_bot = 300, 470
    base = np.array([150.0, 98.0, 56.0])    # warm wood brown

    # --- body front face (slight vertical shading) ---------------------- #
    front = (xx >= x0) & (xx <= x1) & (yy >= body_top) & (yy <= body_bot)
    shade = (0.78 + 0.22 * (1.0 - (yy - body_top) / (body_bot - body_top)))[..., None]
    img[front] = (base * shade)[front]

    # --- body right side face (parallelogram, darker) ------------------- #
    for x in range(x1, min(w, x1 + 60)):
        off = int((x - x1) * 0.7)
        y_a, y_b = body_top - off, body_bot - off
        img[max(0, y_a):min(h, y_b), x] = base * 0.55

    # --- rounded lid = half-cylinder dome (curvature shading) ----------- #
    cx = (x0 + x1) / 2.0
    ax, by = (x1 - x0) / 2.0 + 6, 120.0
    lid_base = np.array([138.0, 86.0, 48.0])
    nx = (xx - cx) / ax
    dome = (np.abs(nx) <= 1.0) & (yy <= body_top) & (yy >= body_top - by * np.sqrt(np.clip(1 - nx * nx, 0, 1)))
    nz = np.sqrt(np.clip(1.0 - nx * nx, 0, 1))
    lit = (0.45 + 0.55 * np.clip(nz - 0.25 * nx, 0.1, 1.0))[..., None]
    img[dome] = (lid_base * lit)[dome]

    # --- vertical metal bands over lid + body (subtle, low-contrast so SAM
    # does not slice the uniform front into separate panels) -------------- #
    band = base * 0.72
    for bxc in (x0 + 70, cx, x1 - 70):
        strip = (np.abs(xx - bxc) <= 7) & (yy >= body_top - by) & (yy <= body_bot) & (front | dome)
        img[strip] = band

    # --- bottom trim band ---------------------------------------------- #
    trim = (xx >= x0) & (xx <= x1) & (yy >= body_bot - 22) & (yy <= body_bot)
    img[trim] = base * 0.5

    # --- gold lock (small box on the front) ---------------------------- #
    lock = (np.abs(xx - cx) <= 26) & (yy >= body_top + 8) & (yy <= body_top + 70)
    img[lock] = np.array([222.0, 180.0, 72.0])
    keyhole = (np.abs(xx - cx) <= 5) & (yy >= body_top + 30) & (yy <= body_top + 52)
    img[keyhole] = np.array([40.0, 30.0, 16.0])

    # --- two feet (small dark cubes) ----------------------------------- #
    for fxc in (x0 + 36, x1 - 36):
        foot = (np.abs(xx - fxc) <= 22) & (yy >= body_bot) & (yy <= body_bot + 26)
        img[foot] = base * 0.42

    return np.clip(img, 0, 255).astype(np.uint8)


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    for name, fn in [("treasure_chest", treasure_chest)]:
        out = HERE / f"concept_{name}.png"
        Image.fromarray(fn()).save(out)
        print("wrote", out.name)


if __name__ == "__main__":
    main()
