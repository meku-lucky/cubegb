"""Debug / evaluation harness for the recognition pipeline.

Runs the stages on an image and writes a side-by-side **montage** PNG —
``original | segmentation overlay | depth map`` — plus a printed per-part
summary (chosen primitive type, size, position). This makes the two main
failure modes visible at a glance:

* **over-segmentation** — the overlay shows one object split into many tints;
* **type misclassification** — the summary shows e.g. a round part fit as a cube.

It reuses the real pipeline (:func:`recognition.fit.image_to_cgb`) so what you
see is what Studio / the CLI produce. Usage::

    python -m recognition.debug test_images/chair.png \
        --sam-checkpoint models/sam_vit_h_4b8939.pth --out-dir debug_out

Only ``numpy`` + ``Pillow`` are needed for the montage (both already in the
recognition extras); the heavy models load lazily via the pipeline.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

if __package__ in (None, ""):  # pragma: no cover - import-path shim
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Distinct, high-contrast tints for mask overlays (RGB, 0-255).
_PALETTE = np.array([
    [228, 26, 28], [55, 126, 184], [77, 175, 74], [152, 78, 163],
    [255, 127, 0], [255, 215, 0], [166, 86, 40], [247, 129, 191],
    [153, 153, 153], [0, 206, 209], [60, 179, 113], [199, 21, 133],
], dtype=np.float32)


def _overlay_masks(image_rgb: np.ndarray, masks: list, alpha: float = 0.5) -> np.ndarray:
    """Alpha-blend each mask with a distinct tint; outline kept regions."""
    out = image_rgb.astype(np.float32).copy()
    for i, m in enumerate(masks):
        color = _PALETTE[i % len(_PALETTE)]
        seg = np.asarray(m.mask, dtype=bool)
        out[seg] = (1 - alpha) * out[seg] + alpha * color
        # crude 1px outline: dilate-xor via row/col shifts
        edge = seg ^ _erode(seg)
        out[edge] = color
    return np.clip(out, 0, 255).astype(np.uint8)


def _erode(mask: np.ndarray) -> np.ndarray:
    """1-px binary erosion using 4-neighbour shifts (no scipy dependency)."""
    m = mask
    e = m.copy()
    e[1:, :] &= m[:-1, :]
    e[:-1, :] &= m[1:, :]
    e[:, 1:] &= m[:, :-1]
    e[:, :-1] &= m[:, 1:]
    return e


def _depth_heatmap(depth: np.ndarray) -> np.ndarray:
    """Normalise a depth map to an RGB grayscale-ish heatmap (near = bright)."""
    d = np.asarray(depth, dtype=np.float32)
    lo, hi = float(d.min()), float(d.max())
    norm = (d - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(d)
    g = (norm * 255).astype(np.uint8)
    # simple blue->red ramp for readability
    rgb = np.stack([g, (np.abs(0.5 - norm) * 510).clip(0, 255).astype(np.uint8), 255 - g], axis=-1)
    return rgb


def _hstack(images: list, pad: int = 8) -> np.ndarray:
    """Horizontally stack RGB images of equal height with a light gutter."""
    h = max(im.shape[0] for im in images)
    parts = []
    for im in images:
        if im.shape[0] != h:  # letterbox to common height
            canvas = np.full((h, im.shape[1], 3), 255, np.uint8)
            canvas[: im.shape[0]] = im
            im = canvas
        parts.append(im)
        parts.append(np.full((h, pad, 3), 245, np.uint8))
    return np.concatenate(parts[:-1], axis=1)


def run_debug(
    image_path: str,
    *,
    sam_checkpoint: str,
    out_dir: str = "debug_out",
    device: Optional[str] = None,
    sam_model_type: str = "vit_h",
    max_segments: int = 12,
) -> dict:
    """Run the pipeline and write ``<out_dir>/<stem>_montage.png``; return summary."""
    from PIL import Image

    from .segment import Segmenter, load_image_rgb
    from .depth import DepthEstimator, default_intrinsics
    from .fit import image_to_cgb

    image_rgb = load_image_rgb(image_path)
    h, w = image_rgb.shape[:2]

    segmenter = Segmenter(sam_checkpoint, model_type=sam_model_type, device=device)
    masks = segmenter.segment(image_rgb, max_masks=max_segments)

    depth_est = DepthEstimator(None, device=device)
    depth = depth_est.estimate(image_rgb)

    overlay = _overlay_masks(image_rgb, masks)
    heat = _depth_heatmap(depth)
    montage = _hstack([image_rgb, overlay, heat])

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    montage_path = out_dir_p / f"{stem}_montage.png"
    Image.fromarray(montage).save(montage_path)

    cgb_path = out_dir_p / f"{stem}.cgb"
    summary = image_to_cgb(
        image_path, str(cgb_path),
        sam_checkpoint=sam_checkpoint, device=device,
        sam_model_type=sam_model_type, max_segments=max_segments,
    )

    print(f"\n=== {stem} ===")
    print(f"image {w}x{h} | kept masks: {len(masks)} | primitives: {summary['n_primitives']}")
    print(f"montage: {montage_path}")
    print(f"cgb:     {cgb_path}")
    print("parts:")
    import cgb as _cgb
    doc = _cgb.load(str(cgb_path))
    for p in doc["primitives"]:
        pr = p.get("params", {})
        dims = pr.get("size") or [pr.get("radius"), pr.get("height")]
        dims = [round(float(v), 3) for v in dims if v is not None]
        pos = [round(float(v), 3) for v in p["transform"]["position"]]
        print(f"  - {p['id']:<12} {p['type']:<9} dims={dims} pos={pos}")
    return {"montage": str(montage_path), "cgb": str(cgb_path), **summary}


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="recognition.debug",
        description="Visualise the recognition pipeline (segmentation + depth + fit).",
    )
    parser.add_argument("images", nargs="+", help="One or more input images")
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--sam-model-type", default="vit_h")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", default="debug_out")
    parser.add_argument("--max-segments", type=int, default=12)
    args = parser.parse_args(argv)

    for img in args.images:
        try:
            run_debug(
                img, sam_checkpoint=args.sam_checkpoint, out_dir=args.out_dir,
                device=args.device, sam_model_type=args.sam_model_type,
                max_segments=args.max_segments,
            )
        except Exception as exc:
            print(f"[error] {img}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
