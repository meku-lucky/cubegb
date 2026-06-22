"""Build README figures into images/ (hero render + pipeline stage strips).

Run once after the recognition stack + a SAM checkpoint are available:

    python scripts/_build_doc_figures.py

Uses the treasure-chest concept art (single-view stages) and its 2x2 sheet
(multi-view stages) — the cleanest end-to-end example from testing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
IMAGES = ROOT / "images"
SAM = str(ROOT / "models" / "sam_vit_h_4b8939.pth")
PAD = 16
LABEL_H = 28
BG = (245, 246, 249)


def _font(sz=18):
    try:
        return ImageFont.truetype("arial.ttf", sz)
    except Exception:
        return ImageFont.load_default()


def _fit(img: Image.Image, h: int) -> Image.Image:
    w = int(img.width * h / img.height)
    return img.resize((w, h), Image.LANCZOS)


def _strip(panels: list[tuple[str, Image.Image]], h: int = 240) -> Image.Image:
    """Lay panels left-to-right with labels and ' -> ' arrows between them."""
    font = _font(18); afont = _font(26)
    imgs = [(_lbl, _fit(im.convert("RGB"), h)) for _lbl, im in panels]
    arrow_w = 44
    total_w = sum(im.width for _, im in imgs) + arrow_w * (len(imgs) - 1) + PAD * 2
    total_h = h + LABEL_H + PAD * 2
    canvas = Image.new("RGB", (total_w, total_h), BG)
    drw = ImageDraw.Draw(canvas)
    x = PAD
    for i, (label, im) in enumerate(imgs):
        canvas.paste(im, (x, PAD + LABEL_H))
        tw = drw.textlength(label, font=font)
        drw.text((x + (im.width - tw) / 2, PAD), label, fill=(60, 64, 72), font=font)
        x += im.width
        if i < len(imgs) - 1:
            drw.text((x + arrow_w / 2 - 8, PAD + LABEL_H + h / 2 - 14), "→",
                     fill=(120, 126, 138), font=afont)
            x += arrow_w
    return canvas


def main() -> None:
    IMAGES.mkdir(exist_ok=True)
    from recognition.multiview import split_sheet, carve_occupancy
    from recognition.render import render_cgb
    from recognition import debug as dbg
    from recognition.segment import Segmenter, load_image_rgb
    from recognition.depth import DepthEstimator

    chest_concept = ROOT / "test_images" / "treasure_chest_concept.png"
    chest_sheet = ROOT / "test_images" / "treasure_chest_2x2_views.png"

    # ---- HERO: rendered multi-view blockout --------------------------------
    render_cgb(str(ROOT / "debug_out" / "treasure_chest_mv.cgb"),
               str(IMAGES / "hero-treasure-chest.png"),
               size=720, azimuth=28, elevation=18)
    print("wrote images/hero-treasure-chest.png")

    # ---- SINGLE-VIEW pipeline strip: concept | segmentation | depth | result
    img = load_image_rgb(str(chest_concept))
    seg = Segmenter(SAM, model_type="vit_h")
    masks = seg.segment(img, max_masks=12)
    overlay = dbg._overlay_masks(img, masks)
    depth = DepthEstimator(None).estimate(img)
    heat = dbg._depth_heatmap(depth)
    render_cgb(str(ROOT / "debug_out" / "treasure_chest_concept.cgb"
               if (ROOT / "debug_out" / "treasure_chest_concept.cgb").exists()
               else ROOT / "debug_out" / "treasure_chest_mv.cgb"),
               str(IMAGES / "_sv_result.png"), size=360, azimuth=28, elevation=18)
    sv = _strip([
        ("concept art", Image.fromarray(img)),
        ("1. segment parts (SAM)", Image.fromarray(overlay)),
        ("2. estimate depth", Image.fromarray(heat)),
        ("3. fit primitives (.cgb)", Image.open(IMAGES / "_sv_result.png")),
    ])
    sv.save(IMAGES / "pipeline-single-view.png")
    print("wrote images/pipeline-single-view.png")

    # ---- MULTI-VIEW pipeline strip: sheet | silhouettes | occupancy | result
    sheet = load_image_rgb(str(chest_sheet))
    cells = split_sheet(sheet)
    # 2x2 silhouette panel
    n = 256
    grid = Image.new("RGB", (n, n), (20, 20, 24))
    for c, (r0, c0) in zip(cells, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        s = (np.stack([c.silhouette] * 3, -1).astype(np.uint8) * 255)
        grid.paste(_fit(Image.fromarray(s).resize((n // 2, n // 2)), n // 2),
                   (c0 * n // 2, r0 * n // 2))
    occ = carve_occupancy(cells, res=96)
    front = np.flipud(occ.max(2).T).astype(np.uint8) * 255
    side = np.flipud(occ.max(0).T).astype(np.uint8) * 255
    top = occ.max(1).T.astype(np.uint8) * 255
    occ_img = np.concatenate([front, np.full((front.shape[0], 3), 90, np.uint8),
                              side, np.full((front.shape[0], 3), 90, np.uint8), top], axis=1)
    mv = _strip([
        ("2x2 multi-view sheet", Image.fromarray(sheet)),
        ("1. silhouettes / view", grid),
        ("2. carve voxel solid", Image.fromarray(occ_img)),
        ("3. fit primitives (.cgb)", Image.open(IMAGES / "hero-treasure-chest.png")),
    ])
    mv.save(IMAGES / "pipeline-multi-view.png")
    print("wrote images/pipeline-multi-view.png")

    for tmp in ["_sv_result.png"]:
        try:
            (IMAGES / tmp).unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
