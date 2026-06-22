"""2D silhouette shape analysis for primitive-type priors.

Monocular depth on stylised concept art is weak, so the **2D outline** of a
segment is often the most reliable cue for *which* primitive a part is:

* a **circle** silhouette  -> sphere,
* a **triangle** that tapers to a point -> cone,
* a **long rounded rectangle** (constant width) -> cylinder,
* a **filled rectangle** -> cube/box.

:func:`classify` turns a boolean mask into a soft prior
``{"cube":.., "cylinder":.., "cone":.., "sphere":..}`` (sums to ~1) plus the
raw features, so the fitter can combine this with its 3D residuals rather than
trusting a thin, noisy depth shell. Everything here is pure ``numpy`` + OpenCV
(both already in the recognition extras); it degrades to a flat prior if cv2 is
missing so the pipeline still runs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

TYPES = ("cube", "cylinder", "cone", "sphere")
_FLAT_PRIOR = {t: 0.25 for t in TYPES}


@dataclass
class ShapeFeatures:
    """Silhouette descriptors used for typing (all scale-invariant)."""

    aspect: float = 1.0            # long/short side of the min-area rect (>=1)
    circularity: float = 0.0       # 4*pi*area / perim^2 (1.0 = perfect circle)
    rect_extent: float = 0.0       # area / min-area-rect area (1.0 = fills rect)
    solidity: float = 0.0          # area / convex-hull area
    taper: float = 0.0             # 1 - tip_width/base_width along major axis
    waist: float = 0.0             # mid-width / end-width (>1 = bulges, sphere)
    major_angle: float = 0.0       # orientation of the long axis, radians
    ok: bool = False               # False when cv2 missing / contour degenerate
    extra: dict = field(default_factory=dict)


def _width_profile(mask: np.ndarray, angle: float, bins: int = 12) -> np.ndarray:
    """Mean silhouette width per slice along the major axis (length ``bins``)."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return np.zeros(bins)
    cx, cy = xs.mean(), ys.mean()
    ca, sa = math.cos(-angle), math.sin(-angle)
    # rotate points into the major-axis frame
    u = (xs - cx) * ca - (ys - cy) * sa   # along major axis
    v = (xs - cx) * sa + (ys - cy) * ca   # across
    u0, u1 = u.min(), u.max()
    if u1 - u0 < 1e-6:
        return np.zeros(bins)
    idx = np.clip(((u - u0) / (u1 - u0) * bins).astype(int), 0, bins - 1)
    widths = np.zeros(bins)
    for b in range(bins):
        sel = v[idx == b]
        widths[b] = (sel.max() - sel.min()) if sel.size > 1 else 0.0
    return widths


def features(mask: np.ndarray) -> ShapeFeatures:
    """Compute :class:`ShapeFeatures` from a boolean silhouette mask."""
    try:
        import cv2  # type: ignore
    except ImportError:  # pragma: no cover - env dependent
        return ShapeFeatures(ok=False)

    m = np.ascontiguousarray(np.asarray(mask, dtype=np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return ShapeFeatures(ok=False)
    cnt = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    perim = float(cv2.arcLength(cnt, True))
    if area < 1.0 or perim < 1.0:
        return ShapeFeatures(ok=False)

    (_, _), (rw, rh), rangle = cv2.minAreaRect(cnt)
    long_side, short_side = max(rw, rh), max(min(rw, rh), 1e-6)
    aspect = long_side / short_side
    circularity = 4.0 * math.pi * area / (perim * perim)
    rect_extent = area / max(long_side * short_side, 1e-6)
    hull = cv2.convexHull(cnt)
    solidity = area / max(float(cv2.contourArea(hull)), 1e-6)

    # Major-axis angle in image coords. minAreaRect angle refers to the width
    # edge; convert so major_angle points along the longer side.
    ang = math.radians(rangle)
    if rw < rh:
        ang += math.pi / 2.0

    widths = _width_profile(mask, ang)
    nz = widths[widths > 1e-6]
    if nz.size >= 3:
        end_a = widths[:2].mean()
        end_b = widths[-2:].mean()
        tip, base = min(end_a, end_b), max(end_a, end_b)
        taper = 1.0 - tip / max(base, 1e-6)              # ~1 for a sharp cone
        mid = widths[len(widths) // 3: 2 * len(widths) // 3].mean()
        waist = mid / max((end_a + end_b) / 2.0, 1e-6)    # >1 = bulges (sphere)
    else:
        taper = waist = 0.0

    return ShapeFeatures(
        aspect=float(aspect), circularity=float(circularity),
        rect_extent=float(rect_extent), solidity=float(solidity),
        taper=float(taper), waist=float(waist), major_angle=float(ang), ok=True,
        extra={"widths": widths.tolist()},
    )


def classify(mask: np.ndarray) -> dict:
    """Soft type prior + features for a silhouette.

    Returns ``{"prior": {type: weight}, "type": argmax, "confidence": float,
    "features": ShapeFeatures}``. The prior is a blend of interpretable shape
    scores; it is intentionally *soft* so the 3D fitter can still override with
    strong residual evidence.
    """
    f = features(mask)
    if not f.ok:
        return {"prior": dict(_FLAT_PRIOR), "type": "cube",
                "confidence": 0.0, "features": f}

    def clamp(x: float) -> float:
        return 0.0 if x < 0 else (1.0 if x > 1 else x)

    def gauss(x: float, mu: float, sig: float) -> float:
        return math.exp(-0.5 * ((x - mu) / sig) ** 2)

    a, circ, ext = f.aspect, f.circularity, f.rect_extent
    low_taper = clamp(1.0 - f.taper / 0.4)

    s = {t: 0.0 for t in TYPES}

    # Cone: a strong linear taper to a point is decisive (triangle silhouette).
    s["cone"] = clamp((f.taper - 0.45) / 0.4)

    # Sphere: very round (high circularity), near-1 aspect, bulging mid-section.
    s["sphere"] = (
        clamp((circ - 0.80) / 0.18)
        * clamp(1.0 - (a - 1.0) / 0.45)
        * clamp((f.waist - 1.0) / 0.3)
    )

    # Cylinder: elongated, *rounded* outline (mid circularity band), well filled,
    # constant width (low taper). This is the rounded-rectangle silhouette.
    s["cylinder"] = (
        clamp((a - 1.2) / 1.0)
        * gauss(circ, 0.74, 0.14)
        * clamp((ext - 0.6) / 0.4)
        * low_taper
    )

    # Cube/box: filled rectangle that is either *squarish* (near-1 aspect) or
    # *sharp-cornered* (low circularity); penalised when it is very round.
    s["cube"] = (
        clamp((ext - 0.55) / 0.45)
        * low_taper
        * max(clamp(1.0 - (a - 1.0) / 0.7), clamp((0.55 - circ) / 0.3))
        * clamp(1.0 - (circ - 0.80) / 0.20)   # kill cube when nearly circular
    )

    # Floor so the prior never fully zeroes a type, then normalise.
    for t in TYPES:
        s[t] = max(s[t], 0.0) + 0.04
    total = sum(s.values())
    prior = {t: s[t] / total for t in TYPES}
    best = max(prior, key=prior.get)
    confidence = float(max(0.0, (prior[best] - 0.25) / 0.75))
    return {"prior": prior, "type": best, "confidence": confidence, "features": f}
