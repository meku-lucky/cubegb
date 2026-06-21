"""Phase 5 — fit parametric primitives to per-segment clouds and write ``.cgb``.

This is the core of the recognition pipeline. For each segment point cloud
(produced by :mod:`recognition.depth`) it:

1. Runs **PCA** to find the principal axes and extents, giving a pose-normalised
   oriented frame. Dominant axes that are *close* to a world axis are **snapped**
   to it so the blockout favours clean axis-aligned boxes over slightly tilted
   ones (per the spec: "가급적 월드 축에 정렬").
2. Fits four primitive candidates — **cube** (oriented bounding box), **cylinder**,
   **cone**, **sphere** — and keeps the one with the lowest *normalised* residual.
3. Applies a **symmetry / occlusion** heuristic: a single image only sees front
   surfaces, so the unseen depth (the axis pointing away from the camera, world
   ``-z``) is padded to a plausible thickness so boxes are not paper-thin.
4. Emits one ``cgb`` primitive (with a material colour sampled from the segment's
   mean pixel RGB), validates, and saves the document.

A top-level :func:`image_to_cgb` orchestrates segment → depth → backproject →
fit → save, and a small ``argparse`` CLI wraps it so::

    python -m recognition.fit image.jpg --out result.cgb

All heavy deps (``scipy``, ``open3d``, ``torch``, …) are imported lazily; this
module imports cleanly with only ``numpy`` + the local ``cgb`` package present.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Allow both `python -m recognition.fit` and `python recognition/fit.py`.
if __package__ in (None, ""):  # pragma: no cover - import-path shim
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cgb  # noqa: E402

DEFAULT_COLOR = (0.7, 0.7, 0.72)

# A dominant cloud axis is "snapped" to a world axis when the angle between them
# is below this threshold (radians ≈ 12°). Keeps near-aligned boxes axis-aligned.
_SNAP_ANGLE_RAD = math.radians(12.0)


# --------------------------------------------------------------------------- #
# Fit result container
# --------------------------------------------------------------------------- #
@dataclass
class FitResult:
    """One fitted primitive, ready to hand to the ``cgb`` builders.

    ``residual`` is the normalised RMS error of the chosen primitive (smaller =
    better fit), used only for ranking/diagnostics.
    """

    prim_type: str            # "cube" | "cylinder" | "cone" | "sphere"
    position: tuple[float, float, float]
    rotation_euler: tuple[float, float, float]
    params: dict              # cgb params for this type
    residual: float
    color: tuple[float, float, float] = DEFAULT_COLOR


# --------------------------------------------------------------------------- #
# PCA / pose normalisation
# --------------------------------------------------------------------------- #
def _pca(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(centroid, axes, half_extents)`` for a point cloud.

    ``axes`` is a ``3x3`` rotation whose columns are the principal directions
    sorted by descending variance; ``half_extents`` are the half-widths of the
    cloud along those axes (so the oriented bounding box is
    ``centroid ± axes @ diag(half_extents)``).
    """
    centroid = points.mean(axis=0)
    centered = points - centroid

    # Covariance eigendecomposition (symmetric → eigh). Columns are eigenvectors.
    cov = np.cov(centered, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]  # largest variance first
    axes = evecs[:, order]

    # Make it a proper right-handed rotation (det = +1), not a reflection.
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1.0

    # Extents along each principal axis.
    proj = centered @ axes  # (N, 3) coordinates in the PCA frame
    half_extents = (proj.max(axis=0) - proj.min(axis=0)) / 2.0
    # Recenter: the OBB centre may differ from the centroid for skewed clouds.
    obb_center_local = (proj.max(axis=0) + proj.min(axis=0)) / 2.0
    obb_center = centroid + axes @ obb_center_local
    return obb_center, axes, half_extents


def _snap_axes_to_world(axes: np.ndarray) -> np.ndarray:
    """Snap principal axes toward world axes when they are nearly aligned.

    For each principal axis we find the closest world axis (±x/±y/±z); if the
    angle is within :data:`_SNAP_ANGLE_RAD` we replace it with that world axis.
    The result is re-orthonormalised (Gram-Schmidt) and forced right-handed so
    it stays a valid rotation. This biases the blockout toward clean,
    axis-aligned primitives rather than slightly tilted ones.
    """
    world = np.eye(3)
    snapped = axes.copy()
    used: list[int] = []  # world axes already claimed, to avoid collisions

    # Process principal axes in order (most significant first).
    for i in range(3):
        v = axes[:, i]
        best_j, best_sign, best_cos = -1, 1.0, -1.0
        for j in range(3):
            if j in used:
                continue
            c = float(np.dot(v, world[:, j]))
            if abs(c) > best_cos:
                best_cos = abs(c)
                best_j = j
                best_sign = 1.0 if c >= 0 else -1.0
        if best_j >= 0 and best_cos >= math.cos(_SNAP_ANGLE_RAD):
            snapped[:, i] = best_sign * world[:, best_j]
            used.append(best_j)

    # Re-orthonormalise (Gram-Schmidt) so snapping a subset stays a rotation.
    q = np.zeros((3, 3))
    for i in range(3):
        v = snapped[:, i].copy()
        for k in range(i):
            v -= np.dot(v, q[:, k]) * q[:, k]
        n = np.linalg.norm(v)
        # Degenerate column (two axes snapped parallel): fall back to original.
        q[:, i] = v / n if n > 1e-9 else axes[:, i]
    if np.linalg.det(q) < 0:
        q[:, -1] *= -1.0
    return q


def _rotation_to_euler_xyz(R: np.ndarray) -> tuple[float, float, float]:
    """Decompose a ``3x3`` rotation into XYZ Euler angles (radians).

    Matches the baker's ``euler_matrix(..., "sxyz")`` convention
    (``R = Rx · Ry · Rz``), so re-baking reproduces the same orientation.
    """
    R = np.asarray(R, dtype=np.float64)
    sy = -R[2, 0]
    sy = max(-1.0, min(1.0, sy))  # clamp for asin domain safety
    ry = math.asin(sy)
    if abs(sy) < 0.999999:
        rx = math.atan2(R[2, 1], R[2, 2])
        rz = math.atan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        rx = math.atan2(-R[1, 2], R[1, 1])
        rz = 0.0
    return (rx, ry, rz)


# --------------------------------------------------------------------------- #
# Occlusion / symmetry recovery
# --------------------------------------------------------------------------- #
def _recover_hidden_depth(half_extents: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """Pad the camera-facing-away axis so single-view shells become solids.

    A single photo only captures front surfaces, so the world ``-z`` (depth)
    extent of a raw cloud is a thin shell. Heuristic: find the principal axis
    most aligned with world ``z`` and, if its visible half-extent is much smaller
    than the object's in-plane size, grow it so the object's *depth* is at least
    ~60% of its larger in-plane dimension. This mirrors the "assume depth ≈
    visible width" symmetry assumption and keeps boxes from being paper-thin.

    Returns a copy of ``half_extents`` with the depth axis padded.
    """
    he = half_extents.copy()
    z_world = np.array([0.0, 0.0, 1.0])
    # Which principal axis points most along world z?
    alignment = np.abs(axes.T @ z_world)  # |cos| of each axis with world z
    depth_axis = int(np.argmax(alignment))

    # In-plane reference = largest of the *other* two half-extents.
    others = [he[i] for i in range(3) if i != depth_axis]
    in_plane = max(others) if others else he[depth_axis]

    min_depth = 0.6 * in_plane  # target full depth ≈ 60% of in-plane size
    if he[depth_axis] * 2.0 < min_depth:
        he[depth_axis] = min_depth / 2.0
    return he


def _apply_occlusion_recovery(
    fit: "FitResult",
    axes: np.ndarray,
    visible_he: np.ndarray,
) -> "FitResult":
    """Pad the chosen primitive's depth so single-view shells become solids.

    A single photo only sees front surfaces, so the world-z extent of every
    cloud is a thin shell. We grow the depth dimension to a plausible thickness
    (see :func:`_recover_hidden_depth`) and, because the *visible* surface is the
    front face, shift the centre **back** (world ``-z``) by the amount added so
    the front of the solid stays where the photo saw it. Cube width/height and
    cylinder/cone radius/height are untouched.
    """
    padded_he = _recover_hidden_depth(visible_he, axes)
    z_world = np.array([0.0, 0.0, 1.0])
    alignment = np.abs(axes.T @ z_world)
    depth_axis = int(np.argmax(alignment))
    added = float(padded_he[depth_axis] - visible_he[depth_axis])
    if added <= 1e-9:
        return fit  # nothing padded (e.g. already deep enough)

    # Shift centre back by half the added depth, along the principal depth axis
    # oriented toward world -z (away from camera).
    axis_vec = axes[:, depth_axis]
    if float(np.dot(axis_vec, z_world)) > 0:
        axis_vec = -axis_vec  # point away from the camera
    new_center = np.asarray(fit.position) + axis_vec * (added)

    params = dict(fit.params)
    if fit.prim_type == "cube":
        # Grow only the depth component of the box size.
        size = list(params["size"])
        size[depth_axis] = padded_he[depth_axis] * 2.0
        params["size"] = [float(s) for s in size]
    # For cylinder/cone/sphere the radius already spans the cross-section, so
    # padding the principal-depth half-extent does not change their params; the
    # back-shift alone seats them more solidly. (A thin disc-like cloud fit as a
    # cylinder keeps its radius; only the centre nudges back.)

    return FitResult(
        fit.prim_type,
        (float(new_center[0]), float(new_center[1]), float(new_center[2])),
        fit.rotation_euler,
        params,
        fit.residual,
        fit.color,
    )


# --------------------------------------------------------------------------- #
# Primitive fitters — each returns (params, residual) or None
# --------------------------------------------------------------------------- #
def _fit_cube(
    points: np.ndarray,
    center: np.ndarray,
    axes: np.ndarray,
    half_extents: np.ndarray,
) -> tuple[dict, float]:
    """Oriented bounding box → cube ``size`` (full extent) + residual."""
    size = np.maximum(half_extents * 2.0, 1e-3)
    # Residual: how well points lie *on* the box shell. For each point we take
    # the signed distance to the nearest box face; |that| is the surface error.
    # (An OBB trivially *contains* its points, so an outside-overflow residual is
    # always ~0 and would make cube always win — we need a surface-fit residual
    # comparable to the curved primitives' radial-stdev residuals.)
    local = (points - center) @ axes
    he = np.maximum(half_extents, 1e-9)
    # Distance from each face along each axis (negative inside, 0 on the face).
    dist_to_face = he - np.abs(local)         # >=0 inside, per axis
    surface_err = np.abs(dist_to_face).min(axis=1)  # nearest face distance
    rms = float(np.sqrt((surface_err ** 2).mean()))
    diag = float(np.linalg.norm(size)) + 1e-9
    return {"size": [float(s) for s in size]}, rms / diag


def _fit_sphere(points: np.ndarray) -> tuple[dict, float]:
    """Center + radius via mean radial distance; residual = radial stdev."""
    center = points.mean(axis=0)
    r = np.linalg.norm(points - center, axis=1)
    radius = float(r.mean())
    if radius < 1e-6:
        return {"radius": 1e-3, "segments": cgb.DEFAULT_SEGMENTS}, 1.0
    residual = float(r.std()) / radius  # normalised by radius
    return {"radius": radius, "segments": cgb.DEFAULT_SEGMENTS}, residual


def _fit_cylinder(
    points: np.ndarray,
    center: np.ndarray,
    axes: np.ndarray,
    half_extents: np.ndarray,
) -> tuple[dict, float]:
    """Cylinder along the PCA major axis: radius from cross-section, height from
    the axial extent. Residual = stdev of radial distance to the axis."""
    # Major principal axis (axes[:, 0]) is the cylinder axis; cross-section lies
    # in the other two principal directions (local y/z below).
    local = (points - center) @ axes
    height = float(half_extents[0] * 2.0)
    # Radial distance in the plane perpendicular to the major axis.
    radial = np.sqrt(local[:, 1] ** 2 + local[:, 2] ** 2)
    radius = float(radial.mean())
    if radius < 1e-6 or height < 1e-6:
        return {"radius": 1e-3, "height": 1e-3, "segments": cgb.DEFAULT_SEGMENTS}, 1.0
    residual = float(radial.std()) / radius
    return (
        {"radius": radius, "height": height, "segments": cgb.DEFAULT_SEGMENTS},
        residual,
    )


def _fit_cone(
    points: np.ndarray,
    center: np.ndarray,
    axes: np.ndarray,
    half_extents: np.ndarray,
) -> tuple[Optional[dict], float]:
    """Cone along the PCA major axis: detect a linear radius taper.

    We bin points along the axis and measure mean radius per bin; a good cone has
    radius shrinking roughly linearly toward one end. Returns ``None`` (rejected)
    when the taper is too weak to prefer a cone over a cylinder.
    """
    local = (points - center) @ axes
    t = local[:, 0]  # coordinate along the major axis
    radial = np.sqrt(local[:, 1] ** 2 + local[:, 2] ** 2)
    height = float(half_extents[0] * 2.0)
    if height < 1e-6:
        return None, 1.0

    # Least-squares line radius(t) = a*t + b.
    A = np.stack([t, np.ones_like(t)], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, radial, rcond=None)
    pred = A @ np.array([a, b])
    base_radius = float(np.max(radial))
    if base_radius < 1e-6:
        return None, 1.0

    # Taper strength: radius change across the height vs. the base radius.
    taper = abs(a) * height / base_radius
    if taper < 0.35:  # not tapered enough — let the cylinder win instead
        return None, 1.0

    residual = float(np.sqrt(np.mean((radial - pred) ** 2))) / base_radius
    # cgb cone: base (radius r) at y=-h/2, apex at y=+h/2. Our fit gives the
    # widest end; orientation alignment is handled by the axis→Y mapping later.
    return (
        {"radius": base_radius, "height": height, "segments": cgb.DEFAULT_SEGMENTS},
        residual,
    )


# --------------------------------------------------------------------------- #
# Axis→+Y alignment for cylinder/cone
# --------------------------------------------------------------------------- #
def _axis_rotation_to_y(axis_dir: np.ndarray) -> np.ndarray:
    """Rotation mapping ``+Y`` onto ``axis_dir`` (cylinder/cone axis is +Y).

    cgb cylinders/cones have their axis along local +Y; to orient a fitted part
    we need the rotation that takes +Y to the fitted major axis.
    """
    y = np.array([0.0, 1.0, 0.0])
    a = axis_dir / (np.linalg.norm(axis_dir) + 1e-12)
    v = np.cross(y, a)
    c = float(np.dot(y, a))
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        # Parallel or anti-parallel.
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))


# --------------------------------------------------------------------------- #
# Top-level per-segment fit
# --------------------------------------------------------------------------- #
def fit_primitive(
    points: np.ndarray,
    *,
    color: tuple[float, float, float] = DEFAULT_COLOR,
) -> Optional[FitResult]:
    """Fit the best primitive to a single segment cloud.

    Tries cube / cylinder / cone / sphere and returns the lowest-residual
    :class:`FitResult`, or ``None`` if the cloud is too small to fit (< 16 pts).
    Pose is PCA-normalised with world-axis snapping, and the camera-facing-away
    depth is padded via :func:`_recover_hidden_depth`.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < 16:
        return None

    raw_center, raw_axes, _ = _pca(points)
    axes = _snap_axes_to_world(raw_axes)

    # Recompute extents/centre in the (snapped) frame so they stay consistent.
    # These are the *visible* extents — residuals are computed against the real
    # points, so occlusion padding must NOT be applied here (it would distort the
    # surface-fit comparison and bias the result). Padding is applied afterward,
    # only to the winning primitive's dimensions.
    local = (points - raw_center) @ axes
    lo, hi = local.min(axis=0), local.max(axis=0)
    visible_he = (hi - lo) / 2.0
    center = raw_center + axes @ ((hi + lo) / 2.0)

    candidates: list[FitResult] = []

    # --- cube (oriented bounding box) --------------------------------- #
    cube_params, cube_res = _fit_cube(points, center, axes, visible_he)
    candidates.append(
        FitResult(
            "cube",
            tuple(center),
            _rotation_to_euler_xyz(axes),
            cube_params,
            cube_res,
            color,
        )
    )

    # --- cylinder ----------------------------------------------------- #
    cyl_params, cyl_res = _fit_cylinder(points, center, axes, visible_he)
    cyl_rot = _rotation_to_euler_xyz(_axis_rotation_to_y(axes[:, 0]))
    candidates.append(
        FitResult("cylinder", tuple(center), cyl_rot, cyl_params, cyl_res, color)
    )

    # --- cone (only if a taper is detected) --------------------------- #
    cone_params, cone_res = _fit_cone(points, center, axes, visible_he)
    if cone_params is not None:
        cone_rot = _rotation_to_euler_xyz(_axis_rotation_to_y(axes[:, 0]))
        candidates.append(
            FitResult("cone", tuple(center), cone_rot, cone_params, cone_res, color)
        )

    # --- sphere ------------------------------------------------------- #
    sph_params, sph_res = _fit_sphere(points)
    candidates.append(
        FitResult("sphere", tuple(center), (0.0, 0.0, 0.0), sph_params, sph_res, color)
    )

    # Pick the lowest normalised residual, then apply occlusion recovery to the
    # winner so single-view shells become solid blockout parts.
    best = min(candidates, key=lambda c: c.residual)
    return _apply_occlusion_recovery(best, axes, visible_he)


# --------------------------------------------------------------------------- #
# Color sampling
# --------------------------------------------------------------------------- #
def _mean_color(image_rgb: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    """Mean RGB (0..1) of the masked pixels — used as the primitive material."""
    pix = np.asarray(image_rgb)[np.asarray(mask, dtype=bool)]
    if pix.size == 0:
        return DEFAULT_COLOR
    mean = pix.reshape(-1, pix.shape[-1]).mean(axis=0) / 255.0
    return (float(mean[0]), float(mean[1]), float(mean[2]))


# --------------------------------------------------------------------------- #
# Document assembly
# --------------------------------------------------------------------------- #
def _builder_for(fit: FitResult, prim_id: str):
    """Construct the appropriate ``cgb`` primitive dict from a :class:`FitResult`."""
    transform = cgb.make_transform(
        position=fit.position,
        rotation_euler=fit.rotation_euler,
        scale=(1.0, 1.0, 1.0),
    )
    common = dict(name=prim_id, transform=transform, color=fit.color)
    if fit.prim_type == "cube":
        return cgb.cube(prim_id, fit.params["size"], **common)
    if fit.prim_type == "sphere":
        return cgb.sphere(prim_id, fit.params["radius"], fit.params["segments"], **common)
    if fit.prim_type == "cylinder":
        return cgb.cylinder(
            prim_id, fit.params["radius"], fit.params["height"],
            fit.params["segments"], **common,
        )
    if fit.prim_type == "cone":
        return cgb.cone(
            prim_id, fit.params["radius"], fit.params["height"],
            fit.params["segments"], **common,
        )
    raise ValueError(f"unknown primitive type: {fit.prim_type!r}")


def build_document(
    fits: list[FitResult],
    *,
    source_image: Optional[str] = None,
) -> dict:
    """Assemble fitted primitives into a validated ``.cgb`` document."""
    doc = cgb.new_document(source_image=source_image)
    for i, fit in enumerate(fits):
        prim_id = f"{fit.prim_type}_{i:02d}"
        cgb.add_primitive(doc, _builder_for(fit, prim_id))
    cgb.validate(doc)  # raises cgb.ValidationError on any problem
    return doc


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def image_to_cgb(
    image_path: str,
    out_path: str,
    *,
    sam_checkpoint: str,
    depth_checkpoint: Optional[str] = None,
    device: Optional[str] = None,
    sam_model_type: str = "vit_h",
    depth_backend: str = "auto",
    max_segments: int = 12,
    fov_deg: float = 55.0,
    target_size: float = 1.5,
) -> dict:
    """Run the full image → ``.cgb`` pipeline and save the result.

    Stages: segment (SAM) → depth (Depth Anything V2 / MiDaS) → back-project →
    fit primitives → assemble + validate + save.

    Returns a small summary ``dict`` (output path and per-primitive types/
    residuals). Raises a clear :class:`RuntimeError` if model weights or heavy
    deps are missing.
    """
    # Local imports keep heavy deps lazy and avoid import cycles at module load.
    from .segment import Segmenter, load_image_rgb
    from .depth import DepthEstimator, default_intrinsics, backproject

    image_rgb = load_image_rgb(image_path)
    h, w = image_rgb.shape[:2]

    # 1. Segmentation.
    segmenter = Segmenter(
        sam_checkpoint, model_type=sam_model_type, device=device
    )
    masks = segmenter.segment(image_rgb, max_masks=max_segments)
    if not masks:
        raise RuntimeError(
            "Segmentation produced no usable regions — try a clearer image or a "
            "different SAM backbone."
        )

    # 2. Depth (once for the whole image).
    depth_estimator = DepthEstimator(
        depth_checkpoint, backend=depth_backend, device=device
    )
    depth = depth_estimator.estimate(image_rgb)
    intrinsics = default_intrinsics(w, h, fov_deg=fov_deg)

    # 3 + 4. Back-project each mask and fit a primitive.
    fits: list[FitResult] = []
    for m in masks:
        points = backproject(depth, m.mask, intrinsics, target_size=target_size)
        color = _mean_color(image_rgb, m.mask)
        fit = fit_primitive(points, color=color)
        if fit is not None:
            fits.append(fit)

    if not fits:
        raise RuntimeError(
            "No primitives could be fit (all segment clouds were too small)."
        )

    # 5. Assemble, validate, save.
    doc = build_document(fits, source_image=str(image_path))
    cgb.save(doc, out_path)

    return {
        "out_path": str(out_path),
        "n_segments": len(masks),
        "n_primitives": len(fits),
        "primitives": [
            {"type": f.prim_type, "residual": round(f.residual, 4)} for f in fits
        ],
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="recognition.fit",
        description="Turn a single image into a .cgb parametric blockout.",
    )
    parser.add_argument("image", help="Path to the input image (jpg/png/…)")
    parser.add_argument(
        "--out", "-o", default=None,
        help="Output .cgb path (default: input with .cgb extension)",
    )
    parser.add_argument(
        "--sam-checkpoint", required=True,
        help="Path to a SAM checkpoint (e.g. sam_vit_h_4b8939.pth)",
    )
    parser.add_argument(
        "--sam-model-type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"],
        help="SAM backbone matching the checkpoint (default: vit_h)",
    )
    parser.add_argument(
        "--depth-checkpoint", default=None,
        help="Depth Anything V2 HF id / local .pth, or MiDaS model name "
             "(default: Depth-Anything-V2-Small-hf)",
    )
    parser.add_argument(
        "--depth-backend", default="auto",
        choices=["auto", "depth_anything_v2", "midas"],
        help="Depth backend (default: auto = Depth Anything V2, then MiDaS)",
    )
    parser.add_argument(
        "--device", default=None,
        help="Torch device (cuda/mps/cpu); default auto-detects",
    )
    parser.add_argument(
        "--max-segments", type=int, default=12,
        help="Maximum number of segments/primitives to keep (default: 12)",
    )
    parser.add_argument(
        "--fov", type=float, default=55.0,
        help="Assumed horizontal field of view in degrees (default: 55)",
    )
    parser.add_argument(
        "--target-size", type=float, default=1.5,
        help="Metric size the whole object is normalised to, in metres "
             "(default: 1.5)",
    )
    args = parser.parse_args(argv)

    out_path = args.out or str(Path(args.image).with_suffix(".cgb"))

    try:
        summary = image_to_cgb(
            args.image,
            out_path,
            sam_checkpoint=args.sam_checkpoint,
            depth_checkpoint=args.depth_checkpoint,
            device=args.device,
            sam_model_type=args.sam_model_type,
            depth_backend=args.depth_backend,
            max_segments=args.max_segments,
            fov_deg=args.fov,
            target_size=args.target_size,
        )
    except (RuntimeError, cgb.ValidationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    types = ", ".join(p["type"] for p in summary["primitives"])
    print(
        f"recognized {args.image} -> {summary['out_path']} "
        f"({summary['n_primitives']} primitives: {types})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
