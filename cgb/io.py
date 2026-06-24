"""Serialization / deserialization for the ``.cgb`` parametric primitive format.

The ``.cgb`` document is the single source of truth in CubeGB. It is plain,
human-readable JSON so it stays git-diffable and lossless. This module keeps the
on-disk representation as ordinary ``dict``/``list`` structures and provides a
thin set of builder helpers for constructing valid documents programmatically.

See ``cgb/schema.json`` and ``docs/cgb-format.md`` for the full specification.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Union

PathLike = Union[str, Path]

FORMAT = "cgb"
VERSION = "0.1"
GENERATOR = "CubeGB v0.1"

PRIMITIVE_TYPES = ("cube", "sphere", "cylinder", "cone")
DEFAULT_SEGMENTS = 16


# --------------------------------------------------------------------------- #
# Load / save
# --------------------------------------------------------------------------- #
def load(path: PathLike) -> dict:
    """Load a ``.cgb`` document from disk and return it as a ``dict``.

    No validation is performed here; call :func:`cgb.validate.validate` if you
    need to guarantee the document is well-formed.
    """
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def loads(text: str) -> dict:
    """Parse a ``.cgb`` document from a JSON string."""
    return json.loads(text)


def save(doc: dict, path: PathLike, *, indent: int = 2) -> None:
    """Write a ``.cgb`` document to disk as pretty, human-readable JSON.

    ``ensure_ascii=False`` keeps names (which may be non-ASCII) readable in the
    file, and a trailing newline keeps the file POSIX/git friendly.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(dumps(doc, indent=indent))
        fh.write("\n")


def dumps(doc: dict, *, indent: int = 2) -> str:
    """Serialize a ``.cgb`` document to a deterministic JSON string."""
    return json.dumps(doc, indent=indent, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def new_document(
    *,
    source_image: Optional[str] = None,
    units: str = "meter",
    up_axis: str = "Y",
    created_at: Optional[str] = None,
    generator: str = GENERATOR,
) -> dict:
    """Create an empty, schema-valid ``.cgb`` document skeleton."""
    if created_at is None:
        created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "format": FORMAT,
        "version": VERSION,
        "metadata": {
            "generator": generator,
            "source_image": source_image,
            "created_at": created_at,
            "up_axis": up_axis,
        },
        "units": units,
        "primitives": [],
        "operations": [],
    }


def make_transform(
    position: Iterable[float] = (0.0, 0.0, 0.0),
    rotation_euler: Iterable[float] = (0.0, 0.0, 0.0),
    scale: Iterable[float] = (1.0, 1.0, 1.0),
) -> dict:
    """Build a transform block. ``rotation_euler`` is radians in XYZ order."""
    return {
        "position": [float(v) for v in position],
        "rotation_euler": [float(v) for v in rotation_euler],
        "scale": [float(v) for v in scale],
    }


def taper(x_ratio: float, z_ratio: float) -> dict:
    """Build a ``deform`` block that tapers the cross-section along +Y.

    The -Y end keeps scale 1; the +Y end is scaled by ``(x_ratio, z_ratio)``,
    linear in between. ``[0.2, 1]`` narrows a blade to a tip; ``[1.6, 1.6]``
    flares a column/pot outward.
    """
    return {"taper": [float(x_ratio), float(z_ratio)]}


def _primitive(
    prim_id: str,
    prim_type: str,
    params: dict,
    *,
    name: Optional[str] = None,
    transform: Optional[dict] = None,
    color: Optional[Iterable[float]] = None,
    material_name: Optional[str] = None,
    parent: Optional[str] = None,
    deform: Optional[dict] = None,
) -> dict:
    prim: dict[str, Any] = {
        "id": prim_id,
        "name": name if name is not None else prim_id,
        "type": prim_type,
        "transform": transform if transform is not None else make_transform(),
        "params": params,
        "parent": parent,
    }
    if deform:
        prim["deform"] = deform
    if color is not None or material_name is not None:
        material: dict[str, Any] = {}
        if color is not None:
            material["color"] = [float(c) for c in color]
        if material_name is not None:
            material["name"] = material_name
        prim["material"] = material
    return prim


def cube(prim_id: str, size: Iterable[float], **kw) -> dict:
    """Axis-aligned box. ``size`` is full extent ``[x, y, z]``."""
    return _primitive(prim_id, "cube", {"size": [float(s) for s in size]}, **kw)


def sphere(prim_id: str, radius: float, segments: int = DEFAULT_SEGMENTS, **kw) -> dict:
    return _primitive(
        prim_id, "sphere", {"radius": float(radius), "segments": int(segments)}, **kw
    )


def _add_sweep(
    params: dict,
    sweep_start: Optional[float],
    sweep_end: Optional[float],
    sweep_caps: Optional[bool],
) -> None:
    """Attach optional partial-sweep params (omitted entirely when not used)."""
    if sweep_start is not None:
        params["sweep_start"] = float(sweep_start)
    if sweep_end is not None:
        params["sweep_end"] = float(sweep_end)
    if sweep_caps is not None:
        params["sweep_caps"] = bool(sweep_caps)


def cylinder(
    prim_id: str,
    radius: float,
    height: float,
    segments: int = DEFAULT_SEGMENTS,
    *,
    sweep_start: Optional[float] = None,
    sweep_end: Optional[float] = None,
    sweep_caps: Optional[bool] = None,
    **kw,
) -> dict:
    params = {"radius": float(radius), "height": float(height), "segments": int(segments)}
    _add_sweep(params, sweep_start, sweep_end, sweep_caps)
    return _primitive(prim_id, "cylinder", params, **kw)


def cone(
    prim_id: str,
    radius: float,
    height: float,
    segments: int = DEFAULT_SEGMENTS,
    *,
    sweep_start: Optional[float] = None,
    sweep_end: Optional[float] = None,
    sweep_caps: Optional[bool] = None,
    **kw,
) -> dict:
    params = {"radius": float(radius), "height": float(height), "segments": int(segments)}
    _add_sweep(params, sweep_start, sweep_end, sweep_caps)
    return _primitive(prim_id, "cone", params, **kw)


def add_primitive(doc: dict, primitive: dict) -> dict:
    """Append a primitive to ``doc['primitives']`` and return the primitive."""
    doc.setdefault("primitives", []).append(primitive)
    return primitive
