"""Validation for ``.cgb`` documents.

Two layers of checking:

1. **Schema validation** against ``cgb/schema.json`` (structure, types, required
   fields, per-primitive ``params``). Requires the ``jsonschema`` package.
2. **Semantic validation** that the schema cannot express: unique primitive ids,
   ``parent`` references that resolve, and the absence of parent cycles.

Use :func:`validate` for the combined check. It raises
:class:`ValidationError` on the first problem, or returns ``None`` on success.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

SCHEMA_PATH = Path(__file__).with_name("schema.json")


class ValidationError(Exception):
    """Raised when a ``.cgb`` document fails schema or semantic validation."""


@lru_cache(maxsize=1)
def load_schema() -> dict:
    with open(SCHEMA_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def validate_schema(doc: dict) -> None:
    """Validate ``doc`` against the JSON Schema. Raises :class:`ValidationError`."""
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ValidationError(
            "jsonschema is required for schema validation. "
            "Install it with `pip install jsonschema`."
        ) from exc

    try:
        jsonschema.validate(instance=doc, schema=load_schema())
    except jsonschema.ValidationError as exc:
        loc = "/".join(str(p) for p in exc.absolute_path) or "<root>"
        raise ValidationError(f"Schema error at {loc}: {exc.message}") from exc


def _validate_params(prim: dict) -> None:
    """Cross-field parameter checks (ranges the JSON Schema cannot express).

    Currently: partial-sweep angles on cylinder/cone must form a positive span.
    The per-bound 0..360 range is enforced by the schema; here we require
    ``sweep_start < sweep_end`` so the swept arc is non-empty.
    """
    params = prim.get("params") or {}
    pid = prim.get("id")

    if prim.get("type") in ("cylinder", "cone"):
        has_start = "sweep_start" in params
        has_end = "sweep_end" in params
        if has_start or has_end:
            start = float(params.get("sweep_start", 0.0))
            end = float(params.get("sweep_end", 360.0))
            if not (end > start):
                raise ValidationError(
                    f"Primitive {pid!r}: sweep_end ({end}) must be greater than "
                    f"sweep_start ({start})."
                )


def validate_semantics(doc: dict) -> None:
    """Check invariants the JSON Schema cannot express.

    - Primitive ids are unique.
    - Every non-null ``parent`` refers to an existing primitive id.
    - The parent graph has no cycles.
    """
    primitives = doc.get("primitives", [])

    ids: set[str] = set()
    for prim in primitives:
        pid = prim.get("id")
        if pid in ids:
            raise ValidationError(f"Duplicate primitive id: {pid!r}")
        ids.add(pid)

    # Parametric range checks the JSON Schema cannot express (cross-field).
    for prim in primitives:
        _validate_params(prim)

    parent_of: dict[str, Optional[str]] = {}
    for prim in primitives:
        parent = prim.get("parent")
        if parent is not None and parent not in ids:
            raise ValidationError(
                f"Primitive {prim['id']!r} references unknown parent {parent!r}"
            )
        parent_of[prim["id"]] = parent

    # Cycle detection via parent-chain walk with a visited guard per node.
    for start in parent_of:
        seen = set()
        node: Optional[str] = start
        while node is not None:
            if node in seen:
                raise ValidationError(
                    f"Parent cycle detected involving primitive {start!r}"
                )
            seen.add(node)
            node = parent_of.get(node)


def validate(doc: dict, *, schema: bool = True, semantics: bool = True) -> None:
    """Full validation. Raises :class:`ValidationError` on the first failure."""
    if schema:
        validate_schema(doc)
    if semantics:
        validate_semantics(doc)


def is_valid(doc: dict, **kw) -> bool:
    """Return ``True`` if ``doc`` validates, ``False`` otherwise."""
    try:
        validate(doc, **kw)
        return True
    except ValidationError:
        return False
