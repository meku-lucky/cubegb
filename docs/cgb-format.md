# The `.cgb` Format (v0.1)

`.cgb` is CubeGB's **source of truth**: a small, human-readable JSON document
that describes a 3D object as a set of editable parametric primitives. Every
other artifact — glTF/OBJ meshes, the web viewer, the Blender import — is
*derived* from a `.cgb`. The format is intentionally plain JSON so it stays
lossless, lightweight (KB, not MB), and `git diff`-friendly.

The machine-checkable schema lives at [`cgb/schema.json`](../cgb/schema.json).
This document is the human-readable companion and also fixes the **geometry and
coordinate conventions** that all consumers (viewer, baker, Blender add-on) must
follow so their output matches.

## Document structure

```json
{
  "format": "cgb",
  "version": "0.1",
  "metadata": {
    "generator": "CubeGB v0.1",
    "source_image": "optional/path or null",
    "created_at": "ISO8601",
    "up_axis": "Y"
  },
  "units": "meter",
  "primitives": [ /* see below */ ],
  "operations": []
}
```

| Field | Meaning |
|---|---|
| `format` | Always `"cgb"`. |
| `version` | Format version, `"MAJOR.MINOR"`. This document is `0.1`. |
| `metadata` | Free-form provenance. `up_axis` is `"Y"` for v0.1. |
| `units` | One of `meter`, `centimeter`, `millimeter`, `inch`, `foot`. Default `meter`. |
| `primitives` | The list of shapes (below). |
| `operations` | CSG ops (union/difference/intersection). **Empty and unimplemented in v0.1**; reserved for v0.2+. |

## Primitives

```json
{
  "id": "seat",
  "name": "seat",
  "type": "cube",
  "transform": {
    "position": [0.0, 0.45, 0.0],
    "rotation_euler": [0.0, 0.0, 0.0],
    "scale": [1.0, 1.0, 1.0]
  },
  "params": { "size": [0.45, 0.06, 0.45] },
  "material": { "color": [0.62, 0.42, 0.24], "name": "wood" },
  "parent": null
}
```

- `id` — unique within the document.
- `name` — human label (defaults to `id`); becomes the object/node name in
  exports and Blender.
- `type` — one of `cube`, `sphere`, `cylinder`, `cone` (v0.1).
- `transform` — placement in the scene (see conventions below).
- `params` — type-specific dimensions (see table).
- `material` — optional `{ color: [r,g,b] (0..1), name }`.
- `parent` — `id` of another primitive, or `null`. **Logical grouping only in
  v0.1** (see "Hierarchy").

### Per-type `params`

| Type | `params` | Shape definition (local space, before transform) |
|---|---|---|
| `cube` | `{ "size": [x, y, z] }` | Axis-aligned box centered at the origin. `size` is the **full extent**, so it spans `[-x/2, +x/2]` × `[-y/2, +y/2]` × `[-z/2, +z/2]`. |
| `sphere` | `{ "radius": r, "segments": 16 }` | UV sphere centered at the origin. `segments` = longitudinal divisions; latitudinal rings ≈ `max(3, segments // 2)`. |
| `cylinder` | `{ "radius": r, "height": h, "segments": 16 }` | Capped cylinder, axis along **+Y**, centered at the origin (spans `y ∈ [-h/2, +h/2]`). `segments` = radial divisions. |
| `cone` | `{ "radius": r, "height": h, "segments": 16 }` | Capped cone, axis along **+Y**, base (radius `r`) at `y = -h/2`, apex at `y = +h/2`, centered. `segments` = radial divisions. |

`segments` defaults to `16` if omitted. Keep it low — low-poly blockout is the
goal.

> **Encode dimensions in `params`, not `scale`.** Prefer keeping
> `transform.scale = [1, 1, 1]` and putting size into `params` (e.g. a thicker
> seat is a larger `size[1]`, not a `y` scale). This keeps the primitive clean
> and editable downstream. Non-uniform `scale` is *allowed* but discouraged.

## Coordinate & transform conventions

These are normative — every consumer must agree:

- **Up axis:** `Y` (`up_axis: "Y"`). Right-handed coordinate system.
- **Units:** as declared by `units` (default meters). Consumers do not rescale.
- **Transform order:** a primitive's local geometry is transformed as
  **scale → rotate → translate**, i.e. `M = T(position) · R(rotation_euler) · S(scale)`.
- **Rotation:** `rotation_euler` is in **radians**, applied in **XYZ order**
  (rotate about X, then Y, then Z).
- **Positions are in world space** (see Hierarchy).

### Hierarchy (`parent`)

In v0.1, `parent` expresses **logical grouping only** — it does *not* compose
transforms. Every primitive's `transform` is authored in **world space**, so a
child's `position` is its absolute world position regardless of its parent.

Consumers must:
- **Viewer / baker:** place every primitive directly at its world transform;
  `parent` may be used only for naming/grouping.
- **Blender add-on:** create the parent relationship for outliner organization,
  but preserve each object's authored world transform (e.g. set the child's
  parent-inverse so it does not move when parented).

True hierarchical transform composition is reserved for a future version and
would be introduced behind a version bump.

## Validation

A document is valid when it passes both:
1. **Schema** validation against `cgb/schema.json` (structure, types, per-type
   `params`).
2. **Semantic** checks: unique ids, every `parent` resolves, no parent cycles.

Use `cgb.validate(doc)` (see [`cgb/validate.py`](../cgb/validate.py)).

## Example samples

Hand-authored, schema-valid examples live in [`samples/`](../samples):
`chair.cgb`, `table.cgb`, `simple_building.cgb`.
