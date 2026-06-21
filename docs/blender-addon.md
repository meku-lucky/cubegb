# Blender Importer Add-on

This is CubeGB's killer feature: it imports a `.cgb` as **real, editable Blender
primitives** — not a baked mesh. The cube you import is a Blender cube you can
grab, scale, and modify; the cylinder is a Blender cylinder with live
radius/segments. Target: Blender 3.x / 4.x.

## Install

1. Blender ▸ *Edit ▸ Preferences ▸ Add-ons ▸ Install…*
2. Select [`blender_addon/cubegb_import.py`](../blender_addon/cubegb_import.py).
3. Enable **CubeGB Importer** (category *Import-Export*).

It also runs straight from the *Text Editor*: open the file and press **Run**.

## Use

*File ▸ Import ▸ CubeGB (.cgb)*, then pick a `.cgb`. Each primitive appears in
the outliner as a named, editable object, grouped under an empty named after the
file.

## How it works

- **Editability.** Objects are created with the native operators
  (`primitive_cube_add`, `primitive_uv_sphere_add`, `primitive_cylinder_add`,
  `primitive_cone_add`). Authored dimensions live on the *object transform*, and
  `transform_apply` is never called — so each object stays a clean parametric
  primitive you can keep editing.
- **Y-up → Z-up.** `.cgb` is Y-up; Blender is Z-up. A single basis-change
  rotation (+90° about X) is applied to each primitive's world transform so
  position *and* orientation convert together and the scene is consistently
  oriented. This same rotation maps the primitives' +Y axis onto Blender's +Z,
  so cylinders/cones come out upright automatically.
- **Materials.** One Principled BSDF material per `material.name`, colored from
  `material.color`, reused across primitives.
- **Hierarchy.** v0.1 `parent` is logical-only: children are parented for
  outliner organization, but `matrix_parent_inverse` is set so they keep their
  authored world transform (no movement).

## Notes

- The add-on is self-contained — it does **not** import the project's `cgb`
  package (Blender's bundled Python won't have it on the path).
- Unknown primitive types are skipped with a warning rather than aborting the
  import; malformed files report a clear error.
