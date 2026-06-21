# SPDX-License-Identifier: MIT
"""
CubeGB Importer — Blender add-on for importing `.cgb` files.

`.cgb` is CubeGB's source-of-truth format: a small JSON document that describes a
3D object as a set of editable *parametric primitives* (cube / sphere / cylinder
/ cone). The whole point of CubeGB — and the killer feature of this importer — is
that the imported result stays **editable as native Blender primitives**. After
import the user can grab a cube and tweak its dimensions, drop into Edit Mode on a
clean box, or scale a cylinder, exactly as if they had added it with
``Add > Mesh > ...`` themselves. We deliberately do NOT produce baked/triangulated
meshes.

This file is intentionally self-contained: it imports only ``bpy``, ``mathutils``,
``json`` and ``os`` so it works when installed as an add-on (where Blender's bundled
Python has no knowledge of the CubeGB project's ``cgb`` package on ``sys.path``).
It also works when pasted into Blender's Text Editor and run directly.

---------------------------------------------------------------------------
Coordinate handling: Y-up (.cgb) -> Z-up (Blender)
---------------------------------------------------------------------------
`.cgb` is **Y-up, right-handed** (``metadata.up_axis == "Y"``). Blender is
**Z-up, right-handed**. To reconcile them we apply a single, global *basis change*
to every primitive's world transform: a +90° rotation about the X axis. That
rotation maps the Y-up frame onto Blender's Z-up frame:

    (x, y, z)_Yup  ->  (x, -z, y)_Zup

We build the matrix ``BASIS_YUP_TO_ZUP`` once and left-multiply every object's
authored world matrix by it. Because we multiply the *whole* 4x4 world matrix
(not just the translation), both the **position** and the **orientation** of each
primitive are converted consistently, so the assembled scene is correctly oriented
as a whole. This is cleaner and less error-prone than swizzling components by hand
on each field.

Two of Blender's ``primitive_*_add`` operators create geometry whose natural axis
is +Z (cylinder and cone). In `.cgb` those shapes have their axis along +Y. The
global Y-up -> Z-up basis change rotates +Y onto +Z, so a `.cgb` "+Y axis"
cylinder/cone ends up pointing the right way in Blender automatically — we do NOT
need any extra per-shape fix-up. (See ``add_primitive`` for details.)

---------------------------------------------------------------------------
Editability
---------------------------------------------------------------------------
We create geometry with the real ``bpy.ops.mesh.primitive_*_add`` operators and
keep the parametric dimensions on the *object-level* transform (location /
rotation_euler / scale) rather than baking them into the mesh data. The mesh data
stays a clean unit-ish primitive, so:
  * cubes remain clean boxes (default add gives a 2 m cube; we add ``size=1`` and
    encode the requested extents via scale/dimensions),
  * cylinders/cones/spheres keep their radius & height as authored, so the user
    can still meaningfully edit them.
We never apply transforms (no ``object.transform_apply``), which is what keeps the
primitives parametrically editable downstream.
"""

import json
import os

import bpy
from mathutils import Euler, Matrix, Vector


bl_info = {
    "name": "CubeGB Importer",
    "author": "CubeGB",
    "version": (0, 1, 0),
    "blender": (3, 0, 0),
    "location": "File > Import > CubeGB (.cgb)",
    "description": "Import .cgb parametric blockouts as editable native Blender primitives",
    "category": "Import-Export",
}


# ---------------------------------------------------------------------------
# Global Y-up (.cgb) -> Z-up (Blender) basis change.
# +90 degrees about X:  (x, y, z)_Yup -> (x, -z, y)_Zup
# Applied (left-multiplied) to every primitive's authored world matrix so that
# both translation and orientation are converted consistently.
# ---------------------------------------------------------------------------
BASIS_YUP_TO_ZUP = Matrix.Rotation(1.5707963267948966, 4, 'X')


def _vec3(seq, default=(0.0, 0.0, 0.0)):
    """Coerce a JSON list into a length-3 tuple of floats, tolerating bad data."""
    try:
        x, y, z = seq
        return (float(x), float(y), float(z))
    except (TypeError, ValueError):
        return tuple(float(v) for v in default)


def authored_world_matrix(transform):
    """Build the primitive's authored world matrix in `.cgb` (Y-up) space.

    `.cgb` transform order is scale -> rotate -> translate, i.e.
    ``M = T(position) . R(rotation_euler) . S(scale)``, with rotation in radians
    applied in XYZ order. We return that matrix *before* the Y-up -> Z-up basis
    change (the caller applies the basis change).
    """
    position = _vec3(transform.get("position"), (0.0, 0.0, 0.0))
    rotation = _vec3(transform.get("rotation_euler"), (0.0, 0.0, 0.0))
    scale = _vec3(transform.get("scale"), (1.0, 1.0, 1.0))

    mat_t = Matrix.Translation(Vector(position))
    # Euler XYZ order matches the spec ("rotate about X, then Y, then Z").
    mat_r = Euler(rotation, 'XYZ').to_matrix().to_4x4()
    mat_s = Matrix.Diagonal((scale[0], scale[1], scale[2], 1.0)).to_4x4()
    return mat_t @ mat_r @ mat_s


def get_or_create_material(material_spec):
    """Return a Principled-BSDF material for the given `.cgb` material spec.

    Materials are reused by name so that, e.g., every "wood" primitive shares one
    material. ``material.color`` (linear RGB, 0..1) becomes the base color.
    Returns ``None`` if there is no material spec.
    """
    if not material_spec:
        return None

    color = _vec3(material_spec.get("color"), (0.8, 0.8, 0.8))
    name = material_spec.get("name") or "cgb_material"
    rgba = (color[0], color[1], color[2], 1.0)

    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        # Viewport solid-shading / fallback color.
        mat.diffuse_color = rgba
        principled = mat.node_tree.nodes.get("Principled BSDF")
        if principled is not None and "Base Color" in principled.inputs:
            principled.inputs["Base Color"].default_value = rgba
    return mat


def add_primitive(prim):
    """Create a native Blender primitive object for one `.cgb` primitive.

    Geometry is created with the real ``primitive_*_add`` operators so the result
    stays editable. Where a shape needs to match authored dimensions we encode
    those dimensions on the object transform (scale / dimensions), never by baking
    into the mesh, so the object remains a clean parametric primitive.

    Returns the created object, or ``None`` for an unknown primitive type.
    """
    ptype = prim.get("type")
    params = prim.get("params") or {}

    if ptype == "cube":
        # Add a clean 1 m cube, then set object dimensions to the full extents.
        # `size` in `.cgb` is the full extent of the box. Using `obj.dimensions`
        # writes the extents into object scale, leaving the mesh a unit cube.
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        obj = bpy.context.active_object
        size = _vec3(params.get("size"), (1.0, 1.0, 1.0))
        obj.dimensions = Vector(size)

    elif ptype == "sphere":
        radius = float(params.get("radius", 1.0))
        segments = int(params.get("segments", 16))
        # Latitudinal rings ~= max(3, segments // 2) per the spec.
        ring_count = max(3, segments // 2)
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=radius, segments=max(3, segments), ring_count=ring_count
        )
        obj = bpy.context.active_object

    elif ptype == "cylinder":
        radius = float(params.get("radius", 1.0))
        height = float(params.get("height", 1.0))
        segments = int(params.get("segments", 16))
        # Blender's cylinder axis is +Z by default. `.cgb` cylinders are +Y. The
        # global Y-up -> Z-up basis change rotates +Y onto +Z, so this Z-axis
        # cylinder ends up correctly oriented once the world matrix is applied;
        # no per-shape fix-up is needed here. `depth` == full height, centered.
        bpy.ops.mesh.primitive_cylinder_add(
            radius=radius, depth=height, vertices=max(3, segments)
        )
        obj = bpy.context.active_object

    elif ptype == "cone":
        radius = float(params.get("radius", 1.0))
        height = float(params.get("height", 1.0))
        segments = int(params.get("segments", 16))
        # Like the cylinder, Blender's cone axis is +Z; `.cgb` cones are +Y with
        # base at y=-h/2 and apex at y=+h/2. The Y-up -> Z-up basis change makes
        # +Y -> +Z, so the cone points the right way after the world matrix is
        # applied. radius2=0 gives a pointed apex.
        bpy.ops.mesh.primitive_cone_add(
            radius1=radius, radius2=0.0, depth=height, vertices=max(3, segments)
        )
        obj = bpy.context.active_object

    else:
        return None

    return obj


def import_cgb_document(doc, context, file_label):
    """Build Blender objects for a parsed `.cgb` document.

    Returns ``(created_objects_by_id, root_empty)``.
    """
    primitives = doc.get("primitives") or []

    # An empty to group the whole import under, named after the source file. This
    # keeps the outliner tidy and makes the whole import easy to move/delete.
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0.0, 0.0, 0.0))
    root = context.active_object
    root.name = file_label

    objects_by_id = {}

    # First pass: create every primitive with its authored world transform,
    # converted Y-up -> Z-up. We set the *object-level* matrix_world so geometry
    # stays editable.
    for prim in primitives:
        obj = add_primitive(prim)
        if obj is None:
            # Unknown type: skip but keep going.
            continue

        # Name from `name`, falling back to `id`.
        obj.name = prim.get("name") or prim.get("id") or obj.name

        # Combine the authored (Y-up) world matrix with the existing object
        # transform that the add-ops/dimensions produced (e.g. the cube's scale
        # encoding its extents). We left-apply the basis change so position AND
        # orientation convert consistently.
        authored = authored_world_matrix(prim.get("transform") or {})
        obj.matrix_world = BASIS_YUP_TO_ZUP @ authored @ obj.matrix_world

        # Material (reused by name).
        mat = get_or_create_material(prim.get("material"))
        if mat is not None:
            obj.data.materials.clear()
            obj.data.materials.append(mat)

        prim_id = prim.get("id")
        if prim_id is not None:
            objects_by_id[prim_id] = obj

    # Second pass: hierarchy. In v0.1 `parent` is LOGICAL grouping only —
    # transforms are world-space and do NOT compose. So we set the Blender parent
    # for outliner organization but preserve each child's authored world transform
    # by setting matrix_parent_inverse to the parent's inverse world matrix (this
    # cancels the parent transform so the child does not move).
    for prim in primitives:
        prim_id = prim.get("id")
        parent_id = prim.get("parent")
        obj = objects_by_id.get(prim_id)
        if obj is None:
            continue

        parent_obj = objects_by_id.get(parent_id) if parent_id else None
        if parent_obj is not None and parent_obj is not obj:
            obj.parent = parent_obj
            obj.matrix_parent_inverse = parent_obj.matrix_world.inverted()
        else:
            # Top-level primitives hang off the file root empty for organization.
            obj.parent = root
            obj.matrix_parent_inverse = root.matrix_world.inverted()

    return objects_by_id, root


def load_cgb(operator, context, filepath):
    """Read, validate-lightly and import a `.cgb` file. Returns a result set."""
    # --- Read & parse, reporting errors gracefully ------------------------
    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, IOError) as exc:
        operator.report({'ERROR'}, "Could not read file: %s" % exc)
        return {'CANCELLED'}
    except json.JSONDecodeError as exc:
        operator.report({'ERROR'}, "Invalid JSON in .cgb file: %s" % exc)
        return {'CANCELLED'}

    if not isinstance(doc, dict) or doc.get("format") != "cgb":
        operator.report(
            {'ERROR'},
            "Not a CubeGB document (missing or wrong \"format\": \"cgb\").",
        )
        return {'CANCELLED'}

    if "primitives" not in doc or not isinstance(doc.get("primitives"), list):
        operator.report({'ERROR'}, "CubeGB document has no 'primitives' list.")
        return {'CANCELLED'}

    file_label = os.path.splitext(os.path.basename(filepath))[0] or "cgb_import"

    # Make sure we are in Object Mode before adding objects (operators require it).
    if context.mode != 'OBJECT' and context.active_object is not None:
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except RuntimeError:
            pass

    try:
        objects_by_id, root = import_cgb_document(doc, context, file_label)
    except Exception as exc:  # noqa: BLE001 - surface any unexpected build error
        operator.report({'ERROR'}, "Failed to import .cgb: %s" % exc)
        return {'CANCELLED'}

    operator.report(
        {'INFO'},
        "Imported %d primitive(s) from %s" % (len(objects_by_id), os.path.basename(filepath)),
    )
    return {'FINISHED'}


# ---------------------------------------------------------------------------
# Operator + menu registration
# ---------------------------------------------------------------------------
#
# We avoid a hard module-level dependency on bpy_extras so the file is robust, but
# ImportHelper is the idiomatic base for file-import operators, so we use it.
from bpy_extras.io_utils import ImportHelper  # noqa: E402
from bpy.props import StringProperty  # noqa: E402
from bpy.types import Operator  # noqa: E402


class IMPORT_SCENE_OT_cubegb(Operator, ImportHelper):
    """Import a CubeGB (.cgb) parametric blockout as editable Blender primitives"""

    bl_idname = "import_scene.cubegb"
    bl_label = "Import CubeGB"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".cgb"
    filter_glob: StringProperty(default="*.cgb", options={'HIDDEN'})

    def execute(self, context):
        return load_cgb(self, context, self.filepath)


def menu_func_import(self, context):
    self.layout.operator(IMPORT_SCENE_OT_cubegb.bl_idname, text="CubeGB (.cgb)")


_CLASSES = (IMPORT_SCENE_OT_cubegb,)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)


# Allow running directly from Blender's Text Editor: re-register cleanly then
# invoke the file selector.
if __name__ == "__main__":
    try:
        unregister()
    except Exception:  # noqa: BLE001 - not registered yet on first run
        pass
    register()
    bpy.ops.import_scene.cubegb('INVOKE_DEFAULT')
