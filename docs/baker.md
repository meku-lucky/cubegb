# Mesh Baker

Bakes a `.cgb` document into a standard low-poly mesh (glTF/GLB/OBJ). The `.cgb`
is the source of truth; the mesh is a *derived* artifact. Each primitive becomes
its own **named node**, so the parts stay separable when you open the result in
Blender or a glTF viewer.

## CLI

```bash
python -m bake.baker INPUT.cgb [--out OUT] [--format {glb,gltf,obj}] [--segments N] [--no-validate]
```

Examples:

```bash
python -m bake.baker samples/chair.cgb --out chair.glb
python -m bake.baker samples/table.cgb --format obj --out table.obj
python -m bake.baker samples/simple_building.cgb --segments 24 --out house.glb
```

- `--format` is inferred from `--out`'s extension if omitted (default `glb`).
- `--segments` overrides the tessellation of *all* curved primitives. Omit it to
  use each primitive's own `segments` (default 16). **Low-poly is the goal** —
  keep this small.
- The input is validated against the schema first unless `--no-validate` is set.

## Python API

```python
from bake.baker import bake_file, bake_scene
import cgb

bake_file("samples/chair.cgb", "chair.glb", fmt="glb")

# Or work with a trimesh.Scene directly:
scene = bake_scene(cgb.load("samples/chair.cgb"), segments_override=16)
print(len(scene.geometry), "named objects")
```

## Conventions

- Geometry matches [cgb-format.md](cgb-format.md): cubes use full-extent `size`;
  cylinders/cones have their axis along **+Y** and are centered; spheres are UV
  spheres. Transforms apply as scale → rotate (Euler XYZ radians) → translate.
- glTF/GLB preserve per-primitive node names and colors. OBJ also exports but,
  being a simpler format, merges parts into groups on round-trip — prefer
  **GLB** when you need named, separable objects.
- Output stays low-poly (the sample scenes bake to well under a few thousand
  triangles total).
