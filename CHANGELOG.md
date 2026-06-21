# Changelog

All notable changes to CubeGB are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **`.cgb` format v0.1** — JSON Schema (`cgb/schema.json`), IO + builders
  (`cgb/io.py`), and schema + semantic validation (`cgb/validate.py`).
  Primitives: `cube`, `sphere`, `cylinder`, `cone`. `operations` reserved.
- Hand-authored sample documents: chair, table, simple building.
- **Web viewer** — single-file three.js page rendering `.cgb` with
  drag-and-drop, OrbitControls, and per-primitive listing.
- **Mesh baker** — `.cgb` → glTF/GLB/OBJ with named, separable, low-poly nodes
  and a CLI (`python -m bake.baker`).
- **Blender importer add-on** — imports `.cgb` as editable native primitives
  with Y-up → Z-up conversion.
- **Recognition pipeline** — SAM segmentation, Depth Anything V2 depth,
  back-projection, and PCA-based primitive fitting → `.cgb`
  (`python -m recognition.fit`).
- **ComfyUI nodes** — CubeGB Generate / Save / Bake / Preview.
- Documentation and pytest suite (format + baker).

[Unreleased]: https://github.com/cubegb/cubegb
