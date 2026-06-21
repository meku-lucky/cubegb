# Contributing to CubeGB

Thanks for your interest in CubeGB! This is an early-stage, open-source project.

## Architecture you should know first

CubeGB is built **middle-out**, and one principle governs everything:

> **`.cgb` is the single source of truth.** Meshes (glTF/OBJ) and the Blender
> import are *derived* from it. When in doubt, make `.cgb` correct first.

Read [docs/cgb-format.md](docs/cgb-format.md) before changing any component —
it fixes the geometry and coordinate conventions (Y-up, centered primitives,
transform order) that the viewer, baker, and Blender add-on must all agree on.
If you change a convention, change it in the format doc and *every* consumer.

## Development setup

```bash
python -m pip install -r requirements.txt
python -m pip install pytest
python -m pytest                # format + baker tests must pass
```

For recognition work, also install `requirements-recognition.txt` and the model
weights (see [docs/recognition.md](docs/recognition.md)).

## Project layout

See the layout table in the [README](README.md#repository-layout). Each
component maps to a development phase; the downstream tools (Phases 0–3) are
verifiable without any ML.

## Guidelines

- **Keep `.cgb` human-readable** and `git diff`-friendly (pretty JSON).
- **Stay low-poly** — blockout is the goal; all mesh output defaults to low
  tessellation.
- Add or update tests under `tests/` for format and baker changes.
- Match the existing code style: type hints, docstrings, clear comments.
- New primitive types or `operations` (CSG) are a format change → bump the
  `.cgb` version and update the schema, IO, baker, viewer, and add-on together.

## License

By contributing you agree your contributions are licensed under the project's
[MIT License](LICENSE).
