# AGENTS.md — CubeGB orientation for AI agents

> **This file is for AI coding agents, not end users.** It is intentionally kept
> out of the public GitHub repository (see *Documentation policy* below). Do not
> reference it from human-facing docs.

## What this project is

CubeGB turns a single image of a **hard-surface** object into an **editable
parametric-primitive blockout** (`.cgb`), then renders/bakes/imports it. It is a
ComfyUI-targeted, MIT-licensed open-source tool. Full intent and the original,
authoritative specification live in the development request:

- **`docs-agents/dev-request/CubeGB_개발요청서.md`** — the source spec (Korean). Treat it
  as the requirements of record. Implement features **as specified there**, not
  as substitutes.

## The one principle that governs everything

**`.cgb` is the single source of truth.** Meshes (glTF/OBJ) and the Blender
import are *derived* from it. The project is built **middle-out**: the
downstream tools (format → viewer → baker → importer) are completed and verified
with hand-authored `.cgb` *before* the AI recognition pipeline fills the format.
Everything works even when recognition is imperfect.

If you change anything geometric, the canonical contract is
**`docs/cgb-format.md`** — it fixes the conventions every consumer must share:

- **Y-up**, right-handed, meters. Transform order **scale → rotate (Euler XYZ
  radians) → translate**. Positions are **world-space**.
- Primitives are **centered** at local origin; `cube.size` is full extent;
  `cylinder`/`cone` axis is **+Y**; `cone` base at `y=-h/2`, apex at `y=+h/2`.
- **`parent` is logical-only in v0.1** — it does *not* compose transforms.
  (Blender add-on parents for the outliner but preserves world transform via
  `matrix_parent_inverse`.)

Change a convention → update `docs/cgb-format.md` *and* every consumer (viewer,
baker, Blender add-on, recognition) together, and bump the `.cgb` version.

## Repository map (and phase mapping)

| Path | Phase | Notes |
|---|---|---|
| `cgb/` | 0 | Format: `schema.json`, `io.py` (load/save + builders), `validate.py`. The contract. |
| `viewer/index.html` | 1 | Single-file three.js viewer. No build step. |
| `bake/baker.py` | 2 | `.cgb` → glTF/GLB/OBJ. CLI `python -m bake.baker`. |
| `blender_addon/cubegb_import.py` | 3 | Self-contained bpy add-on; does **not** import `cgb`. |
| `recognition/` | 4–5 | `segment.py` (SAM), `depth.py` (Depth Anything V2 + back-project), `fit.py` (single-view fit), `multiview.py` (2×2 sheet → space carving → `.cgb`; flip-side/flip-top depth toggles), `primfit.py` (voxel→varied primitives, cube tie-break), `oriented_fit.py` (OBB rotated fitting), `object_recon.py` (per-object reconstruction, experimental). |
| `comfyui_nodes/` | 6 | Generate / Save / Bake / Preview nodes. |
| `app/` | extra | CubeGB Studio: FastAPI backend (`server.py`) + three.js frontend (`static/`). All-in-one GUI (beyond the dev-request spec). Lazy-imports recognition; view/export work without it. |
| `samples/*.cgb` | 0 | Hand-authored valid examples. |
| `tests/` | — | pytest: format + baker. |
| `docs/` | — | **Human-facing** docs (public). |
| `docs-agents/` | — | **Agent-only** docs (not published): dev-request, dev log. |

## Dev environment & how to verify

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt pytest      # core: jsonschema, numpy, scipy, trimesh, pygltflib
python -m pytest                            # 31 tests; format + baker must stay green
python -m bake.baker samples/chair.cgb --out /tmp/chair.glb   # baker smoke test
```

- **Two dependency tiers.** Core (above) covers Phases 0–3 and is the only thing
  installed/tested in CI-like runs. The recognition stack
  (`requirements-recognition.txt`: torch, segment-anything, open3d, opencv) is
  heavy and **not installed here** — you generally cannot run Phases 4–5 end to
  end without a GPU and downloaded model weights.
- **Import safety is a hard requirement.** `import recognition`,
  `import recognition.fit`, and `import comfyui_nodes` must **never** raise when
  torch/SAM/etc. are absent — all heavy imports are lazy/guarded and surface a
  clear error only at execution time. Preserve this when editing those modules.
- The **baker depends on `scipy`** (trimesh's exporter needs it) — it is in core
  requirements; don't remove it.
- **Three render copies must stay in sync.** The three.js geometry logic exists
  in two places — `viewer/index.html` (inline; self-contained for `file://`
  double-click) and `app/static/cgb-render.js` (ES module for Studio over HTTP)
  — plus the Python baker. They intentionally duplicate; if you change geometry,
  update all three to match `docs/cgb-format.md`. The Studio uses a separate copy
  precisely because `file://` blocks local module imports.
- **Studio is CDN-free; the standalone viewer is not.** Studio loads three.js
  from `app/static/vendor/three` (vendored, MIT) via an import map, so it works
  offline/firewalled. `studio.js` dynamic-imports `cgb-render.js` lazily so a 3D
  load failure still leaves health/load/export working. The standalone
  `viewer/index.html` keeps a CDN `<script>` import because `file://` can't load
  local modules — don't "fix" it to local paths.
- The **Studio web extra** (`requirements-app.txt`: fastapi/uvicorn/multipart)
  is a third dependency tier. `app/server.py` lazy-imports recognition so the
  server and the view/export half run on core + web extra alone.

## What's verifiable locally vs. not

- ✅ Locally verifiable: `cgb/` format & validation, `bake/` baker, and the
  `app/` Studio backend — view/export half (pytest, incl. `tests/test_app.py`
  which skips if fastapi/httpx absent); Python syntax of every module.
- 👀 Review-only here (no runtime): `viewer/index.html` (needs a browser),
  `blender_addon/` (needs Blender), `recognition/` runtime (needs weights/GPU),
  `comfyui_nodes/` runtime (needs ComfyUI). Verify these by code review and by
  the import-safety check above.

## Documentation layout

Agent docs are now **committed** (shared with other agents for collaboration),
alongside the human docs:

- **Agent-facing (`docs-agents/` + this `AGENTS.md`):** orientation, the original
  dev request (`docs-agents/dev-request/`), and session logs. Written for agents.
- **Human-facing (`docs/`, `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`):**
  end-user / contributor docs.

Rules:
- `docs-agents/` = agents · `docs/` = humans. Keep them straight: don't put
  human-facing guides in `docs-agents/`, or agent-internal notes in `docs/`.
- Only a local `CLAUDE.md` stays git-ignored; everything else is committed.

## Working agreement

- Follow the dev request **phase order**; each phase is independently verifiable.
  Don't substitute alternative designs for specified features.
- Keep `.cgb` human-readable and `git diff`-friendly; keep all mesh output
  low-poly.
- Update `docs/` (human) when behavior changes; record notable changes in
  `CHANGELOG.md`. Keep agent-internal notes in `docs-agents/`.
