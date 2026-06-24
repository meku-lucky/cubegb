# Development log (agent-only)

> Internal notes for agents. Not published. See `AGENTS.md` for orientation and
> `docs-agents/dev-request/CubeGB_개발요청서.md` for the authoritative spec.

## 2026-06-22 — Initial build, Phases 0–6

Built the full MVP skeleton in one pass, middle-out per the spec.

### State
- **Phase 0 (format):** `cgb/schema.json` (draft-07, per-type `params` via
  `if/then`), `cgb/io.py` (load/save + `new_document`/`cube`/`sphere`/`cylinder`/
  `cone`/`make_transform`/`add_primitive` builders), `cgb/validate.py` (schema +
  semantic: unique ids, parent resolves, no cycles). 3 hand-authored samples.
  **Tested** (`tests/test_io.py`, `tests/test_validate.py`).
- **Phase 1 (viewer):** `viewer/index.html`, three.js r0.165 via importmap,
  drag-drop, OrbitControls, primitive list, auto-framing. Review-only (browser).
- **Phase 2 (baker):** `bake/baker.py`, trimesh, named nodes per primitive,
  glb/gltf/obj, `--segments` override, CLI. **Tested** (`tests/test_baker.py`).
- **Phase 3 (Blender add-on):** `blender_addon/cubegb_import.py`, native editable
  primitives, Y-up→Z-up basis change, logical-parent via `matrix_parent_inverse`.
  Self-contained (no `cgb` import). Review-only (Blender).
- **Phase 4–5 (recognition):** `recognition/{segment,depth,fit}.py`. SAM +
  Depth Anything V2 (MiDaS fallback), pinhole back-projection, PCA fit with
  world-axis snapping, occlusion/thickness padding. Lazy/guarded heavy imports.
  CLI `python -m recognition.fit`. Runtime needs weights/GPU (not run here).
- **Phase 6 (ComfyUI):** `comfyui_nodes/{__init__,nodes.py}`. Generate/Save/Bake/
  Preview, custom `CGB` socket, lazy imports so the pack loads without torch.

### Key decisions / invariants
- **`parent` is logical-only in v0.1** (no transform composition). Samples are
  authored in world space. All consumers must honor this.
- **Y-up, centered primitives, +Y axis** for cylinder/cone. Canonical contract:
  `docs/cgb-format.md`. Baker is the reference implementation of the geometry.
- **scipy is a core dep** — trimesh's exporter needs it. Don't drop it.
- Import-safety: `import recognition*` and `import comfyui_nodes` must not raise
  without torch. There are no tests guarding this beyond the manual check in
  `AGENTS.md`; if you add CI, add an import-safety test.

### Verified here
- `python -m pytest` → 31 passed.
- Import guards confirmed with torch absent.
- Baker CLI produced valid glb/obj from all 3 samples.

## 2026-06-22 — CubeGB Studio (all-in-one GUI, beyond spec)

User asked for one GUI doing image → .cgb → 3D view → export. Chose a **custom
local web app** (FastAPI + reuse the three.js viewer) over Gradio/desktop.

- `app/server.py` — FastAPI: `GET /`, `GET /api/health`, `POST /api/generate`
  (lazy `recognition.fit.image_to_cgb`), `POST /api/bake` (`bake.baker`). Only
  `cgb` + baker load at startup; recognition is lazy → server runs without torch.
- `app/static/{index.html,studio.js,cgb-render.js}` — UI + reusable `CGBViewer`.
  `cgb-render.js` duplicates the standalone viewer's geometry logic on purpose
  (standalone is `file://`, can't import local modules; Studio is HTTP).
- `requirements-app.txt`, pyproject `app` extra + `cubegb-studio` console script.
- `tests/test_app.py` (skips without fastapi/httpx). **Verified here**: health,
  index, bake→valid GLB, bad-format 400, invalid-doc 400, generate-without-
  checkpoint clear 400. Full suite: 37 passed.
- Generate step needs weights/GPU → not run end-to-end here; UX returns a clear
  actionable error when SAM checkpoint / torch is missing.

## 2026-06-22 — Studio: vendored three.js (CDN-free) + resilient init

Windows user reported Studio stuck at "기능 확인 중…". Root cause: three.js was
loaded from `unpkg.com` via import map; that host was blocked, so the ES module
graph failed and `studio.js` never ran (health/checkHealth never fired).

Fix (both layers):
- **Vendored three.js r165 locally** at `app/static/vendor/three/`
  (`three.module.js` + `addons/controls/OrbitControls.js`, MIT). `index.html`
  import map now points to `/static/vendor/three/...` — no CDN.
- **Resilient init**: `studio.js` no longer top-level-imports the viewer; it
  `await import('/static/cgb-render.js')` lazily. On failure it shows a viewport
  message but health/`.cgb` load/primitive list/export still work (added
  three-free helpers `colorToHex`, `validateDocLocal`, `entriesFromDoc`).
- Verified: vendored files + studio.js + health all serve 200 with correct MIME;
  `node --check studio.js` passes.
- Note: standalone `viewer/index.html` still uses CDN (file:// can't import local
  modules) — intentionally left as-is.

### Not yet done / candidates (Phase 7+)
- Recognition quality is untested against real images (no weights/GPU in this
  env). Needs a real-image validation pass.
- No CI workflow yet. No import-safety regression test.
- Phase 7 ideas from the spec: differentiable render-and-compare refinement,
  capsule/torus/truncated_cone, CSG `operations`, FBX export, primitive merge/
  simplify.
- Recognition's world-axis mapping (`camera +z → world -z`) is internally
  consistent and validates, but eyeball it in the viewer once real output exists.
