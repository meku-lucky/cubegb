# Web Viewer

A single self-contained HTML page that renders `.cgb` files with
[three.js](https://threejs.org/). It is the "eyes" for every CubeGB artifact —
use it to confirm a hand-authored or AI-generated `.cgb` looks right.

## Usage

1. Open [`viewer/index.html`](../viewer/index.html) in any modern browser
   (double-click it, or *File ▸ Open*). An internet connection is needed the
   first time so it can fetch three.js from the CDN.
2. **Drag a `.cgb` file** onto the page, or click **Load .cgb** and pick one.
3. Orbit with the left mouse button, pan with the right, zoom with the wheel.

Try `samples/chair.cgb`, `samples/table.cgb`, or `samples/simple_building.cgb`.

## Features

- Renders each primitive (`cube`/`sphere`/`cylinder`/`cone`) with the geometry
  conventions in [cgb-format.md](cgb-format.md) — so what you see matches the
  baker and the Blender import.
- Per-primitive material colors, OrbitControls, grid + axes helpers, automatic
  camera framing of the loaded model.
- A side panel listing every primitive (name, type, color swatch); click an
  entry to highlight and focus it.
- Readable error overlay for invalid JSON or non-`cgb` files; loading a new file
  clears the previous model.

## Notes

- No build step and no server required — everything is inline in the one HTML
  file. three.js is loaded via an ES-module import map.
- `parent` is treated as logical-only (v0.1): every primitive is placed at its
  own world transform; parent transforms are not composed.
