# Vendored three.js

These files are bundled with CubeGB Studio so the web GUI works **without any
CDN / internet access** (some environments block `unpkg.com` and similar).

- **Library:** [three.js](https://threejs.org/) r165 (`three@0.165.0`)
- **License:** MIT (© three.js authors) — see the header in `three.module.js`.
- **Files:**
  - `three.module.js` — from `three@0.165.0/build/three.module.js`
  - `addons/controls/OrbitControls.js` — from `three@0.165.0/examples/jsm/controls/OrbitControls.js`

The Studio page (`app/static/index.html`) maps the `three` and `three/addons/`
import specifiers to these local files via an import map.

To update: download the matching `build/three.module.js` and
`examples/jsm/controls/OrbitControls.js` for the new version, keeping the same
paths, and bump the version here.

> The standalone viewer (`viewer/index.html`) still loads three.js from a CDN on
> purpose: it is opened directly via `file://` (double-click), where browsers
> block local ES-module imports. Studio is served over HTTP, so it can use these
> vendored modules.
