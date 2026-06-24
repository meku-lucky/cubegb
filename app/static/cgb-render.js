// ===========================================================================
// CGBViewer — a reusable three.js renderer for .cgb documents.
//
// Geometry conventions MUST match the Python baker and the standalone viewer
// (see docs/cgb-format.md): Y-up, right-handed, meters; transform order
// scale -> rotate(XYZ radians) -> translate; positions are world-space; `parent`
// is logical-only (no transform composition).
//
// Served over HTTP by CubeGB Studio, so ES-module imports work. (The standalone
// viewer/index.html keeps an inline copy because it is opened via file://, where
// local module imports are blocked.)
// ===========================================================================
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const DEFAULT_COLOR = [0.7, 0.7, 0.72];
const VOXEL_CLAY = [0.66, 0.70, 0.76];

// Per-voxel colour for the voxel panels.
//   'multiview' → material.color (sampled from the view facing each voxel)
//   'front'     → the front-only colour, stored as a hex in prim.name
//   'object'    → distinct hue per SAM object group (material.name "objN")
//   'pure'      → flat clay (shape only)
function voxelColor(out, prim, mode) {
  if (mode === 'pure') return out.setRGB(VOXEL_CLAY[0], VOXEL_CLAY[1], VOXEL_CLAY[2]);
  if (mode === 'object') {
    const name = (prim.material && prim.material.name) || 'bg';
    const m = /^obj(\d+)$/.exec(name);
    if (!m) return out.setRGB(0.40, 0.42, 0.46);          // background / ungrouped
    const id = parseInt(m[1], 10);
    return out.setHSL((id * 0.6180339887) % 1, 0.62, 0.56); // golden-ratio distinct hues
  }
  if (mode === 'front') {
    const h = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(prim.name || '');
    if (h) return out.setRGB(parseInt(h[1], 16) / 255, parseInt(h[2], 16) / 255, parseInt(h[3], 16) / 255);
  }
  const c = (prim.material && prim.material.color) || DEFAULT_COLOR; // 'multiview' / fallback
  return out.setRGB(c[0], c[1], c[2]);
}

// Partial sweep (matches bake/baker.py:_sweep_params). Returns null for a full
// 0..360 sweep so the native three.js geometry is used unchanged.
function sweepParams(params) {
  if (params.sweep_start == null && params.sweep_end == null) return null;
  const start = (params.sweep_start != null) ? params.sweep_start : 0;
  const end = (params.sweep_end != null) ? params.sweep_end : 360;
  if (Math.abs((end - start) - 360) <= 1e-6) return null;
  const caps = (params.sweep_caps != null) ? !!params.sweep_caps : true;
  return { start: start * Math.PI / 180, length: (end - start) * Math.PI / 180, caps };
}

// Hand-built wedge geometry — vertex placement matches bake/baker.py exactly
// (x = r*sin(theta), z = r*cos(theta), axis +Y) so the viewer and the baked
// mesh agree on the open arc's direction. A cone is the rTop === 0 case.
function buildSweptGeometry(rBottom, rTop, height, segments, thetaStart, thetaLength, caps) {
  const n = Math.max(2, segments | 0);
  const half = height / 2;
  const EPS = 1e-6;
  const pos = [];
  const idx = [];
  const bot = [];
  const top = [];
  for (let i = 0; i <= n; i++) {
    const theta = thetaStart + (i / n) * thetaLength;
    const sx = Math.sin(theta), cz = Math.cos(theta);
    bot.push(pos.length / 3); pos.push(rBottom * sx, -half, rBottom * cz);
    top.push(pos.length / 3); pos.push(rTop * sx, half, rTop * cz);
  }
  for (let i = 0; i < n; i++) {
    const b0 = bot[i], b1 = bot[i + 1], t0 = top[i], t1 = top[i + 1];
    if (rBottom > EPS) idx.push(b0, b1, t1);
    if (rTop > EPS) idx.push(b0, t1, t0);
    if (rBottom <= EPS && rTop > EPS) idx.push(b0, b1, t1);
  }
  if (rBottom > EPS) {
    const c = pos.length / 3; pos.push(0, -half, 0);
    for (let i = 0; i < n; i++) idx.push(c, bot[i + 1], bot[i]);
  }
  if (rTop > EPS) {
    const c = pos.length / 3; pos.push(0, half, 0);
    for (let i = 0; i < n; i++) idx.push(c, top[i], top[i + 1]);
  }
  if (caps) {
    const ends = [[0, false], [n, true]];
    for (let e = 0; e < ends.length; e++) {
      const i = ends[e][0], flip = ends[e][1];
      const ab = pos.length / 3; pos.push(0, -half, 0);
      const at = pos.length / 3; pos.push(0, half, 0);
      let q = [ab, bot[i], top[i], at];
      if (flip) q = q.slice().reverse();
      idx.push(q[0], q[1], q[2], q[0], q[2], q[3]);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setIndex(idx);
  g.computeVertexNormals();
  return g;
}

// Low-poly chamfered box (24 verts, 44 tris) — mirrors bake/baker.py
// :_beveled_box_mesh exactly, including the convex outward-winding fix.
function buildBeveledBox(size, bevelRatio) {
  const hx = size[0] / 2, hy = size[1] / 2, hz = size[2] / 2;
  let r = Math.min(bevelRatio, 0.5) * Math.min(size[0], size[1], size[2]);
  r = Math.min(r, hx * 0.999, hy * 0.999, hz * 0.999);
  const pos = [];
  const idx = {};
  const SS = [1, -1];
  const key = (sx, sy, sz, a) => sx + ',' + sy + ',' + sz + ',' + a;
  for (const sx of SS) for (const sy of SS) for (const sz of SS) {
    idx[key(sx, sy, sz, 'X')] = pos.length / 3; pos.push(sx * hx, sy * (hy - r), sz * (hz - r));
    idx[key(sx, sy, sz, 'Y')] = pos.length / 3; pos.push(sx * (hx - r), sy * hy, sz * (hz - r));
    idx[key(sx, sy, sz, 'Z')] = pos.length / 3; pos.push(sx * (hx - r), sy * (hy - r), sz * hz);
  }
  const I = (sx, sy, sz, a) => idx[key(sx, sy, sz, a)];
  const tris = [];
  const quad = (a, b, c, d) => { tris.push([a, b, c]); tris.push([a, c, d]); };
  for (const sx of SS) quad(I(sx, 1, 1, 'X'), I(sx, 1, -1, 'X'), I(sx, -1, -1, 'X'), I(sx, -1, 1, 'X'));
  for (const sy of SS) quad(I(1, sy, 1, 'Y'), I(1, sy, -1, 'Y'), I(-1, sy, -1, 'Y'), I(-1, sy, 1, 'Y'));
  for (const sz of SS) quad(I(1, 1, sz, 'Z'), I(1, -1, sz, 'Z'), I(-1, -1, sz, 'Z'), I(-1, 1, sz, 'Z'));
  for (const sx of SS) for (const sy of SS) quad(I(sx, sy, 1, 'X'), I(sx, sy, -1, 'X'), I(sx, sy, -1, 'Y'), I(sx, sy, 1, 'Y'));
  for (const sx of SS) for (const sz of SS) quad(I(sx, 1, sz, 'X'), I(sx, -1, sz, 'X'), I(sx, -1, sz, 'Z'), I(sx, 1, sz, 'Z'));
  for (const sy of SS) for (const sz of SS) quad(I(1, sy, sz, 'Y'), I(-1, sy, sz, 'Y'), I(-1, sy, sz, 'Z'), I(1, sy, sz, 'Z'));
  for (const sx of SS) for (const sy of SS) for (const sz of SS) tris.push([I(sx, sy, sz, 'X'), I(sx, sy, sz, 'Y'), I(sx, sy, sz, 'Z')]);
  const v = (i) => [pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]];
  const out = [];
  for (const [a, b, c] of tris) {
    const A = v(a), B = v(b), C = v(c);
    const nx = (B[1] - A[1]) * (C[2] - A[2]) - (B[2] - A[2]) * (C[1] - A[1]);
    const ny = (B[2] - A[2]) * (C[0] - A[0]) - (B[0] - A[0]) * (C[2] - A[2]);
    const nz = (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0]);
    const dot = nx * (A[0] + B[0] + C[0]) + ny * (A[1] + B[1] + C[1]) + nz * (A[2] + B[2] + C[2]);
    if (dot < 0) out.push(a, c, b); else out.push(a, b, c);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setIndex(out);
  g.computeVertexNormals();
  return g;
}

function buildGeometry(type, params, deform) {
  params = params || {};
  const segments = (params.segments != null) ? params.segments : 16;
  switch (type) {
    case 'cube': {
      const s = params.size || [1, 1, 1];
      const bevel = deform && deform.bevel ? deform.bevel : 0;
      if (bevel > 0) return buildBeveledBox(s, bevel);
      return new THREE.BoxGeometry(s[0], s[1], s[2]);
    }
    case 'sphere': {
      const r = params.radius != null ? params.radius : 0.5;
      return new THREE.SphereGeometry(r, segments, Math.max(3, Math.floor(segments / 2)));
    }
    case 'cylinder': {
      const r = params.radius != null ? params.radius : 0.5;
      const h = params.height != null ? params.height : 1;
      const sw = sweepParams(params);
      if (sw) return buildSweptGeometry(r, r, h, segments, sw.start, sw.length, sw.caps);
      return new THREE.CylinderGeometry(r, r, h, segments);
    }
    case 'cone': {
      const r = params.radius != null ? params.radius : 0.5;
      const h = params.height != null ? params.height : 1;
      const sw = sweepParams(params);
      if (sw) return buildSweptGeometry(r, 0, h, segments, sw.start, sw.length, sw.caps);
      return new THREE.ConeGeometry(r, h, segments); // centered, apex +Y
    }
    default:
      throw new Error('Unknown primitive type: "' + type + '"');
  }
}

// Local-space shape deform (taper along +Y) — identical formula to
// bake/baker.py:_apply_deform so the preview matches the baked mesh.
function applyDeform(geom, deform) {
  if (!deform) return;
  const taper = deform.taper;
  if (taper) {
    const tx = taper[0], tz = taper[1];
    const pos = geom.attributes.position;
    let ymin = Infinity, ymax = -Infinity;
    for (let i = 0; i < pos.count; i++) {
      const y = pos.getY(i);
      if (y < ymin) ymin = y;
      if (y > ymax) ymax = y;
    }
    const h = ymax - ymin;
    if (h > 1e-9) {
      for (let i = 0; i < pos.count; i++) {
        const t = (pos.getY(i) - ymin) / h;
        pos.setX(i, pos.getX(i) * (1 + (tx - 1) * t));
        pos.setZ(i, pos.getZ(i) * (1 + (tz - 1) * t));
      }
      pos.needsUpdate = true;
      geom.computeVertexNormals();
    }
  }
}

export function validateDoc(doc) {
  if (doc == null || typeof doc !== 'object') throw new Error('Not a JSON object.');
  if (doc.format !== 'cgb') {
    throw new Error('Not a CubeGB file: expected "format":"cgb" but got ' +
      JSON.stringify(doc.format) + '.');
  }
  if (!Array.isArray(doc.primitives)) throw new Error('Invalid .cgb: "primitives" must be an array.');
}

export class CGBViewer {
  /**
   * @param {HTMLElement} container  element to fill with the canvas
   * @param {object} [opts]          { onSelect(id|null), background }
   */
  constructor(container, opts = {}) {
    this.container = container;
    this.onSelect = opts.onSelect || (() => {});
    this.edges = opts.edges !== false;   // per-primitive edge overlay (off for dense voxels)
    this.primEntries = [];
    this.activeId = null;

    const w = container.clientWidth || 800;
    const h = container.clientHeight || 600;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(opts.background != null ? opts.background : 0x1a1d23);

    this.camera = new THREE.PerspectiveCamera(50, w / h, 0.01, 5000);
    this.camera.position.set(4, 3, 6);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(w, h);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.target.set(0, 0.5, 0);

    this.scene.add(new THREE.HemisphereLight(0xdfe9ff, 0x2a2620, 0.9));
    const dir = new THREE.DirectionalLight(0xffffff, 1.6);
    dir.position.set(5, 10, 7);
    this.scene.add(dir);
    const fill = new THREE.DirectionalLight(0xbcd2ff, 0.35);
    fill.position.set(-6, 4, -5);
    this.scene.add(fill);

    this.grid = new THREE.GridHelper(20, 20, 0x4a5160, 0x32373f);
    this.grid.material.opacity = 0.6;
    this.grid.material.transparent = true;
    this.scene.add(this.grid);
    this.scene.add(new THREE.AxesHelper(1.5));

    this.modelGroup = new THREE.Group();
    this.scene.add(this.modelGroup);
    this.voxelGroup = null;   // InstancedMesh group for voxel docs

    this._onResize = () => this.resize();
    window.addEventListener('resize', this._onResize);
    this._animate = this._animate.bind(this);
    this._animate();
  }

  resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    if (!w || !h) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  _animate() {
    this._raf = requestAnimationFrame(this._animate);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  /** Build the scene from a .cgb document. Returns the primitive entries. */
  loadDoc(doc) {
    validateDoc(doc);
    this.clear();
    const group = new THREE.Group();
    const entries = [];

    doc.primitives.forEach((prim, idx) => {
      const geom = buildGeometry(prim.type, prim.params, prim.deform);
      applyDeform(geom, prim.deform);
      const colorArr = (prim.material && Array.isArray(prim.material.color))
        ? prim.material.color : DEFAULT_COLOR;
      const color = new THREE.Color(colorArr[0], colorArr[1], colorArr[2]);
      const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.75, metalness: 0.05 });
      const mesh = new THREE.Mesh(geom, mat);
      mesh.name = prim.name || prim.id || ('primitive_' + idx);

      const t = prim.transform || {};
      const pos = t.position || [0, 0, 0];
      const rot = t.rotation_euler || [0, 0, 0];
      const scl = t.scale || [1, 1, 1];
      mesh.rotation.order = 'XYZ';
      mesh.scale.set(scl[0], scl[1], scl[2]);
      mesh.rotation.set(rot[0], rot[1], rot[2]);
      mesh.position.set(pos[0], pos[1], pos[2]);

      if (this.edges) {
        const edges = new THREE.EdgesGeometry(geom, 25);
        mesh.add(new THREE.LineSegments(edges,
          new THREE.LineBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.22 })));
      }

      group.add(mesh);
      entries.push({
        id: prim.id != null ? prim.id : ('idx_' + idx),
        mesh, baseColor: color.clone(), name: mesh.name, type: prim.type,
        colorHex: '#' + color.getHexString(),
      });
    });

    this.modelGroup = group;
    this.scene.add(group);
    this.primEntries = entries;
    this.frame();
    return entries;
  }

  clear() {
    if (this.modelGroup) {
      this.modelGroup.traverse((obj) => {
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
          if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
          else obj.material.dispose();
        }
      });
      this.scene.remove(this.modelGroup);
    }
    this.primEntries = [];
    this.activeId = null;
  }

  clearVoxels() {
    if (this.voxelGroup) {
      this.voxelGroup.geometry.dispose();
      this.voxelGroup.material.dispose();
      this.scene.remove(this.voxelGroup);
      this.voxelGroup = null;
    }
  }

  /**
   * Render a voxel .cgb (many same-size cubes) as a single InstancedMesh — fast
   * enough for thousands of voxels across several panels.
   * @param {object} doc    a .cgb whose primitives are uniform cubes
   * @param {string} mode   'front' (material.color) | 'pure' (flat clay) |
   *                        'object' (palette by material.name "objN" / "bg")
   */
  loadVoxels(doc, mode = 'front') {
    this.clearVoxels();
    const prims = (doc && doc.primitives) || [];
    if (!prims.length) return 0;
    // Unit box + per-instance scale, so voxel docs with mixed cube sizes (e.g.
    // several objects at different scales) render correctly.
    const geo = new THREE.BoxGeometry(1, 1, 1);
    const mat = new THREE.MeshStandardMaterial({ roughness: 0.82, metalness: 0.04 });
    const inst = new THREE.InstancedMesh(geo, mat, prims.length);
    const dummy = new THREE.Object3D();
    const col = new THREE.Color();

    prims.forEach((p, i) => {
      const pos = (p.transform && p.transform.position) || [0, 0, 0];
      const sz = (p.params && p.params.size) || [0.05, 0.05, 0.05];
      dummy.position.set(pos[0], pos[1], pos[2]);
      dummy.scale.set(sz[0], sz[1], sz[2]);
      dummy.updateMatrix();
      inst.setMatrixAt(i, dummy.matrix);
      inst.setColorAt(i, voxelColor(col, p, mode));
    });
    inst.instanceMatrix.needsUpdate = true;
    if (inst.instanceColor) inst.instanceColor.needsUpdate = true;

    this.voxelGroup = inst;
    this.scene.add(inst);
    this._frameObject(inst);
    return prims.length;
  }

  frame() {
    if (!this.primEntries.length) return;
    this._frameObject(this.modelGroup);
  }

  _frameObject(obj) {
    const box = new THREE.Box3().setFromObject(obj);
    if (box.isEmpty()) return;
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z) || 1;
    const fov = this.camera.fov * (Math.PI / 180);
    let dist = (maxDim / 2) / Math.tan(fov / 2) * 1.7;
    const d = new THREE.Vector3(1, 0.8, 1.2).normalize().multiplyScalar(dist);
    this.camera.position.copy(center).add(d);
    this.camera.near = Math.max(0.001, dist / 1000);
    this.camera.far = dist * 1000;
    this.camera.updateProjectionMatrix();
    this.controls.target.copy(center);
    this.controls.update();

    const span = Math.max(4, Math.ceil(maxDim * 2));
    this.scene.remove(this.grid);
    this.grid.geometry.dispose();
    this.grid.material.dispose();
    this.grid = new THREE.GridHelper(span, span, 0x4a5160, 0x32373f);
    this.grid.material.opacity = 0.6;
    this.grid.material.transparent = true;
    this.scene.add(this.grid);
  }

  select(id) {
    if (this.activeId === id) { this.clearSelection(); return; }
    this.clearSelection();
    this.activeId = id;
    const entry = this.primEntries.find(e => e.id === id);
    if (!entry) return;
    entry.mesh.material.emissive = new THREE.Color(0x5b9dff);
    entry.mesh.material.emissiveIntensity = 0.45;
    const box = new THREE.Box3().setFromObject(entry.mesh);
    if (!box.isEmpty()) {
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z) || 0.5;
      const fov = this.camera.fov * (Math.PI / 180);
      const dist = (maxDim / 2) / Math.tan(fov / 2) * 2.4;
      const dir3 = new THREE.Vector3().subVectors(this.camera.position, this.controls.target);
      if (dir3.lengthSq() < 1e-6) dir3.set(1, 0.8, 1.2);
      dir3.normalize().multiplyScalar(dist);
      this.camera.position.copy(center).add(dir3);
      this.controls.target.copy(center);
      this.controls.update();
    }
    this.onSelect(id);
  }

  clearSelection() {
    this.primEntries.forEach((e) => {
      if (e.mesh && e.mesh.material) {
        e.mesh.material.emissive = new THREE.Color(0x000000);
        e.mesh.material.emissiveIntensity = 0;
      }
    });
    this.activeId = null;
    this.onSelect(null);
  }
}
