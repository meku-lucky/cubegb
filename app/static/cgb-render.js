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

function buildGeometry(type, params) {
  params = params || {};
  const segments = (params.segments != null) ? params.segments : 16;
  switch (type) {
    case 'cube': {
      const s = params.size || [1, 1, 1];
      return new THREE.BoxGeometry(s[0], s[1], s[2]);
    }
    case 'sphere': {
      const r = params.radius != null ? params.radius : 0.5;
      return new THREE.SphereGeometry(r, segments, Math.max(3, Math.floor(segments / 2)));
    }
    case 'cylinder': {
      const r = params.radius != null ? params.radius : 0.5;
      const h = params.height != null ? params.height : 1;
      return new THREE.CylinderGeometry(r, r, h, segments);
    }
    case 'cone': {
      const r = params.radius != null ? params.radius : 0.5;
      const h = params.height != null ? params.height : 1;
      return new THREE.ConeGeometry(r, h, segments); // centered, apex +Y
    }
    default:
      throw new Error('Unknown primitive type: "' + type + '"');
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
      const geom = buildGeometry(prim.type, prim.params);
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

      const edges = new THREE.EdgesGeometry(geom, 25);
      mesh.add(new THREE.LineSegments(edges,
        new THREE.LineBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.22 })));

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

  frame() {
    if (!this.primEntries.length) return;
    const box = new THREE.Box3().setFromObject(this.modelGroup);
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
