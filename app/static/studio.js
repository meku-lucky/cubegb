// ===========================================================================
// CubeGB Studio — frontend orchestration.
// Wires the image → generate → 3D view → export flow to the FastAPI backend.
//
// The viewport is a 2x2 debug quad: ① the carved voxel solid, ② the final
// primitives, ③④ reserved for future intermediate-stage views. The 3D viewer
// (CGBViewer, which pulls in three.js) is loaded *dynamically* — if it fails,
// the rest of the app (health, .cgb loading, primitive list, export) still works.
// ===========================================================================

const $ = (id) => document.getElementById(id);

// --- state ---
let primViewer = null;       // ② final primitives
let voxelViewer = null;      // ① carved voxels (front colour)
let pureViewer = null;       // ③ pure voxels (flat clay)
let objectViewer = null;     // ④ voxels coloured by SAM object group
let viewerStatus = 'loading'; // 'loading' | 'ready' | 'failed'
let currentDoc = null;       // the .cgb document currently shown
let currentVoxelDoc = null;
let imageFile = null;        // selected input image File
let sheetFile = null;        // selected 2x2 multi-view sheet File
let entries = [];            // primitive entries for the sidebar list

// --- elements ---
const vpPrims = $('vpPrims'), vpVoxel = $('vpVoxel'), vpPure = $('vpPure'), vpObject = $('vpObject');
const emptyPrims = $('emptyPrims'), emptyVoxel = $('emptyVoxel');
const emptyPure = $('emptyPure'), emptyObject = $('emptyObject');
const imgDrop = $('imgDrop'), imgInput = $('imgInput'), imgPreview = $('imgPreview');
const sheetDrop = $('sheetDrop'), sheetInput = $('sheetInput'), sheetPreview = $('sheetPreview');
const genBtn = $('genBtn'), genStatus = $('genStatus');
const loadBtn = $('loadBtn'), cgbInput = $('cgbInput');
const primList = $('primList'), primCount = $('primCount'), emptyMsg = $('emptyMsg');
const dlCgb = $('dlCgb'), dlGlb = $('dlGlb'), dlObj = $('dlObj'), exportName = $('exportName');
const capNote = $('capNote');
const toast = $('toast'), toastMsg = $('toastMsg');

// ---------------------------------------------------------------------------
// three-free helpers (work even if the 3D viewer fails to load)
// ---------------------------------------------------------------------------
function colorToHex(arr) {
  const c = Array.isArray(arr) ? arr : [0.7, 0.7, 0.72];
  const h = (v) => Math.max(0, Math.min(255, Math.round((v || 0) * 255)))
    .toString(16).padStart(2, '0');
  return '#' + h(c[0]) + h(c[1]) + h(c[2]);
}
function validateDocLocal(doc) {
  if (doc == null || typeof doc !== 'object') throw new Error('JSON 객체가 아닙니다.');
  if (doc.format !== 'cgb') {
    throw new Error('CubeGB 파일이 아닙니다: "format":"cgb"가 필요하지만 '
      + JSON.stringify(doc.format) + ' 입니다.');
  }
  if (!Array.isArray(doc.primitives)) throw new Error('잘못된 .cgb: "primitives"는 배열이어야 합니다.');
}
function entriesFromDoc(doc) {
  return doc.primitives.map((p, i) => ({
    id: p.id != null ? p.id : ('idx_' + i),
    name: p.name || p.id || ('primitive_' + i),
    type: p.type, colorHex: colorToHex(p.material && p.material.color), mesh: null,
  }));
}

// ---------------------------------------------------------------------------
// Toast / status helpers
// ---------------------------------------------------------------------------
function showError(msg) { toastMsg.textContent = msg; toast.classList.add('show'); }
$('toastClose').addEventListener('click', () => toast.classList.remove('show'));
function setStatus(html) { genStatus.innerHTML = html; }

// ---------------------------------------------------------------------------
// Lazy 3D viewers — dynamic import so a failure never breaks the app.
// ---------------------------------------------------------------------------
async function initViewers() {
  try {
    const mod = await import('/static/cgb-render.js'); // pulls in three.js
    primViewer = new mod.CGBViewer(vpPrims, {
      onSelect: (id) => {
        [...primList.children].forEach((li) => li.classList.toggle('active', li.dataset.id === id));
      },
    });
    voxelViewer = new mod.CGBViewer(vpVoxel, { background: 0x121519 });
    pureViewer = new mod.CGBViewer(vpPure, { background: 0x121519 });
    objectViewer = new mod.CGBViewer(vpObject, { background: 0x121519 });
    viewerStatus = 'ready';
    if (currentDoc) renderDocs();  // a doc arrived before the viewers were ready
  } catch (e) {
    viewerStatus = 'failed';
    emptyPrims.innerHTML = '3D 미리보기를 불러오지 못했습니다(오프라인/차단).<br>목록과 내보내기는 정상 사용 가능합니다.';
    emptyPrims.style.display = 'flex';
    console.error('CGBViewer load failed:', e);
  }
}

// ---------------------------------------------------------------------------
// Capabilities
// ---------------------------------------------------------------------------
async function checkHealth() {
  try {
    const h = await (await fetch('/api/health')).json();
    if (h.default_sam_checkpoint) $('samCkpt').value = h.default_sam_checkpoint;
    if (h.default_depth_checkpoint) $('depthCkpt').value = h.default_depth_checkpoint;
    if (h.recognition_available) {
      capNote.innerHTML = '<span class="ok">●</span> 인식 파이프라인 사용 가능' +
        (h.default_sam_checkpoint_exists ? ' · SAM 체크포인트 감지됨'
          : ' · <span class="warn">SAM 체크포인트 경로를 생성 옵션에 입력하세요</span>');
    } else {
      capNote.innerHTML = '<span class="warn">●</span> 인식(torch/SAM) 미설치 — ' +
        '이미지 생성은 비활성. <b>.cgb 불러오기 → 보기 → 내보내기</b>는 사용 가능합니다.' +
        '<br>설치: <code>pip install -r requirements-recognition.txt</code>';
    }
  } catch (e) {
    capNote.innerHTML = '<span class="err">●</span> 서버 상태 확인 실패: ' + e.message;
  }
}

checkHealth();
initViewers();

// ---------------------------------------------------------------------------
// Step 1/2 — image & multi-view sheet selection
// ---------------------------------------------------------------------------
function wireDrop(drop, input, preview, onPick) {
  drop.addEventListener('click', () => input.click());
  input.addEventListener('change', (e) => { if (e.target.files[0]) onPick(e.target.files[0]); input.value = ''; });
  ['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => {
    e.preventDefault(); drop.classList.add('drag');
  }));
  ['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => {
    e.preventDefault(); drop.classList.remove('drag');
  }));
  drop.addEventListener('drop', (e) => {
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f && f.type.startsWith('image/')) onPick(f);
  });
  function onPickInternal(file) { preview.src = URL.createObjectURL(file); preview.style.display = 'block'; }
  return onPickInternal;
}
const showImg = wireDrop(imgDrop, imgInput, imgPreview, (f) => { imageFile = f; showImg(f); genBtn.disabled = false; });
const showSheet = wireDrop(sheetDrop, sheetInput, sheetPreview, (f) => { sheetFile = f; showSheet(f); genBtn.disabled = false; });

// ---------------------------------------------------------------------------
// Step 3 — generate OR load an existing .cgb
// ---------------------------------------------------------------------------
genBtn.addEventListener('click', async () => {
  if (!imageFile && !sheetFile) { showError('이미지 또는 멀티뷰 시트를 선택하세요.'); return; }
  toast.classList.remove('show');
  genBtn.disabled = true;
  const mode = sheetFile ? '멀티뷰(정밀)' : '단일 이미지';
  setStatus(`<span class="spinner"></span><span class="muted">생성 중… (${mode}, 수십 초 소요 가능)</span>`);

  const fd = new FormData();
  if (imageFile) fd.append('image', imageFile);
  if (sheetFile) fd.append('sheet', sheetFile);
  fd.append('sam_checkpoint', $('samCkpt').value);
  fd.append('depth_checkpoint', $('depthCkpt').value);
  fd.append('device', $('device').value);
  fd.append('sam_model_type', $('samType').value);
  fd.append('max_segments', $('maxSeg').value);
  fd.append('voxel_res', $('voxelRes').value);

  try {
    const res = await fetch('/api/generate', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || ('HTTP ' + res.status));
    applyDoc(data.cgb, data.voxel_cgb || null);
    const vx = data.summary && data.summary.voxels;
    setStatus('<span class="ok">완료</span> · 프리미티브 ' + data.cgb.primitives.length + '개'
      + (vx ? ` · 복셀 ${vx.toLocaleString()}개` : ''));
  } catch (e) {
    setStatus('<span class="err">실패</span>');
    showError(e.message);
  } finally {
    genBtn.disabled = false;
  }
});

loadBtn.addEventListener('click', () => cgbInput.click());
cgbInput.addEventListener('change', (e) => {
  const f = e.target.files[0]; cgbInput.value = '';
  if (!f) return;
  const reader = new FileReader();
  reader.onload = (ev) => {
    try {
      const doc = JSON.parse(ev.target.result);
      applyDoc(doc, null);
      const base = f.name.replace(/\.cgb$/i, '');
      if (base) exportName.value = base;
      setStatus('<span class="ok">불러옴</span> · ' + f.name);
    } catch (err) { showError('"' + f.name + '" 파싱 실패: ' + err.message); }
  };
  reader.readAsText(f);
});

// ---------------------------------------------------------------------------
// Apply docs to the viewers + UI (works with or without the 3D viewers)
// ---------------------------------------------------------------------------
function applyDoc(doc, voxelDoc) {
  try { validateDocLocal(doc); }
  catch (e) { showError('표시할 수 없는 .cgb: ' + e.message); return; }
  currentDoc = doc;
  currentVoxelDoc = voxelDoc || null;
  entries = entriesFromDoc(doc);
  renderDocs();
  rebuildList();
  dlCgb.disabled = dlGlb.disabled = dlObj.disabled = false;
}

function renderDocs() {
  // Panel ② final primitives
  if (primViewer && currentDoc) {
    try { entries = primViewer.loadDoc(currentDoc); emptyPrims.style.display = 'none'; }
    catch (e) { showError('3D 렌더 실패(목록·내보내기는 가능): ' + e.message); }
  }
  // Panels ①③④ — the same voxel doc rendered three ways (front / pure / object)
  const voxelPanels = [
    [voxelViewer, emptyVoxel, 'front'],
    [pureViewer, emptyPure, 'pure'],
    [objectViewer, emptyObject, 'object'],
  ];
  for (const [v, empty, mode] of voxelPanels) {
    if (!v) continue;
    if (currentVoxelDoc) {
      try { v.loadVoxels(currentVoxelDoc, mode); empty.style.display = 'none'; }
      catch (e) { console.error('voxel render failed', e); }
    } else {
      v.clearVoxels();
      empty.style.display = 'flex';
    }
  }
}

function rebuildList() {
  primList.innerHTML = '';
  if (!entries.length) { emptyMsg.style.display = 'block'; primCount.textContent = ''; return; }
  emptyMsg.style.display = 'none';
  primCount.textContent = '(' + entries.length + ')';
  entries.forEach((entry) => {
    const li = document.createElement('li');
    li.dataset.id = entry.id;
    const sw = document.createElement('span');
    sw.className = 'swatch'; sw.style.background = entry.colorHex;
    const meta = document.createElement('div');
    const nm = document.createElement('div'); nm.className = 'name'; nm.textContent = entry.name;
    const tp = document.createElement('div'); tp.className = 'type'; tp.textContent = entry.type;
    meta.appendChild(nm); meta.appendChild(tp);
    li.appendChild(sw); li.appendChild(meta);
    if (primViewer) li.addEventListener('click', () => primViewer.select(entry.id));
    primList.appendChild(li);
  });
}

// ---------------------------------------------------------------------------
// Step 4 — export
// ---------------------------------------------------------------------------
function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

dlCgb.addEventListener('click', () => {
  if (!currentDoc) return;
  downloadBlob(new Blob([JSON.stringify(currentDoc, null, 2)], { type: 'application/json' }),
    (exportName.value || 'cubegb') + '.cgb');
});

async function bake(fmt, btn) {
  if (!currentDoc) return;
  const label = btn.textContent;
  btn.disabled = true; btn.textContent = '…';
  try {
    const res = await fetch('/api/bake', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ doc: currentDoc, format: fmt, segments: 0, filename: exportName.value || 'cubegb' }),
    });
    if (!res.ok) {
      let detail = 'HTTP ' + res.status;
      try { detail = (await res.json()).detail || detail; } catch (_) {}
      throw new Error(detail);
    }
    downloadBlob(await res.blob(), (exportName.value || 'cubegb') + '.' + fmt);
  } catch (e) { showError('내보내기 실패: ' + e.message); }
  finally { btn.disabled = false; btn.textContent = label; }
}
dlGlb.addEventListener('click', () => bake('glb', dlGlb));
dlObj.addEventListener('click', () => bake('obj', dlObj));
