// ===========================================================================
// CubeGB Studio — frontend orchestration.
// Wires the image → generate → 3D view → export flow to the FastAPI backend,
// reusing CGBViewer (app/static/cgb-render.js) for the 3D view.
// ===========================================================================
import { CGBViewer } from '/static/cgb-render.js';

const $ = (id) => document.getElementById(id);

// --- state ---
let viewer = null;
let currentDoc = null;     // the .cgb document currently shown
let imageFile = null;      // selected input image File
let entries = [];

// --- elements ---
const vp3d = $('vp3d'), vpEmpty = $('vpEmpty');
const imgDrop = $('imgDrop'), imgInput = $('imgInput'), imgPreview = $('imgPreview');
const genBtn = $('genBtn'), genStatus = $('genStatus');
const loadBtn = $('loadBtn'), cgbInput = $('cgbInput');
const primList = $('primList'), primCount = $('primCount'), emptyMsg = $('emptyMsg');
const dlCgb = $('dlCgb'), dlGlb = $('dlGlb'), dlObj = $('dlObj'), exportName = $('exportName');
const capNote = $('capNote');
const toast = $('toast'), toastMsg = $('toastMsg');

// ---------------------------------------------------------------------------
// Viewer init
// ---------------------------------------------------------------------------
viewer = new CGBViewer(vp3d, {
  onSelect: (id) => {
    [...primList.children].forEach((li) => li.classList.toggle('active', li.dataset.id === id));
  },
});

// ---------------------------------------------------------------------------
// Toast / status helpers
// ---------------------------------------------------------------------------
function showError(msg) { toastMsg.textContent = msg; toast.classList.add('show'); }
$('toastClose').addEventListener('click', () => toast.classList.remove('show'));
function setStatus(html) { genStatus.innerHTML = html; }

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

// ---------------------------------------------------------------------------
// Step 1 — image selection
// ---------------------------------------------------------------------------
function pickImage(file) {
  if (!file) return;
  imageFile = file;
  const url = URL.createObjectURL(file);
  imgPreview.src = url;
  imgPreview.style.display = 'block';
  genBtn.disabled = false;
}
imgDrop.addEventListener('click', () => imgInput.click());
imgInput.addEventListener('change', (e) => { pickImage(e.target.files[0]); imgInput.value = ''; });
['dragenter', 'dragover'].forEach((ev) => imgDrop.addEventListener(ev, (e) => {
  e.preventDefault(); imgDrop.classList.add('drag');
}));
['dragleave', 'drop'].forEach((ev) => imgDrop.addEventListener(ev, (e) => {
  e.preventDefault(); imgDrop.classList.remove('drag');
}));
imgDrop.addEventListener('drop', (e) => {
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if (f && f.type.startsWith('image/')) pickImage(f);
});

// ---------------------------------------------------------------------------
// Step 2 — generate (.cgb from image) OR load an existing .cgb
// ---------------------------------------------------------------------------
genBtn.addEventListener('click', async () => {
  if (!imageFile) { showError('먼저 이미지를 선택하세요.'); return; }
  toast.classList.remove('show');
  genBtn.disabled = true;
  setStatus('<span class="spinner"></span><span class="muted">생성 중… (모델 추론은 수십 초 걸릴 수 있습니다)</span>');

  const fd = new FormData();
  fd.append('image', imageFile);
  fd.append('sam_checkpoint', $('samCkpt').value);
  fd.append('depth_checkpoint', $('depthCkpt').value);
  fd.append('device', $('device').value);
  fd.append('sam_model_type', $('samType').value);
  fd.append('max_segments', $('maxSeg').value);

  try {
    const res = await fetch('/api/generate', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || ('HTTP ' + res.status));
    applyDoc(data.cgb);
    setStatus('<span class="ok">완료</span> · 프리미티브 ' + data.cgb.primitives.length + '개');
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
      applyDoc(doc);
      const base = f.name.replace(/\.cgb$/i, '');
      if (base) exportName.value = base;
      setStatus('<span class="ok">불러옴</span> · ' + f.name);
    } catch (err) { showError('"' + f.name + '" 파싱 실패: ' + err.message); }
  };
  reader.readAsText(f);
});

// ---------------------------------------------------------------------------
// Apply a .cgb doc to the viewer + UI
// ---------------------------------------------------------------------------
function applyDoc(doc) {
  try {
    entries = viewer.loadDoc(doc);
  } catch (e) { showError('표시할 수 없는 .cgb: ' + e.message); return; }
  currentDoc = doc;
  vpEmpty.style.display = 'none';
  rebuildList();
  dlCgb.disabled = dlGlb.disabled = dlObj.disabled = false;
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
    li.addEventListener('click', () => viewer.select(entry.id));
    primList.appendChild(li);
  });
}

// ---------------------------------------------------------------------------
// Step 3 — export
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
  const name = (exportName.value || 'cubegb') + '.cgb';
  downloadBlob(new Blob([JSON.stringify(currentDoc, null, 2)], { type: 'application/json' }), name);
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
