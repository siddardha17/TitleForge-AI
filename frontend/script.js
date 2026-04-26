/* ── app init ── */
const API = window.location.origin;
let processingCount = 0;

/* ── DOM refs ── */
const singleInput   = document.getElementById('single-input');
const normalizeBtn  = document.getElementById('normalize-btn');
const clearBtn      = document.getElementById('clear-btn');
const resultBox     = document.getElementById('result-box');
const resultTitle   = document.getElementById('result-title');
const diffBefore    = document.getElementById('diff-before');
const diffAfter     = document.getElementById('diff-after');
const timeMeta      = document.getElementById('time-meta');
const backendMeta   = document.getElementById('backend-meta');
const copyBtn       = document.getElementById('copy-btn');

const uploadZone    = document.getElementById('upload-zone');
const fileInput     = document.getElementById('file-input');
const fileNameEl    = document.getElementById('file-name');
const uploadBtn     = document.getElementById('upload-btn');
const progressWrap  = document.getElementById('progress-wrap');
const progressBar   = document.getElementById('progress-bar');
const bulkResult    = document.getElementById('bulk-result');

const statCount     = document.getElementById('stat-count');
const statBackend   = document.getElementById('stat-backend');
const statStatus    = document.getElementById('stat-status');

const toast         = document.getElementById('toast');

/* ── Tabs ── */
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab, .tab-panel').forEach(el => el.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById(tab.dataset.panel).classList.add('active');
  });
});

/* ── Toast ── */
let toastTimer;
function showToast(msg, type = 'info') {
  const icon = type === 'success' ? '✅' : type === 'error' ? '❌' : 'ℹ️';
  toast.innerHTML = `<span>${icon}</span> ${msg}`;
  toast.className = `toast show ${type}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), 3500);
}

/* ── Health check on load ── */
async function checkHealth() {
  try {
    const r  = await fetch(`${API}/health`);
    const d  = await r.json();
    const ok = d.status === 'ok';
    statStatus.textContent  = ok ? (d.model_loaded ? '✅ Ready' : '⚠ Fallback') : '❌ Down';
    statBackend.textContent = d.model_backend || 'unknown';
  } catch {
    statStatus.textContent = '❌ Offline';
  }
}
checkHealth();

/* ── Example items ── */
document.querySelectorAll('.example-item').forEach(el => {
  el.addEventListener('click', () => {
    singleInput.value = el.dataset.raw;
    singleInput.focus();
    document.querySelector('[data-panel="single-panel"]').click();
  });
});

/* ── Single normalize ── */
normalizeBtn.addEventListener('click', async () => {
  const raw = singleInput.value.trim();
  if (!raw) { showToast('Please enter a product title.', 'error'); return; }

  setLoading(normalizeBtn, true);
  resultBox.classList.remove('visible');

  try {
    const res  = await fetch(`${API}/normalize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ raw_title: raw }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    processingCount++;
    statCount.textContent = processingCount;

    resultTitle.textContent = data.normalized_title;
    diffBefore.textContent  = data.raw_title;
    diffAfter.textContent   = data.normalized_title;
    timeMeta.querySelector('span').textContent    = `${data.processing_time_ms} ms`;
    backendMeta.querySelector('span').textContent = data.model_backend;

    resultBox.className = 'result-box result-success visible';
    showToast('Title normalized successfully!', 'success');

  } catch (err) {
    resultBox.innerHTML = `
      <div class="result-error">
        <div class="result-label">Error</div>
        <div class="result-title" style="color:var(--error)">${err.message}</div>
      </div>`;
    resultBox.className = 'result-box visible';
    showToast(err.message, 'error');
  } finally {
    setLoading(normalizeBtn, false);
  }
});

/* ── Clear button ── */
clearBtn.addEventListener('click', () => {
  singleInput.value = '';
  resultBox.classList.remove('visible');
  singleInput.focus();
});

/* ── Copy button ── */
copyBtn.addEventListener('click', async () => {
  const text = resultTitle.textContent;
  if (!text) return;
  await navigator.clipboard.writeText(text);
  showToast('Copied to clipboard!', 'success');
});

/* ── File upload ── */
uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f) handleFileSelected(f);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFileSelected(fileInput.files[0]);
});

function handleFileSelected(file) {
  if (!file.name.endsWith('.csv')) {
    showToast('Please select a CSV file.', 'error'); return;
  }
  fileInput._selectedFile = file;
  fileNameEl.textContent = `📄 ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  uploadBtn.disabled = false;
}

uploadBtn.addEventListener('click', async () => {
  const file = fileInput._selectedFile || fileInput.files[0];
  if (!file) { showToast('Please select a CSV file first.', 'error'); return; }

  setLoading(uploadBtn, true);
  progressWrap.classList.add('visible');
  progressBar.style.width = '10%';
  bulkResult.style.display = 'none';

  const formData = new FormData();
  formData.append('file', file);

  // Simulate progress
  let pct = 10;
  const ticker = setInterval(() => {
    pct = Math.min(pct + 8, 85);
    progressBar.style.width = `${pct}%`;
  }, 400);

  try {
    const res = await fetch(`${API}/bulk-normalize`, { method: 'POST', body: formData });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    clearInterval(ticker);
    progressBar.style.width = '100%';

    const blob     = await res.blob();
    const url      = URL.createObjectURL(blob);
    const a        = document.createElement('a');
    a.href         = url;
    a.download     = `normalized_${file.name}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    bulkResult.style.display = 'block';
    bulkResult.innerHTML = `
      <div class="result-success" style="border-radius:var(--radius-md);padding:1rem 1.25rem;">
        <div class="result-label">✅ Bulk processing complete</div>
        <div style="font-size:0.88rem;color:var(--text-secondary);margin-top:0.3rem">
          Your normalized CSV has been downloaded.
        </div>
      </div>`;
    showToast('Bulk normalization complete!', 'success');

  } catch (err) {
    clearInterval(ticker);
    progressBar.style.width = '0%';
    bulkResult.style.display = 'block';
    bulkResult.innerHTML = `
      <div class="result-error" style="border-radius:var(--radius-md);padding:1rem 1.25rem;">
        <div class="result-label">❌ Error</div>
        <div style="font-size:0.88rem;color:var(--error);">${err.message}</div>
      </div>`;
    showToast(err.message, 'error');
  } finally {
    setLoading(uploadBtn, false);
    setTimeout(() => progressWrap.classList.remove('visible'), 800);
  }
});

/* ── Keyboard shortcut: Ctrl+Enter to normalize ── */
singleInput.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') normalizeBtn.click();
});

/* ── Helper: set loading state ── */
function setLoading(btn, state) {
  if (state) {
    btn.classList.add('loading');
    btn.disabled = true;
  } else {
    btn.classList.remove('loading');
    btn.disabled = false;
  }
}
