// ══════════════════════════════════════════════════════
//  THREE.JS BACKGROUND — particle field with connections
// ══════════════════════════════════════════════════════
(function initThree() {
  const canvas   = document.getElementById('bg');
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setClearColor(0x060410, 1);

  const scene  = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(70, innerWidth / innerHeight, 0.1, 1000);
  camera.position.z = 60;

  const N = 220;
  const positions  = new Float32Array(N * 3);
  const velocities = [];

  for (let i = 0; i < N; i++) {
    positions[i*3]   = (Math.random() - 0.5) * 150;
    positions[i*3+1] = (Math.random() - 0.5) * 100;
    positions[i*3+2] = (Math.random() - 0.5) * 40;
    velocities.push({
      x: (Math.random() - 0.5) * 0.012,
      y: (Math.random() - 0.5) * 0.012,
    });
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const mat = new THREE.PointsMaterial({ color: 0x5522bb, size: 0.45, transparent: true, opacity: 0.65 });
  const points = new THREE.Points(geo, mat);
  scene.add(points);

  const lineMat = new THREE.LineBasicMaterial({ color: 0x3311aa, transparent: true, opacity: 0.12 });
  let linesMesh = null;
  let frame = 0;

  function buildLines() {
    if (linesMesh) { scene.remove(linesMesh); linesMesh.geometry.dispose(); }
    const lp = [];
    const THRESH = 20;
    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) {
        const dx = positions[i*3] - positions[j*3];
        const dy = positions[i*3+1] - positions[j*3+1];
        if (dx*dx + dy*dy < THRESH*THRESH) {
          lp.push(positions[i*3], positions[i*3+1], positions[i*3+2],
                  positions[j*3], positions[j*3+1], positions[j*3+2]);
        }
      }
    }
    const lg = new THREE.BufferGeometry();
    lg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(lp), 3));
    linesMesh = new THREE.LineSegments(lg, lineMat);
    scene.add(linesMesh);
  }

  function animate() {
    requestAnimationFrame(animate);
    frame++;
    for (let i = 0; i < N; i++) {
      positions[i*3]   += velocities[i].x;
      positions[i*3+1] += velocities[i].y;
      if (Math.abs(positions[i*3])   > 75) velocities[i].x *= -1;
      if (Math.abs(positions[i*3+1]) > 50) velocities[i].y *= -1;
    }
    geo.attributes.position.needsUpdate = true;
    if (frame % 5 === 0) buildLines();
    scene.rotation.y = Math.sin(Date.now() * 0.00007) * 0.06;
    renderer.render(scene, camera);
  }
  animate();

  window.addEventListener('resize', () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  });
  renderer.setSize(innerWidth, innerHeight);
})();


// ══════════════════════════════════════════════════════
//  CLOCK
// ══════════════════════════════════════════════════════
function updateClock() {
  const el = document.getElementById('systemClock');
  if (!el) return;
  el.textContent = new Date().toLocaleTimeString('en-US', { hour12: false });
}
updateClock();
setInterval(updateClock, 1000);


// ══════════════════════════════════════════════════════
//  CHAR COUNT
// ══════════════════════════════════════════════════════
const inputText = document.getElementById('inputText');
const charCountEl = document.getElementById('charCount');
inputText.addEventListener('input', () => {
  charCountEl.textContent = inputText.value.length;
});


// ══════════════════════════════════════════════════════
//  SESSION STATS
// ══════════════════════════════════════════════════════
let session = { total: 0, toxic: 0, safe: 0, scores: [], categoryTotals: {} };

function updateStats(results) {
  session.total++;
  const hasToxic = Object.values(results).some(r => r.prediction === 1);
  if (hasToxic) session.toxic++; else session.safe++;

  Object.keys(results).forEach(k => {
    if (!session.categoryTotals[k]) session.categoryTotals[k] = { sum: 0, count: 0 };
    session.categoryTotals[k].sum   += results[k].confidence;
    session.categoryTotals[k].count += 1;
  });

  const maxConf = Math.max(...Object.values(results).map(r => r.prediction === 1 ? r.confidence : 0));
  session.scores.push(maxConf);

  animateCounter('statTotal', session.total);
  animateCounter('statToxic', session.toxic);
  animateCounter('statSafe',  session.safe);
  const rate = ((session.toxic / session.total) * 100).toFixed(0);
  document.getElementById('statRate').textContent = session.total ? rate + '%' : '—';
  document.getElementById('sessionCount').textContent =
    `SESSION: ${session.total} SCAN${session.total !== 1 ? 'S' : ''}`;

  animateGauge(maxConf * 100);
  renderBreakdown();
  renderSparkline();
}

function animateCounter(id, target) {
  const el = document.getElementById(id);
  if (!el) return;
  let cur = parseInt(el.textContent) || 0;
  const step = () => {
    cur = Math.min(cur + 1, target);
    el.textContent = cur;
    if (cur < target) requestAnimationFrame(step);
  };
  step();
}

function animateGauge(pct) {
  const arc    = document.getElementById('gaugeArc');
  const needle = document.getElementById('gaugeNeedle');
  const valEl  = document.getElementById('gaugeVal');
  const totalLen = 251.2;
  arc.style.transition = 'stroke-dashoffset 1s cubic-bezier(0.22,1,0.36,1)';
  arc.style.strokeDashoffset = totalLen * (1 - pct / 100);
  const deg = -90 + (pct / 100) * 180;
  needle.style.transition = 'transform 1s cubic-bezier(0.22,1,0.36,1)';
  needle.setAttribute('transform', `rotate(${deg}, 100, 110)`);

  let cur = 0;
  const tgt = Math.round(pct);
  const step = () => {
    cur = Math.min(cur + Math.ceil((tgt - cur) / 5) || 1, tgt);
    valEl.textContent = cur + '%';
    if (cur < tgt) requestAnimationFrame(step);
  };
  step();
}

function renderBreakdown() {
  const el = document.getElementById('categoryBreakdown');
  el.innerHTML = '';
  const cats = Object.keys(session.categoryTotals).sort((a, b) => {
    const avgA = session.categoryTotals[a].sum / session.categoryTotals[a].count;
    const avgB = session.categoryTotals[b].sum / session.categoryTotals[b].count;
    return avgB - avgA;
  });
  if (cats.length === 0) {
    el.innerHTML = '<div class="breakdown-empty">Run an analysis to see breakdown</div>';
    return;
  }
  cats.forEach((k, i) => {
    const avg = session.categoryTotals[k].sum / session.categoryTotals[k].count;
    const pct = (avg * 100).toFixed(0);
    const isToxic = avg > 0.5;
    const item = document.createElement('div');
    item.className = 'breakdown-item';
    item.style.animationDelay = `${i * 0.05}s`;
    item.innerHTML = `
      <div class="breakdown-row">
        <span class="breakdown-name">${escapeHTML(k)}</span>
        <span class="breakdown-pct" style="color:${isToxic ? 'var(--accent)' : 'var(--primary)'}">${pct}%</span>
      </div>
      <div class="breakdown-bar">
        <div class="breakdown-fill ${isToxic ? 'toxic-fill' : 'safe-fill'}" style="width:0%" data-w="${pct}"></div>
      </div>`;
    el.appendChild(item);
    requestAnimationFrame(() => requestAnimationFrame(() => {
      item.querySelector('.breakdown-fill').style.width = pct + '%';
    }));
  });
}

function renderSparkline() {
  const scores = session.scores.slice(-30);
  if (scores.length < 2) return;
  const W = 260, H = 50;
  const pts = scores.map((v, i) => {
    const x = (i / (scores.length - 1)) * W;
    const y = H - v * (H - 8) - 4;
    return [x, y];
  });
  const d    = pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ');
  const fill = d + ` L${W},${H} L0,${H} Z`;
  document.getElementById('sparkLine').setAttribute('d', d);
  document.getElementById('sparkPath').setAttribute('d', fill);
}


// ══════════════════════════════════════════════════════
//  HISTORY
// ══════════════════════════════════════════════════════
let history = [];

function saveToHistory(text, data) {
  history.unshift({ id: Date.now(), text, results: data.results, timestamp: new Date() });
  renderHistory();
  document.getElementById('histCount').textContent = history.length;
}

function renderHistory() {
  const list  = document.getElementById('historyList');
  const empty = document.getElementById('historyEmpty');
  list.querySelectorAll('.history-card').forEach(el => el.remove());

  if (history.length === 0) {
    empty.style.display = 'flex';
    document.getElementById('histCount').textContent = 0;
    return;
  }
  empty.style.display = 'none';

  history.forEach((entry, idx) => {
    const results  = entry.results || {};
    const hasToxic = Object.values(results).some(r => r.prediction === 1);
    const card = document.createElement('div');
    card.className = `history-card ${hasToxic ? 'has-toxic' : 'all-safe'}`;
    card.style.animationDelay = `${idx * 0.04}s`;

    const timeStr     = entry.timestamp.toLocaleTimeString('en-US', { hour12: false });
    const toxicLabels = Object.keys(results).filter(k => results[k].prediction === 1);
    const pillsHTML   = toxicLabels.slice(0, 3).map(k =>
      `<span class="pill toxic-pill">${escapeHTML(k)} ${(results[k].confidence * 100).toFixed(0)}%</span>`
    ).join('');

    card.innerHTML = `
      <div class="card-text">${escapeHTML(entry.text)}</div>
      <div class="card-meta">
        <span class="card-time">${timeStr}</span>
        <span class="card-status ${hasToxic ? 'toxic-status' : 'safe-status'}">${hasToxic ? '⚠ FLAGGED' : '✓ CLEAN'}</span>
      </div>
      ${pillsHTML ? `<div class="card-pills">${pillsHTML}</div>` : ''}`;

    card.addEventListener('click', () => {
      inputText.value = entry.text;
      charCountEl.textContent = entry.text.length;
      renderResults({ results: entry.results });
      renderHeatmap(entry.text, entry.results);
      document.querySelector('.main-panel').scrollTop = 0;
      showToast('↩ LOADED FROM SCAN LOG');
    });
    list.appendChild(card);
  });
}

document.getElementById('clearBtn').addEventListener('click', () => {
  if (!history.length) return;
  history = [];
  renderHistory();
  showToast('⊘ SCAN LOG PURGED');
});


// ══════════════════════════════════════════════════════
//  WORD HEATMAP
// ══════════════════════════════════════════════════════
const TOXIC_WORDS = new Set([
  'fuck','fucking','bitch','bastard','idiot',
  'stupid','retard','faggot','kill','die',
  'garbage','loser','ugly','scum','filth','pig','monster',
  'racist','sexist','bigot','slur','nigga','nigger','slut',
  'whore','nazi','curb stomp','stabbed','killed','swine',
  'fuck','fucking','motherfucker','bitch','bastard',
  'abuse','attack','threaten','bully','harass','rape'
]);

function renderHeatmap(text, results) {
  const section   = document.getElementById('heatmapSection');
  const container = document.getElementById('heatmap');

  if (!section || !container) return;

  section.style.display = 'block';
  container.innerHTML = '';

  // Safe handling of results
  const values = Object.values(results || {});
  const toxicScore = values.length
    ? Math.max(...values.map(r => r.prediction === 1 ? r.confidence : 0))
    : 0;

  const words = text.split(/(\s+)/);

  words.forEach((token, i) => {

    // Preserve spaces
    if (/^\s+$/.test(token)) {
      container.appendChild(document.createTextNode(token));
      return;
    }

    const clean = token.toLowerCase().replace(/[^a-z]/g, '');
    const isToxic = TOXIC_WORDS.has(clean);

    const span = document.createElement('span');
    span.className = 'word';
    span.textContent = token;

    // ---------- FIXED HEAT LOGIC ----------
    let heat = 0;

    if (isToxic) {
      heat = 0.85; // strong, consistent highlight
    } else if (toxicScore > 0.75) {
      // very subtle context effect (optional)
      heat = 0.12;
    }

    // ---------- APPLY STYLES ----------
    if (heat > 0) {
      span.style.background = `rgba(255, 0, 80, ${heat * 0.35})`;
      span.style.color = `rgba(255, ${180 - heat*100}, ${180 - heat*100}, 1)`;
      span.style.padding = '2px 4px';
      span.style.borderRadius = '3px';
      span.title = isToxic
        ? '⚠ Toxic word detected'
        : 'Contextual risk';
    } else {
      span.style.color = 'rgba(220,228,245,0.85)';
    }

    // ---------- CLEAN ANIMATION ----------
    span.style.opacity = '0';
    span.style.transform = 'translateY(3px)';
    span.style.transition = `opacity 0.25s ${i * 0.01}s, transform 0.25s ${i * 0.01}s`;

    container.appendChild(span);

    requestAnimationFrame(() => {
      span.style.opacity = '1';
      span.style.transform = 'translateY(0)';
    });
  });
}

// ══════════════════════════════════════════════════════
//  LOADER ANIMATION
// ══════════════════════════════════════════════════════
const loader  = document.getElementById('loader');
const button  = document.getElementById('analyzeBtn');
let loaderIv  = null;

const LOADER_STEPS = [
  { id: 'lstep1', label: 'TOKENIZING INPUT',      threshold: 0  },
  { id: 'lstep2', label: 'RUNNING CLASSIFIER',    threshold: 35 },
  { id: 'lstep3', label: 'AGGREGATING SIGNALS',   threshold: 75 },
];

function startLoader() {
  loader.classList.add('visible');
  button.disabled = true;
  const bar    = document.querySelector('.loader-progress');
  const pctEl  = document.getElementById('loaderPct');
  const labelEl = document.getElementById('loaderLabel');
  LOADER_STEPS.forEach(s => {
    const el = document.getElementById(s.id);
    el.classList.remove('active','done');
  });
  let pct = 0;
  loaderIv = setInterval(() => {
    pct = Math.min(pct + Math.random() * 3.5 + 0.5, 92);
    bar.style.width  = pct + '%';
    pctEl.textContent = Math.round(pct) + '%';
    LOADER_STEPS.forEach((s, i) => {
      const el = document.getElementById(s.id);
      if (pct >= s.threshold) {
        el.classList.add('active');
        labelEl.textContent = s.label;
        if (i > 0) {
          const prev = document.getElementById(LOADER_STEPS[i-1].id);
          prev.classList.remove('active');
          prev.classList.add('done');
        }
      }
    });
  }, 100);

  // Cold-start notice after 5 seconds
  loaderIv._coldStartTimer = setTimeout(() => {
    const waitMsg = document.createElement('div');
    waitMsg.id = 'coldStartMsg';
    waitMsg.style.cssText = 'font-size:9px;letter-spacing:0.12em;color:rgba(168,85,247,0.7);margin-top:4px;animation:fadeSlide 0.4s both';
    waitMsg.textContent = '⟳ PLEASE WAIT — BACKEND MAY BE COLD-STARTING...';
    loader.appendChild(waitMsg);
  }, 5000);
}

function stopLoader() {
  clearInterval(loaderIv);
  if (loaderIv._coldStartTimer) clearTimeout(loaderIv._coldStartTimer);
  const coldMsg = document.getElementById('coldStartMsg');
  if (coldMsg) coldMsg.remove();
  const bar   = document.querySelector('.loader-progress');
  const pctEl = document.getElementById('loaderPct');
  bar.style.width   = '100%';
  pctEl.textContent = '100%';
  // mark last step done
  LOADER_STEPS.forEach(s => {
    const el = document.getElementById(s.id);
    el.classList.remove('active');
    el.classList.add('done');
  });
  setTimeout(() => {
    loader.classList.remove('visible');
    button.disabled = false;
    bar.style.width = '0%';
    pctEl.textContent = '0%';
    LOADER_STEPS.forEach(s => document.getElementById(s.id).classList.remove('active','done'));
  }, 350);
}


// ══════════════════════════════════════════════════════
//  RESULTS RENDER
// ══════════════════════════════════════════════════════
const resultsDiv = document.getElementById('results');
function renderResults(data) {
  resultsDiv.innerHTML = '';
  const results = data.results;
  if (!results || typeof results !== 'object') {
    resultsDiv.innerHTML = '<div class="error-msg">⚠ Unexpected response format.</div>';
    return;
  }

  const hasToxic = Object.values(results).some(r => r.prediction === 1);
  const toxicCount = Object.values(results).filter(r => r.prediction === 1).length;
  const maxConf = Math.max(...Object.values(results).map(r => r.confidence));

  // Summary card
  const summary = document.createElement('div');
  summary.className = `result-summary ${hasToxic ? 'toxic-summary' : 'safe-summary'}`;
  summary.innerHTML = hasToxic
    ? `<span class="summary-icon">⚠</span>
       <div class="summary-text-wrap">
         <div class="summary-headline" style="color:var(--accent)">TOXIC CONTENT DETECTED</div>
         <div class="summary-sub">${toxicCount} CATEGOR${toxicCount > 1 ? 'IES' : 'Y'} FLAGGED — MAX ${(maxConf*100).toFixed(1)}% CONFIDENCE</div>
       </div>`
    : `<span class="summary-icon" style="color:var(--green)">✓</span>
       <div class="summary-text-wrap">
         <div class="summary-headline" style="color:var(--green)">CONTENT CLEARED</div>
         <div class="summary-sub">NO TOXICITY SIGNALS DETECTED ABOVE THRESHOLD</div>
       </div>`;
  resultsDiv.appendChild(summary);

  // Category bars — sort toxic first
  const sorted = Object.keys(results).sort((a, b) => {
    const da = results[a]; const db = results[b];
    if (da.prediction !== db.prediction) return db.prediction - da.prediction;
    return db.confidence - da.confidence;
  });

  sorted.forEach((label, i) => {
    const { confidence: prob, prediction } = results[label];
    const isToxic = prediction === 1;
    const pct = (prob * 100).toFixed(1);

    const item = document.createElement('div');
    item.className = `result-item${isToxic ? ' is-toxic' : ''}`;
    item.style.animationDelay = `${i * 0.06}s`;
    item.innerHTML = `
      <div class="result-row">
        <span class="result-label">${escapeHTML(label)}</span>
        <span class="result-badge ${isToxic ? 'toxic' : 'safe'}">${isToxic ? '⚠ ' : ''}${pct}%</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill ${isToxic ? 'toxic' : ''}"></div>
      </div>`;
    resultsDiv.appendChild(item);

    requestAnimationFrame(() => requestAnimationFrame(() => {
      item.querySelector('.bar-fill').style.width = pct + '%';
    }));
  });

  // ── Rewrite suggestion (only when toxic) ──
  if (data.rewritten_text) {
    const rewriteBox = document.createElement('div');
    rewriteBox.className = 'rewrite-box';
    rewriteBox.innerHTML = `
      <div class="rewrite-label">✦ SUGGESTED REWRITE</div>
      <div class="rewrite-text">${escapeHTML(data.rewritten_text)}</div>
      <button class="copy-btn" id="copyRewrite">⊕ COPY</button>`;
    resultsDiv.appendChild(rewriteBox);
    document.getElementById('copyRewrite').addEventListener('click', () => {
      navigator.clipboard.writeText(data.rewritten_text);
      showToast('✓ COPIED TO CLIPBOARD');
    });
  }
}

// ══════════════════════════════════════════════════════
//  PREDICT — wired to the real HF Space backend
// ══════════════════════════════════════════════════════
const BACKEND = 'https://kanisk29-toxicity-backend.hf.space/predict';

async function predict() {
  const text = inputText.value.trim();
  if (!text) {
    inputText.focus();
    inputText.style.borderColor = 'rgba(255,51,102,0.6)';
    inputText.style.boxShadow   = '0 0 0 3px rgba(255,51,102,0.08)';
    setTimeout(() => {
      inputText.style.borderColor = '';
      inputText.style.boxShadow   = '';
    }, 900);
    showToast('⚠ INPUT REQUIRED');
    return;
  }

  resultsDiv.innerHTML = '';
  document.getElementById('heatmapSection').style.display = 'none';
  startLoader();

  try {
    const response = await fetch(BACKEND, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ text }),
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const data = await response.json();
    stopLoader();

    renderResults(data);
    renderHeatmap(text, data.results);
    updateStats(data.results);
    saveToHistory(text, data);

    const hasToxic = Object.values(data.results).some(r => r.prediction === 1);
    showToast(hasToxic ? '⚠ TOXIC SIGNALS FOUND' : '✓ CONTENT CLEARED');

  } catch (err) {
    console.error(err);
    stopLoader();
    resultsDiv.innerHTML = `
      <div class="error-msg">
        ⚠ BACKEND UNREACHABLE<br>
        <span style="opacity:0.6;font-size:10px">Check that the HuggingFace Space is awake — it may need a moment to cold-start.</span>
      </div>`;
    showToast('✕ CONNECTION FAILED');
  }
}

document.getElementById('analyzeBtn').addEventListener('click', predict);
inputText.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') predict();
});


// ══════════════════════════════════════════════════════
//  TOAST
// ══════════════════════════════════════════════════════
let toastTimer = null;
function showToast(msg) {
  const toast = document.getElementById('toast');
  toast.textContent = msg;
  toast.classList.add('visible');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('visible'), 2400);
}


// ══════════════════════════════════════════════════════
//  UTIL
// ══════════════════════════════════════════════════════
function escapeHTML(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
