// ══════════════════════════════════════════════════════
//  THREE.JS BACKGROUND  —  dual mesh with particles
// ══════════════════════════════════════════════════════
const scene    = new THREE.Scene();
const camera   = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ canvas: document.querySelector('#bg'), antialias: true, alpha: true });

renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0x000000, 0);
camera.position.z = 38;

// Primary icosahedron — wireframe
const geoA  = new THREE.IcosahedronGeometry(11, 1);
const matA  = new THREE.MeshStandardMaterial({ color: 0xa855f7, wireframe: true, transparent: true, opacity: 0.18 });
const meshA = new THREE.Mesh(geoA, matA);
scene.add(meshA);

// Secondary — smaller, offset
const geoB  = new THREE.IcosahedronGeometry(6, 1);
const matB  = new THREE.MeshStandardMaterial({ color: 0x6366f1, wireframe: true, transparent: true, opacity: 0.12 });
const meshB = new THREE.Mesh(geoB, matB);
meshB.position.set(18, -8, -10);
scene.add(meshB);

// Tertiary — extra depth accent
const geoC  = new THREE.IcosahedronGeometry(4, 1);
const matC  = new THREE.MeshStandardMaterial({ color: 0xec4899, wireframe: true, transparent: true, opacity: 0.08 });
const meshC = new THREE.Mesh(geoC, matC);
meshC.position.set(-20, 10, -15);
scene.add(meshC);

// Particle field
const particleCount = 280;
const positions     = new Float32Array(particleCount * 3);
const basePositions = new Float32Array(particleCount * 3);
for (let i = 0; i < particleCount * 3; i++) {
  const v = (Math.random() - 0.5) * 130;
  positions[i]     = v;
  basePositions[i] = v;
}
const pGeo = new THREE.BufferGeometry();
pGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
const pMat      = new THREE.PointsMaterial({ color: 0xa855f7, size: 0.28, transparent: true, opacity: 0.55 });
const particles = new THREE.Points(pGeo, pMat);
scene.add(particles);

// Lights
scene.add(Object.assign(new THREE.PointLight(0xa855f7, 1.4), { position: new THREE.Vector3(20, 20, 20) }));
scene.add(Object.assign(new THREE.PointLight(0x6366f1, 0.7), { position: new THREE.Vector3(-20, -15, 10) }));
scene.add(Object.assign(new THREE.PointLight(0xec4899, 0.5), { position: new THREE.Vector3(0, -20, 15) }));
scene.add(new THREE.AmbientLight(0xffffff, 0.06));

// ── Expanding pulse ring ──────────────────────────────
const ringGeo = new THREE.TorusGeometry(1, 0.05, 8, 80);
const ringMat = new THREE.MeshBasicMaterial({ color: 0xa855f7, transparent: true, opacity: 0 });
const ring    = new THREE.Mesh(ringGeo, ringMat);
scene.add(ring);

// Second ring (slightly delayed)
const ring2Mat = new THREE.MeshBasicMaterial({ color: 0xf472b6, transparent: true, opacity: 0 });
const ring2    = new THREE.Mesh(new THREE.TorusGeometry(1, 0.03, 8, 80), ring2Mat);
scene.add(ring2);

// Mouse parallax
let mouseX = 0, mouseY = 0;
document.addEventListener('mousemove', e => {
  mouseX = (e.clientX / window.innerWidth  - 0.5) * 2;
  mouseY = (e.clientY / window.innerHeight - 0.5) * 2;
});

// ── Colors ────────────────────────────────────────────
const COL_A_NORMAL = new THREE.Color(0xa855f7);
const COL_B_NORMAL = new THREE.Color(0x6366f1);
const COL_P_NORMAL = new THREE.Color(0xa855f7);
const COL_A_BURST  = new THREE.Color(0xf472b6);
const COL_B_BURST  = new THREE.Color(0xfbbf24);
const COL_RING2    = new THREE.Color(0xf472b6);

// ── Submit effect state ───────────────────────────────
let submitEffect = {
  active:    false,
  startTime: 0,
  duration:  2.5,
};
const pVelocities = new Float32Array(particleCount * 3);

function triggerSubmitEffect() {
  submitEffect.active    = true;
  submitEffect.startTime = performance.now() / 1000;

  // Random outward burst per particle
  for (let i = 0; i < particleCount * 3; i += 3) {
    pVelocities[i]   = (Math.random() - 0.5) * 1.0;
    pVelocities[i+1] = (Math.random() - 0.5) * 1.0;
    pVelocities[i+2] = (Math.random() - 0.5) * 1.0;
  }

  // Reset rings
  ring.scale.set(1, 1, 1);
  ring2.scale.set(1, 1, 1);
  ringMat.opacity  = 0.9;
  ring2Mat.opacity = 0.7;
}
window.triggerSubmitEffect = triggerSubmitEffect;

let clock    = 0;
let lastTime = performance.now() / 1000;

function animate() {
  requestAnimationFrame(animate);
  const now = performance.now() / 1000;
  const dt  = now - lastTime;
  lastTime  = now;
  clock    += dt;

  // ── Idle rotation ─────────────────────────────────
  meshA.rotation.x += 0.002 + mouseY * 0.0015;
  meshA.rotation.y += 0.004 + mouseX * 0.0015;
  meshB.rotation.x -= 0.003;
  meshB.rotation.y -= 0.005;
  meshC.rotation.x += 0.0015;
  meshC.rotation.z += 0.002;

  matA.opacity = 0.14 + Math.sin(clock * 1.5) * 0.04;
  matB.opacity = 0.09 + Math.sin(clock * 2.0) * 0.03;

  particles.rotation.y += 0.0003;
  particles.rotation.x  = Math.sin(clock * 0.4) * 0.05;

  // ── Submit burst ──────────────────────────────────
  if (submitEffect.active) {
    const elapsed = now - submitEffect.startTime;
    const t       = Math.min(elapsed / submitEffect.duration, 1);

    // Camera shake — decays fast
    const shake = 1.2 * Math.exp(-elapsed * 5);
    camera.position.x = (Math.random() - 0.5) * shake * 0.4;
    camera.position.y = (Math.random() - 0.5) * shake * 0.4;

    // Mesh scale spike
    const scalePulse = 1 + 0.38 * Math.exp(-elapsed * 3) * Math.sin(elapsed * 20);
    meshA.scale.setScalar(scalePulse);
    meshB.scale.setScalar(1 + 0.22 * Math.exp(-elapsed * 3.5) * Math.sin(elapsed * 15 + 1));

    // Color flash → return to normal
    const colorT = Math.min(elapsed / 0.45, 1);
    matA.color.lerpColors(COL_A_BURST, COL_A_NORMAL, colorT);
    matB.color.lerpColors(COL_B_BURST, COL_B_NORMAL, colorT);
    pMat.color.lerpColors(COL_A_BURST, COL_P_NORMAL, colorT);

    // Particle burst: outward then drift back
    const posAttr = pGeo.attributes.position;
    const decay   = Math.exp(-elapsed * 1.6);
    for (let i = 0; i < particleCount * 3; i += 3) {
      posAttr.array[i]   = basePositions[i]   + pVelocities[i]   * elapsed * 7 * decay;
      posAttr.array[i+1] = basePositions[i+1] + pVelocities[i+1] * elapsed * 7 * decay;
      posAttr.array[i+2] = basePositions[i+2] + pVelocities[i+2] * elapsed * 7 * decay;
    }
    posAttr.needsUpdate = true;

    // Particle size / opacity flare
    pMat.opacity = 0.55 + 0.45 * Math.exp(-elapsed * 1.8);
    pMat.size    = 0.28 + 0.55 * Math.exp(-elapsed * 2.5);

    // Ring 1 — fast expand
    const rs1 = 1 + elapsed * 28;
    ring.scale.set(rs1, rs1, rs1);
    ringMat.opacity = Math.max(0, 0.9 - elapsed * 1.0);

    // Ring 2 — slightly delayed & slower
    const rs2 = 1 + Math.max(0, elapsed - 0.15) * 18;
    ring2.scale.set(rs2, rs2, rs2);
    ring2Mat.opacity = Math.max(0, 0.7 - Math.max(0, elapsed - 0.15) * 1.1);

    // Wireframe opacity flare
    matA.opacity = Math.min(0.6, 0.14 + 0.65 * Math.exp(-elapsed * 2.0));

    if (t >= 1) {
      submitEffect.active = false;
      meshA.scale.setScalar(1);
      meshB.scale.setScalar(1);
      camera.position.set(0, 0, 38);
      pMat.size    = 0.28;
      pMat.opacity = 0.55;
    }
  }

  renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', () => {
  renderer.setSize(window.innerWidth, window.innerHeight);
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
});


// ══════════════════════════════════════════════════════
//  CHAR COUNT
// ══════════════════════════════════════════════════════
const inputText = document.getElementById('inputText');
const charCount = document.getElementById('charCount');
inputText.addEventListener('input', () => {
  charCount.textContent = inputText.value.length;
});


// ══════════════════════════════════════════════════════
//  HISTORY
// ══════════════════════════════════════════════════════
let history = [];

function saveToHistory(text, data) {
  history.unshift({ id: Date.now(), text, results: data.results, timestamp: new Date() });
  renderHistory();
}

function renderHistory() {
  const list  = document.getElementById('historyList');
  const empty = document.getElementById('historyEmpty');

  list.querySelectorAll('.history-card').forEach(el => el.remove());

  if (history.length === 0) { empty.style.display = 'flex'; return; }
  empty.style.display = 'none';

  history.forEach((entry, idx) => {
    const results  = entry.results || {};
    const hasToxic = Object.keys(results).some(k => results[k].prediction === 1);

    const card = document.createElement('div');
    card.className = `history-card ${hasToxic ? 'has-toxic' : 'all-safe'}`;
    card.style.animationDelay = `${idx * 0.04}s`;

    const timeStr  = entry.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const pillsHTML = Object.keys(results).slice(0, 3).map(k => {
      const isToxic = results[k].prediction === 1;
      const pct = (results[k].confidence * 100).toFixed(0);
      return `<span class="pill ${isToxic ? 'toxic-pill' : 'safe-pill'}">${k} ${pct}%</span>`;
    }).join('');

    card.innerHTML = `
      <div class="card-text">${escapeHTML(entry.text)}</div>
      <div class="card-meta">
        <span class="card-time">${timeStr}</span>
        <div class="card-pills">${pillsHTML}</div>
      </div>`;

    card.addEventListener('click', () => {
      inputText.value = entry.text;
      charCount.textContent = entry.text.length;
      renderResults(entry);
      document.querySelector('.main-panel').scrollTop = 0;
    });

    list.appendChild(card);
  });
}

document.getElementById('clearBtn').addEventListener('click', () => {
  history = [];
  renderHistory();
});


// ══════════════════════════════════════════════════════
//  ANALYZE
// ══════════════════════════════════════════════════════
const button     = document.getElementById('analyzeBtn');
const loader     = document.getElementById('loader');
const resultsDiv = document.getElementById('results');

button.addEventListener('click', predict);

async function predict() {
  const text = inputText.value.trim();
  if (!text) {
    inputText.focus();
    inputText.style.borderColor = 'rgba(255,77,109,0.5)';
    setTimeout(() => inputText.style.borderColor = '', 800);
    return;
  }

  // 🎆 Fire background burst on click
  window.triggerSubmitEffect();

  resultsDiv.innerHTML = '';
  loader.classList.add('visible');
  button.disabled = true;

  try {
    const response = await fetch('http://127.0.0.1:8000/predict', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ text })
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();

    loader.classList.remove('visible');
    button.disabled = false;
    renderResults(data);
    saveToHistory(text, data);

  } catch {
    loader.classList.remove('visible');
    button.disabled = false;
    resultsDiv.innerHTML = `
      <div class="error-msg">
        ⚠ Backend unreachable — make sure the server is running on <strong>localhost:8000</strong>
      </div>`;
  }
}


// ══════════════════════════════════════════════════════
//  RENDER RESULTS
// ══════════════════════════════════════════════════════
function renderResults(data) {
  resultsDiv.innerHTML = '';
  const results = data.results;
  if (!results || typeof results !== 'object') {
    resultsDiv.innerHTML = '<div class="error-msg">Unexpected response format.</div>';
    return;
  }

  Object.keys(results).forEach((label, i) => {
    const { confidence: prob, prediction } = results[label];
    const isToxic = prediction === 1;
    const pct     = (prob * 100).toFixed(1);

    const item = document.createElement('div');
    item.className = 'result-item';
    item.style.animationDelay = `${i * 0.06}s`;

    item.innerHTML = `
      <div class="result-row">
        <span class="result-label">${escapeHTML(label)}</span>
        <span class="result-badge ${isToxic ? 'toxic' : 'safe'}">${pct}%</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill ${isToxic ? 'toxic' : ''}"></div>
      </div>`;

    resultsDiv.appendChild(item);

    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        item.querySelector('.bar-fill').style.width = pct + '%';
      });
    });
  });
}


// ══════════════════════════════════════════════════════
//  UTIL
// ══════════════════════════════════════════════════════
function escapeHTML(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
