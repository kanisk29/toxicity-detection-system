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

  if (history.length === 0) { 
    empty.style.display = 'flex'; 
    return; 
  }
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

  resultsDiv.innerHTML = '';
  loader.classList.add('visible');
  button.disabled = true;

  try {
    const response = await fetch('https://kanisk29-toxicity-backend.hf.space/predict', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ text })
    });

    console.log("STATUS:", response.status);

    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const data = await response.json();
    console.log("DATA:", data);

    loader.classList.remove('visible');
    button.disabled = false;

    renderResults(data);
    saveToHistory(text, data);

  } catch (err) {
    console.error(err);

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