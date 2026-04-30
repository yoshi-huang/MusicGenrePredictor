const COLOR_MAP = {
  blues:'#1E90FF', classical:'#FFA500', country:'#CD853F', disco:'#FFD700',
  hiphop:'#9ACD32', jazz:'#4682B4', metal:'#708090', pop:'#9400D3',
  reggae:'#228B22', rock:'#B22222'
};

// Tesla design tokens
const TESLA_BLUE = '#3E6AE1';

let activeTab = 'url';

function switchTab(tab) {
  activeTab = tab;
  document.querySelectorAll('.tab').forEach((t, i) => {
    const isActive = (i === 0 && tab === 'url') || (i === 1 && tab === 'file');
    t.classList.toggle('active', isActive);
    t.setAttribute('aria-selected', isActive);
  });
  document.getElementById('panel-url').classList.toggle('active', tab === 'url');
  document.getElementById('panel-file').classList.toggle('active', tab === 'file');
}

function setProgress(pct, label) {
  document.getElementById('progress-fill').style.width = pct + '%';
  document.getElementById('progress-label').textContent = label;
}

function showError(msg) {
  const el = document.getElementById('error-msg');
  el.textContent = msg;
  el.style.display = 'block';
}

function fmt(s) {
  s = Math.max(0, s || 0);
  const m = Math.floor(s / 60), sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, '0')}`;
}

// ── Timeline canvas ──────────────────────────────────────────────────────────
const canvas = document.getElementById('timeline-canvas');
const ctx    = canvas.getContext('2d');
let voteSeq  = [];

function buildSegments(seq) {
  const segs = [];
  if (!seq.length) return segs;
  let s = 0;
  for (let i = 1; i < seq.length; i++) {
    if (seq[i] !== seq[i - 1]) { segs.push({ start: s, end: i, idx: seq[s] }); s = i; }
  }
  segs.push({ start: s, end: seq.length, idx: seq[s] });
  return segs;
}

function nearestGenre(seq, i, dir) {
  for (let j = i + dir; j >= 0 && j < seq.length; j += dir) {
    if (seq[j] !== -1) return seq[j];
  }
  return null;
}

// Oklab interpolation
function hexToLinear(hex) {
  const c = parseInt(hex.slice(1), 16);
  return [c >> 16, (c >> 8) & 0xff, c & 0xff].map(v => {
    v /= 255;
    return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
  });
}
function linearToSrgb(v) {
  return v <= 0.0031308 ? v * 12.92 : 1.055 * v ** (1 / 2.4) - 0.055;
}
function rgbToOklab([r, g, b]) {
  const l = 0.4122214708*r + 0.5363325363*g + 0.0514459929*b;
  const m = 0.2119034982*r + 0.6806995451*g + 0.1073969566*b;
  const s = 0.0883024619*r + 0.2817188376*g + 0.6299787005*b;
  const [l_, m_, s_] = [l, m, s].map(v => Math.cbrt(v));
  return [
    0.2104542553*l_ + 0.7936177850*m_ - 0.0040720468*s_,
    1.9779984951*l_ - 2.4285922050*m_ + 0.4505937099*s_,
    0.0259040371*l_ + 0.7827717662*m_ - 0.8086757660*s_,
  ];
}
function oklabToRgb([L, a, b]) {
  const l_ = L + 0.3963377774*a + 0.2158037573*b;
  const m_ = L - 0.1055613458*a - 0.0638541728*b;
  const s_ = L - 0.0894841775*a - 1.2914855480*b;
  const [l, m, s] = [l_, m_, s_].map(v => v ** 3);
  const r  =  4.0767416621*l - 3.3077115913*m + 0.2309699292*s;
  const g  = -1.2684380046*l + 2.6097574011*m - 0.3413193965*s;
  const bv = -0.0041960863*l - 0.7034186147*m + 1.7076147010*s;
  return [r, g, bv].map(v => Math.round(Math.min(1, Math.max(0, linearToSrgb(v))) * 255));
}
function lerpOklab(hex1, hex2, t) {
  const lab1 = rgbToOklab(hexToLinear(hex1));
  const lab2 = rgbToOklab(hexToLinear(hex2));
  const lab  = lab1.map((v, i) => v * (1 - t) + lab2[i] * t);
  const [r, g, b] = oklabToRgb(lab);
  return `rgb(${r},${g},${b})`;
}

function colorOf(idx) {
  const label = Object.keys(COLOR_MAP)[idx];
  return COLOR_MAP[label] || '#888';
}

function drawTimeline(playPct) {
  const W = canvas.clientWidth, H = canvas.clientHeight;
  const total = voteSeq.length;
  if (!total) return;
  ctx.clearRect(0, 0, W, H);

  const segs   = buildSegments(voteSeq);
  const GENRES = Object.keys(COLOR_MAP);

  for (const seg of segs) {
    const x1   = Math.floor((seg.start / total) * W);
    const x2   = Math.ceil((seg.end   / total) * W);
    const segW = x2 - x1;

    if (seg.idx === -1) {
      const prevIdx = nearestGenre(voteSeq, seg.start, -1);
      const nextIdx = nearestGenre(voteSeq, seg.end - 1, +1);
      const c1 = prevIdx !== null ? colorOf(prevIdx) : '#444';
      const c2 = nextIdx !== null ? colorOf(nextIdx) : '#444';
      const grad = ctx.createLinearGradient(x1, 0, x2, 0);
      const stops = 8;
      for (let i = 0; i <= stops; i++) {
        grad.addColorStop(i / stops, lerpOklab(c1, c2, i / stops));
      }
      ctx.fillStyle = grad;
      ctx.fillRect(x1, 0, segW, H);
    } else {
      const label = GENRES[seg.idx];
      ctx.fillStyle = colorOf(seg.idx);
      ctx.fillRect(x1, 0, segW, H);

      if (segW > 36) {
        ctx.save();
        ctx.fillStyle = 'rgba(0,0,0,0.45)';
        const pad = 4, th = 14;
        const tw  = Math.min(segW - 4, ctx.measureText(label).width + pad * 2);
        const tx  = x1 + (segW - tw) / 2, ty = (H - th) / 2;
        ctx.beginPath();
        ctx.roundRect(tx, ty, tw, th, 3);
        ctx.fill();
        ctx.fillStyle = '#fff';
        ctx.font = '500 10px "Universal Sans Text", Arial, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, x1 + segW / 2, H / 2, segW - 8);
        ctx.restore();
      }
    }
  }

  // playhead
  if (playPct != null) {
    const px = playPct * W;
    ctx.save();
    ctx.strokeStyle = '#171A20';
    ctx.lineWidth = 2;
    ctx.shadowColor = 'rgba(0,0,0,0.3)';
    ctx.shadowBlur = 4;
    ctx.beginPath();
    ctx.moveTo(px, 0);
    ctx.lineTo(px, H);
    ctx.stroke();
    ctx.restore();
  }
}

function buildLabels(seq) {
  const wrap = document.getElementById('timeline-labels');
  wrap.innerHTML = '';
  const total = seq.length;
  if (!total) return;

  const boundaries = [0];
  for (let i = 1; i < total; i++) {
    if (seq[i] === -1 && seq[i - 1] !== -1) {
      let j = i;
      while (j < total && seq[j] === -1) j++;
      boundaries.push((i + j) / 2);
      i = j - 1;
    }
  }

  boundaries.forEach(t => {
    const span = document.createElement('span');
    span.textContent = fmt(t);
    span.style.left = (t / total * 100) + '%';
    wrap.appendChild(span);
  });
}

function resizeCanvas() {
  const W   = canvas.parentElement.clientWidth;
  const dpr = window.devicePixelRatio || 1;
  canvas.width  = W * dpr;
  canvas.height = 48 * dpr;
  canvas.style.width  = W + 'px';
  canvas.style.height = '48px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  drawTimeline(audio.duration ? audio.currentTime / audio.duration : null);
}

window.addEventListener('resize', resizeCanvas);

// ── Player ───────────────────────────────────────────────────────────────────
const audio     = document.getElementById('audio-el');
const playBtn   = document.getElementById('play-btn');
const iconPlay  = document.getElementById('icon-play');
const iconPause = document.getElementById('icon-pause');
const seekWrap  = document.getElementById('seek-wrap');
const seekFill  = document.getElementById('seek-fill');
const seekThumb = document.getElementById('seek-thumb');
const timeCur   = document.getElementById('time-cur');
const timeDur   = document.getElementById('time-dur');

function updateSeek() {
  const pct = audio.duration ? audio.currentTime / audio.duration : 0;
  const p   = (pct * 100).toFixed(2) + '%';
  seekFill.style.width = p;
  seekThumb.style.left = p;
  timeCur.textContent  = fmt(audio.currentTime);
  seekWrap.setAttribute('aria-valuenow', Math.round(pct * 100));
  drawTimeline(pct);
}

playBtn.addEventListener('click', () => { audio.paused ? audio.play() : audio.pause(); });
audio.addEventListener('play',  () => { iconPlay.style.display = 'none'; iconPause.style.display = ''; });
audio.addEventListener('pause', () => { iconPlay.style.display = '';     iconPause.style.display = 'none'; });
audio.addEventListener('ended', () => { iconPlay.style.display = '';     iconPause.style.display = 'none'; });
audio.addEventListener('timeupdate', updateSeek);
audio.addEventListener('loadedmetadata', () => { timeDur.textContent = fmt(audio.duration); });

let seeking = false;

function seekTo(clientX) {
  const rect = seekWrap.getBoundingClientRect();
  const pct  = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
  if (audio.duration) audio.currentTime = pct * audio.duration;
  updateSeek();
}

canvas.addEventListener('mousedown',  e => { seeking = true; seekTo(e.clientX); });
seekWrap.addEventListener('mousedown', e => { seeking = true; seekTo(e.clientX); });
document.addEventListener('mousemove', e => { if (seeking) seekTo(e.clientX); });
document.addEventListener('mouseup',   () => { seeking = false; });

canvas.addEventListener('touchstart',  e => { seeking = true; seekTo(e.touches[0].clientX); }, { passive: true });
seekWrap.addEventListener('touchstart', e => { seeking = true; seekTo(e.touches[0].clientX); }, { passive: true });
document.addEventListener('touchmove',  e => { if (seeking) seekTo(e.touches[0].clientX); }, { passive: true });
document.addEventListener('touchend',   () => { seeking = false; });

function loadAudio(b64, mime) {
  audio.pause();
  audio.src = `data:${mime};base64,${b64}`;
  audio.load();
  iconPlay.style.display  = '';
  iconPause.style.display = 'none';
  timeCur.textContent = '0:00';
  seekFill.style.width = '0%';
  seekThumb.style.left = '0%';
  drawTimeline(0);
}

// ── Analyze ──────────────────────────────────────────────────────────────────
async function analyze() {
  document.getElementById('error-msg').style.display = 'none';
  document.getElementById('results').style.display   = 'none';
  document.getElementById('progress-wrap').style.display = 'block';
  document.getElementById('analyze-btn').disabled = true;
  setProgress(10, '準備中…');

  try {
    let endpoint, form;

    if (activeTab === 'url') {
      const url = document.getElementById('url-input').value.trim();
      if (!url) { showError('請輸入 YouTube URL'); return; }
      setProgress(20, '下載音訊…');
      endpoint = '/analyze/url';
      form = new FormData();
      form.append('url', url);
    } else {
      const file = document.getElementById('file-input').files[0];
      if (!file) { showError('請選擇音訊檔案'); return; }
      setProgress(20, '上傳檔案…');
      endpoint = '/analyze/file';
      form = new FormData();
      form.append('file', file);
    }

    const res = await fetch(endpoint, { method: 'POST', body: form });
    setProgress(80, '推論中…');

    const data = await res.json();
    if (!res.ok) throw new Error(data.error || '分析失敗');

    setProgress(100, '完成');

    document.getElementById('top-genre').textContent  = data.top_genre;
    document.getElementById('song-title').textContent = data.title || '';
    document.getElementById('bar-chart').src   = 'data:image/png;base64,' + data.bar_chart;
    document.getElementById('spectrogram').src = 'data:image/png;base64,' + data.spectrogram;

    voteSeq = data.vote_seq || [];
    document.getElementById('results').style.display = 'block';
    document.querySelector('.main-layout').classList.add('has-results');

    requestAnimationFrame(() => {
      resizeCanvas();
      buildLabels(voteSeq);
      timeDur.textContent = fmt(voteSeq.length);
    });

    if (data.audio_b64) loadAudio(data.audio_b64, data.audio_mime);

    // scroll to results
    document.getElementById('results').scrollIntoView({ behavior: 'smooth', block: 'start' });

  } catch (e) {
    showError(e.message);
  } finally {
    document.getElementById('analyze-btn').disabled = false;
    setTimeout(() => { document.getElementById('progress-wrap').style.display = 'none'; }, 1400);
  }
}
