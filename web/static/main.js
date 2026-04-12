const GENRE_ICON = {
  blues:'', classical:'', country:'', disco:'',
  hiphop:'', jazz:'', metal:'', pop:'', reggae:'', rock:''
};

let activeTab = 'url';

function switchTab(tab) {
  activeTab = tab;
  document.querySelectorAll('.tab').forEach((t, i) =>
    t.classList.toggle('active', (i === 0 && tab === 'url') || (i === 1 && tab === 'file'))
  );
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

async function analyze() {
  document.getElementById('error-msg').style.display = 'none';
  document.getElementById('results').style.display = 'none';
  document.getElementById('progress-wrap').style.display = 'block';
  document.getElementById('analyze-btn').disabled = true;
  setProgress(10, '準備中...');

  try {
    let endpoint, form;

    if (activeTab === 'url') {
      const url = document.getElementById('url-input').value.trim();
      if (!url) { showError('請輸入 YouTube URL'); return; }
      setProgress(20, '下載音訊...');
      endpoint = '/analyze/url';
      form = new FormData();
      form.append('url', url);
    } else {
      const file = document.getElementById('file-input').files[0];
      if (!file) { showError('請選擇音訊檔案'); return; }
      setProgress(20, '上傳檔案...');
      endpoint = '/analyze/file';
      form = new FormData();
      form.append('file', file);
    }

    const res = await fetch(endpoint, { method: 'POST', body: form });
    setProgress(80, '推論中...');

    const data = await res.json();
    if (!res.ok) throw new Error(data.error || '分析失敗');

    setProgress(100, '完成');
    document.getElementById('top-genre').textContent = data.top_genre;
    document.getElementById('genre-icon').textContent = GENRE_ICON[data.top_genre] || '';
    document.getElementById('bar-chart').src   = 'data:image/png;base64,' + data.bar_chart;
    document.getElementById('spectrogram').src = 'data:image/png;base64,' + data.spectrogram;
    document.getElementById('timeline').src    = 'data:image/png;base64,' + data.timeline;
    document.getElementById('results').style.display = 'block';

  } catch (e) {
    showError(e.message);
  } finally {
    document.getElementById('analyze-btn').disabled = false;
    setTimeout(() => { document.getElementById('progress-wrap').style.display = 'none'; }, 1200);
  }
}
