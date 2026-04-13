"""
Flask web server for Music Genre Predictor
Run: python web/app.py  (from project root)
"""

import io
import os
import sys

import base64
import tempfile

# Must be set before importing torch or numba to avoid LLVM SVML symbol conflict
os.environ.setdefault("NUMBA_DISABLE_INTEL_SVML", "1")

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa.display

from flask import Flask, request, jsonify, send_from_directory

# ── resolve backend on path ───────────────────────────────────────────────────
_ROOT    = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.join(os.path.dirname(_ROOT), "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.audio  import youtube_download, process_audio, runlength_filter, apply_shift_and_transitions, GENRES
from core.model  import load_model
from core.config import paths, audio_cfg, dcbot_cfg

# ── app & lazy model ──────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=os.path.join(_ROOT, "static"))

_model = None
def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model

# ── chart helpers ─────────────────────────────────────────────────────────────
COLOR_MAP = {
    "blues":"#1E90FF","classical":"#FFA500","country":"#CD853F",
    "disco":"#FFD700","hiphop":"#9ACD32","jazz":"#4682B4",
    "metal":"#708090","pop":"#9400D3","reggae":"#228B22","rock":"#B22222",
}

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def _bar_chart(votes: np.ndarray) -> str:
    pct    = 100 * votes / votes.sum()
    colors = [COLOR_MAP.get(g, "gray") for g in GENRES]
    colors[int(pct.argmax())] = "#e74c3c"
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=100)
    bars = ax.bar(GENRES, pct, color=colors)
    for bar, v in zip(bars, pct):
        if v > 5:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("votes (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Genre Prediction")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return _fig_to_b64(fig)

def _spectrogram(mel_db: np.ndarray) -> str:
    sr  = audio_cfg["sample_rate"]
    hop = audio_cfg["hop_length"]
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=100)
    img = librosa.display.specshow(mel_db, sr=sr, hop_length=hop,
                                   y_axis="mel", x_axis="time", ax=ax)
    ax.set_title("Log-Mel Spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    return _fig_to_b64(fig)

def _timeline(vote_seq: list) -> str:
    total  = len(vote_seq)
    labels = [GENRES[v] for v in vote_seq]
    fig, ax = plt.subplots(figsize=(max(8, total * 0.06), 2), dpi=100)
    segs, s = [], 0
    for i in range(1, total):
        if labels[i] != labels[i - 1]:
            segs.append((s, i - s, labels[s]))
            s = i
    segs.append((s, total - s, labels[s]))
    for start, length, label in segs:
        ax.broken_barh([(start, length)], (0, 1), facecolors=COLOR_MAP.get(label, "gray"), edgecolors='none')
        if length >= 3:
            mid = start + length / 2
            ax.text(mid, 0.5, label, ha="center", va="center", fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", boxstyle="round,pad=0.1"))
    ax.set_xlim(0, total)
    ax.set_ylim(0, 1.3)
    ax.set_yticks([])
    ax.set_xlabel("Time (s)")
    tick_step = max(1, total // 20)
    ticks = list(range(0, total, tick_step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t // 60}:{t % 60:02d}" for t in ticks], fontsize=7)
    fig.tight_layout()
    return _fig_to_b64(fig)

def _run_inference(audio_path: str) -> dict:
    model    = get_model()
    segments, mel_db = process_audio(audio_path)
    x        = torch.tensor(np.array(segments)).float()
    with torch.no_grad():
        pred = model(x)
    vote_seq = F.softmax(pred, dim=1).argmax(dim=1).numpy().tolist()
    filtered = runlength_filter(vote_seq, threshold=dcbot_cfg["runlength_threshold"])
    filtered, tagged = apply_shift_and_transitions(filtered)
    counts = np.bincount(filtered, minlength=len(GENRES))
    return {
        "top_genre":     GENRES[int(counts.argmax())],
        "vote_counts":   counts.tolist(),
        "vote_seq":      tagged,
        "bar_chart":     _bar_chart(counts),
        "spectrogram":   _spectrogram(mel_db),
    }

# ── routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/analyze/url", methods=["POST"])
def analyze_url():
    url = request.form.get("url", "").strip()
    if not url:
        return jsonify(error="請提供 YouTube URL"), 400
    tmp = paths["audio_temp_file"]
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    try:
        youtube_download(url, tmp)
        result = _run_inference(tmp)
        # get title via yt-dlp (no download)
        try:
            from yt_dlp import YoutubeDL
            with YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
                info = ydl.extract_info(url, download=False)
                result["title"] = info.get("title", "")
        except Exception:
            result["title"] = ""
        with open(tmp, "rb") as af:
            result["audio_b64"]  = base64.b64encode(af.read()).decode()
            result["audio_mime"] = "audio/mpeg"
    except Exception as e:
        return jsonify(error=str(e)), 500
    return jsonify(result)

@app.route("/analyze/file", methods=["POST"])
def analyze_file():
    f = request.files.get("file")
    if not f or not f.filename.lower().endswith((".mp3", ".wav")):
        return jsonify(error="僅支援 .mp3 / .wav 檔案"), 400
    suffix = os.path.splitext(f.filename)[1]
    mime   = "audio/mpeg" if suffix.lower() == ".mp3" else "audio/wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name
    try:
        result = _run_inference(tmp_path)
        result["title"] = os.path.splitext(f.filename)[0]
        with open(tmp_path, "rb") as af:
            result["audio_b64"]  = base64.b64encode(af.read()).decode()
            result["audio_mime"] = mime
    except Exception as e:
        return jsonify(error=str(e)), 500
    finally:
        os.unlink(tmp_path)
    return jsonify(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)