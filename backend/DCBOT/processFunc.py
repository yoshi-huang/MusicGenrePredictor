# DCBOT processing - delegates to core modules

import os
import sys

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

import numpy as np
import torch
import torch.nn.functional as F

from core.model import load_model, TransformerEncoder
from core.audio import GENRES, youtube_download, process_audio, runlength_filter, apply_shift_and_transitions
from core.config import dcbot_cfg, paths
from core.charts_mpl import BarChartCanvas, SpectrogramCanvas, BrokenBarhCanvas

# ---------------------------------------------------------------------------
# Module-level model cache — loaded once, reused across calls
# ---------------------------------------------------------------------------
_model: TransformerEncoder | None = None

def get_model() -> TransformerEncoder:
    """Return the cached model, loading it on first call."""
    global _model
    if _model is None:
        _model = load_model("transformer_parms.pth")
    return _model


def run(source_url: str):
    """Download, analyse, save charts. Returns (winner_index, result_lines, title)."""
    model = get_model()
    tmp_path = paths["audio_temp_file"]
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    title = youtube_download(source_url, tmp_path)

    segments, mel_db = process_audio(tmp_path)
    x = torch.from_numpy(np.array(segments, dtype=np.float32))  # (N, seq_len, 128)

    # run inference in small batches to avoid OOM on long tracks
    BATCH = 64
    label_list = []
    with torch.no_grad():
        for i in range(0, len(x), BATCH):
            pred = model(x[i:i + BATCH])
            label_list.extend(F.softmax(pred, dim=1).argmax(dim=1).tolist())

    votes_arr = [0] * len(GENRES)
    for idx in label_list:
        votes_arr[idx] += 1

    filtered = runlength_filter(label_list, threshold=dcbot_cfg["runlength_threshold"])
    filtered, tagged = apply_shift_and_transitions(filtered)
    return display(votes_arr, mel_db, tagged, title)


def display(votes_arr, mel_db, tagged, title=""):
    total_votes = np.array(votes_arr)
    winner = int(total_votes.argmax())
    BarChartCanvas(total_votes, GENRES)
    SpectrogramCanvas(mel_db)
    BrokenBarhCanvas(tagged, GENRES)
    lines = "```prediction result:\n"
    for i, genre in enumerate(GENRES):
        marker = "<-- " if i == winner else ""
        lines += f"\n{genre:<10s} : {int(total_votes[i]):<3d} votes {marker}"
    lines += "```"
    return winner, lines, title
