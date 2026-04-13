# DCBOT processing - delegates to core modules

import os
import sys

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

import numpy as np
import torch
import torch.nn.functional as F

from core.model import load_model
from core.audio import GENRES, youtube_download, process_audio, runlength_filter, apply_shift_and_transitions
from core.config import dcbot_cfg, paths
from core.charts_mpl import BarChartCanvas, SpectrogramCanvas, BrokenBarhCanvas


def run(source_url: str):
    """Download, analyse, save charts. Returns (winner_index, result_lines, title)."""
    model = load_model("transformer_parms.pth")
    tmp_path = paths["audio_temp_file"]
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    youtube_download(source_url, tmp_path)

    # get title via yt-dlp metadata (no extra download)
    title = ""
    try:
        from yt_dlp import YoutubeDL
        with YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
            info = ydl.extract_info(source_url, download=False)
            title = info.get("title", "")
    except Exception:
        pass

    mel, mel_db = process_audio(tmp_path)
    x = torch.tensor(mel).float()
    with torch.no_grad():
        pred = model(x)
    votes = F.softmax(pred, dim=1).argmax(dim=1).numpy()
    filtered = runlength_filter(votes.tolist(), threshold=dcbot_cfg["runlength_threshold"])
    filtered, tagged = apply_shift_and_transitions(filtered)
    return display(votes, mel_db, tagged, title)


def display(votes, mel_db, tagged, title=""):
    total_votes = np.bincount([v for v in tagged if v != -1], minlength=10)
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
