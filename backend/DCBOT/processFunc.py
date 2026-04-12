# DCBOT processing - delegates to core modules

import numpy as np
import torch
import torch.nn.functional as F

from core.model import load_model
from core.audio import GENRES, youtube_download, process_audio, runlength_filter
from core.charts_mpl import BarChartCanvas, SpectrogramCanvas, BrokenBarhCanvas


def run(source_url: str):
    """Download, analyse, save charts. Returns (winner_index, result_lines)."""
    model = load_model("transformer_parms.pth")
    tmp_path = "audio_temp_save/temp_audio.mp3"
    import os
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    youtube_download(source_url, tmp_path)
    mel, mel_db = process_audio(tmp_path)
    x = torch.tensor(mel).float()
    with torch.no_grad():
        pred = model(x)
    votes = F.softmax(pred, dim=1).argmax(dim=1).numpy()
    return display(votes, mel_db)


def display(votes, mel_db):
    total_votes = np.bincount(votes, minlength=10)
    winner = int(total_votes.argmax())
    BarChartCanvas(total_votes, GENRES)
    SpectrogramCanvas(mel_db)
    padded = [votes[0]] + list(votes) + [votes[-1]]
    BrokenBarhCanvas(runlength_filter(padded), GENRES)
    lines = "```prediction result:\n"
    for i, genre in enumerate(GENRES):
        marker = "<-- " if i == winner else ""
        lines += f"\n{genre:<10s} : {int(total_votes[i]):<3d} votes {marker}"
    lines += "```"
    return winner, lines
