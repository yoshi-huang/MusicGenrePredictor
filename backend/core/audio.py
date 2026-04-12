# -----------------------------------------------------------------------------
# Audio utilities (shared by GUI and DCBOT)
# -----------------------------------------------------------------------------

import os
import numpy as np
import librosa
from yt_dlp import YoutubeDL
from core.config import paths, audio_cfg

GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]


def youtube_download(url: str, output_path: str, progress_callback=None) -> None:
    """Download audio track via yt_dlp and report progress (0-100)."""

    print(f'{os.path.splitext(output_path)[0]}')

    max_pct = [0]

    def hook(d):
        if d["status"] == "downloading" and progress_callback:
            total_bytes = d.get("total_bytes") or d.get("total_bytes_estimate")
            downloaded = d.get("downloaded_bytes", 0)
            if total_bytes:
                pct = int(downloaded / total_bytes * 100)
                if pct > max_pct[0]:
                    max_pct[0] = pct
                    progress_callback(min(max_pct[0], 99), "downloading")
        elif d["status"] == "finished" and progress_callback:
            progress_callback(100, "converting to mp3...")
        elif d["status"] == "postprocessor" and progress_callback:
            progress_callback(100, "converting to mp3...")

    # strip extension to get the base path for yt-dlp (it appends .mp3 itself)
    base_path = os.path.splitext(output_path)[0]
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': base_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'progress_hooks': [hook],
        'noplaylist': True
    }

    with YoutubeDL(ydl_opts) as ydl:
        try:
            os.remove(paths["audio_temp_file"])
        except Exception:
            pass
        ydl.download([url])


def process_audio(path: str, segment_frames: int = None, step: int = None,
                  progress_callback=None):
    """
    Load an audio file and return (segments, mel_db).

    segments : list of (segment_frames, 128) arrays  — model input
    mel_db   : full log-mel spectrogram array
    """
    if segment_frames is None:
        segment_frames = audio_cfg["segment_frames"]
    if step is None:
        step = audio_cfg["step"]
    sr       = audio_cfg["sample_rate"]
    hop_len  = audio_cfg["hop_length"]

    if progress_callback:
        progress_callback(0, "loading audio")
    y_audio, sr = librosa.load(path, sr=sr)

    if progress_callback:
        progress_callback(25, "computing mel spectrogram")
    mel = librosa.feature.melspectrogram(y=y_audio, sr=sr, hop_length=hop_len)

    if progress_callback:
        progress_callback(50, "converting to log-mel")
    mel_db = librosa.power_to_db(mel, ref=np.max)

    segments = [
        mel_db[:, i:i + segment_frames]
        for i in range(0, mel_db.shape[1] - segment_frames + 1, step)
    ]

    if progress_callback:
        progress_callback(75, "trimming silence")
    segments = np.transpose(np.array(segments), (0, 2, 1))

    if progress_callback:
        progress_callback(100, "processing audio done")
    return list(segments), mel_db


def runlength_filter(seq, threshold: int = 5):
    """Suppress short-run noise in a label sequence."""
    segments, current_label, count = [], seq[0], 1
    for label in seq[1:]:
        if label == current_label:
            count += 1
        else:
            segments.append((current_label, count))
            current_label, count = label, 1
    segments.append((current_label, count))

    new_labels = [l for l, _ in segments]
    lengths    = [n for _, n in segments]
    n = len(segments)

    for _ in range(2):
        for i in range(n):
            if lengths[i] < threshold:
                prev_len = lengths[i - 1] if i > 0     else -1
                next_len = lengths[i + 1] if i < n - 1 else -1
                if prev_len >= next_len and i > 0:
                    new_labels[i] = new_labels[i - 1]
                elif i < n - 1:
                    new_labels[i] = new_labels[i + 1]

    filtered = []
    for label, length in zip(new_labels, lengths):
        filtered.extend([label] * length)
    return filtered
