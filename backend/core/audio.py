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

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COOKIES_PATH = os.path.join(_ROOT, "auth/cookies.txt")

def youtube_download(url: str, output_path: str, progress_callback=None) -> str:
    """Download audio track via yt_dlp and report progress (0-100).

    Returns the video title (empty string on failure).
    Title is extracted from the download info dict — no extra network request needed.
    """

    print(f'{os.path.splitext(output_path)[0]}')

    max_pct = [0]
    title_holder = [""]

    def hook(d):
        # grab title from the info dict the first time it appears
        if not title_holder[0]:
            title_holder[0] = (d.get("info_dict") or {}).get("title", "")
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
        'User-Agent': 'Mozilla/5.0',
        'progress_hooks': [hook],
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        try:
            os.remove(paths["audio_temp_file"])
        except Exception:
            pass
        info = ydl.extract_info(url, download=True)
        if not title_holder[0] and info:
            title_holder[0] = info.get("title", "")

    return title_holder[0]


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


def runlength_filter(seq, threshold: int = 4):
    """Suppress short-run noise in a label sequence.

    Pass 1 – sandwich removal:
      If a short run is flanked by the same label on both sides, remove it
      so the two outer runs merge. Repeat until stable.

    Pass 2 – neighbour merge:
      Merge any remaining short run into its longer neighbour
      (ties → left, since transitions take time). Repeat until stable.

    Re-encodes after every change so adjacent same-label runs collapse
    immediately, preventing phantom short runs from surviving.
    """
    def encode(s):
        runs = []
        cur, cnt = s[0], 1
        for v in s[1:]:
            if v == cur:
                cnt += 1
            else:
                runs.append([cur, cnt])
                cur, cnt = v, 1
        runs.append([cur, cnt])
        return runs

    def flatten(runs):
        return [v for label, cnt in runs for v in [label] * cnt]

    runs = encode(seq)

    # Pass 1: sandwich removal
    changed = True
    while changed:
        changed = False
        for i in range(1, len(runs) - 1):
            if runs[i][1] < threshold and runs[i - 1][0] == runs[i + 1][0]:
                runs[i][0] = runs[i - 1][0]   # absorb into surrounding label
                runs = encode(flatten(runs))
                changed = True
                break  # restart after re-encode

    # Pass 2: neighbour merge for remaining short runs
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(runs):
            if runs[i][1] < threshold:
                prev_len = runs[i - 1][1] if i > 0             else -1
                next_len = runs[i + 1][1] if i < len(runs) - 1 else -1

                if prev_len == -1 and next_len == -1:
                    break
                elif prev_len == -1:
                    runs[i][0] = runs[i + 1][0]
                elif next_len == -1:
                    runs[i][0] = runs[i - 1][0]
                else:
                    runs[i][0] = runs[i - 1][0] if prev_len >= next_len else runs[i + 1][0]

                changed = True
                runs = encode(flatten(runs))
                i = 0
            else:
                i += 1

    return flatten(runs)


def apply_shift_and_transitions(filtered: list, shift: int = 2) -> tuple:
    """
    Apply sliding-window lag compensation and mark transition seconds.

    Returns (filtered, tagged):
      - filtered : shift-corrected sequence (int genre indices), used for vote counts
      - tagged   : same but with -1 at each boundary second (prev & next), used for display
    """
    if len(filtered) > shift:
        filtered = filtered[shift:] + [filtered[-1]] * shift

    tagged = list(filtered)
    for i in range(1, len(filtered)):
        if filtered[i] != filtered[i - 1]:
            tagged[i - 1] = -1
            tagged[i]     = -1
    return filtered, tagged

