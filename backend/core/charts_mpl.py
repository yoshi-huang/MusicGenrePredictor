# -----------------------------------------------------------------------------
# Pure-matplotlib canvas helpers (DCBOT / headless use)
# -----------------------------------------------------------------------------

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

from core.audio import GENRES
from core.config import paths, audio_cfg, dcbot_cfg

COLOR_MAP = {
    "blues": 'dodgerblue', "classical": 'orange',  "country": 'peru',
    "disco": 'y',          "hiphop": 'yellowgreen', "jazz": 'steelblue',
    "metal": 'slategray',  "pop": 'darkviolet',     "reggae": 'green',
    "rock": 'firebrick'
}


class BarChartCanvas:
    def __init__(self, votes: np.ndarray, genres: list = GENRES,
                 save_path: str = None):
        if save_path is None:
            save_path = paths["dcbot_bar_chart"]
        fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
        pct = 100 * np.array(votes) / np.sum(votes)
        bars = ax.bar(genres, pct)
        bars[int(pct.argmax())].set_color("tab:red")
        for bar, v in zip(bars, pct):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                    f"{v: <4.1f}%" if v > 10 else "",
                    ha='center', va='bottom', fontsize=7)
        ax.set_ylabel("possible (%)")
        ax.set_ylim(0, 100)
        ax.set_title("genre prediction votes")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)


class SpectrogramCanvas:
    def __init__(self, mel_db: np.ndarray, sr: int = None,
                 save_path: str = None):
        if sr is None:
            sr = audio_cfg["sample_rate"]
        if save_path is None:
            save_path = paths["dcbot_spectrogram"]
        hop_length = audio_cfg["hop_length"]
        fig, ax = plt.subplots(figsize=(8, 3), dpi=100)
        img = librosa.display.specshow(mel_db, sr=sr, hop_length=hop_length,
                                       y_axis='mel', x_axis='time', ax=ax)
        ax.set_title('log-mel spectrogram')
        step = max(1, len(mel_db.T) // 43 // 6)
        ax.set_xticks(np.arange(0, len(mel_db.T) // 43, step))
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)


def _clear_folder(folder_path: str):
    if not os.path.exists(folder_path):
        return
    for name in os.listdir(folder_path):
        fp = os.path.join(folder_path, name)
        try:
            os.unlink(fp) if os.path.isfile(fp) or os.path.islink(fp) else shutil.rmtree(fp)
        except Exception as e:
            print(f"Error removing {fp}: {e}")


class BrokenBarhCanvas:
    def __init__(self, votes: list, genres: list = GENRES,
                 window_size: int = None, threshold: int = None,
                 output_dir: str = None):
        if window_size is None:
            window_size = dcbot_cfg["chart_window_size"]
        if threshold is None:
            threshold = dcbot_cfg["runlength_threshold"]
        if output_dir is None:
            output_dir = paths["dcbot_windows_dir"] + "/"
        _clear_folder(output_dir)
        self.votes      = votes
        self.genres     = genres
        self.window_size = window_size
        self.threshold  = threshold
        self.output_dir = output_dir
        self._save_all_segments()

    def _save_all_segments(self):
        total = len(self.votes)
        for start in range(0, total, self.window_size):
            self._plot_segment(start, min(start + self.window_size, total))

    def _plot_segment(self, start_sec: int, end_sec: int):
        sequence = [self.genres[self.votes[i]] for i in range(start_sec, end_sec)]
        fig_width = max(5, (end_sec - start_sec) * 0.2)
        fig, ax = plt.subplots(figsize=(fig_width, 2), dpi=100)

        segs, seg_start = [], 0
        for i in range(1, len(sequence)):
            if sequence[i] != sequence[i - 1]:
                segs.append((seg_start, i - seg_start, sequence[seg_start]))
                seg_start = i
        segs.append((seg_start, len(sequence) - seg_start, sequence[seg_start]))

        for s, length, label in segs:
            ax.broken_barh([(start_sec + s, length)], (0, 1),
                           facecolors=COLOR_MAP.get(label, 'gray'))
            if length >= self.threshold:
                mid = start_sec + s + length / 2
                ax.text(mid, 1.06, f"{length}s", ha='center', va='bottom', fontsize=12)
                ax.text(mid, 0.5, label, ha='center', va='center', fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.15'))

        tick_pos, seg_start = [], 0
        for i in range(1, len(sequence) + 1):
            if i == len(sequence) or sequence[i] != sequence[seg_start]:
                if (i - seg_start) >= self.threshold:
                    tick_pos.append(start_sec + seg_start)
                seg_start = i
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([f"{t // 60:>2}:{t % 60:02d}" for t in tick_pos])
        ax.set_xlim(start_sec, end_sec)
        ax.set_ylim(0, 1.35)
        ax.set_yticks([])
        ax.set_xlabel("Time", fontsize=10.5, loc='left')
        ax.tick_params(axis='x', labelsize=10)
        fig.tight_layout()

        os.makedirs(self.output_dir, exist_ok=True)
        fig.savefig(f"{self.output_dir}BrokenBarh_{start_sec:04d}-{end_sec:04d}.png")
        plt.close(fig)
