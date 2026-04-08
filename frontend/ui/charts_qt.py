# -----------------------------------------------------------------------------
# Qt-embedded matplotlib canvas widgets (GUI only)
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa.display

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QSizePolicy

from core.audio import GENRES, runlength_filter
from core.config import audio_cfg

COLOR_MAP = {
    "blues": 'dodgerblue', "classical": 'orange',  "country": 'peru',
    "disco": 'y',          "hiphop": 'yellowgreen', "jazz": 'steelblue',
    "metal": 'slategray',  "pop": 'darkviolet',     "reggae": 'green',
    "rock": 'firebrick'
}


class BarChartCanvas(FigureCanvas):
    def __init__(self, votes: np.ndarray, genres: list = GENRES, parent=None):
        fig, self.ax = plt.subplots(figsize=(5, 3), dpi=100)
        super().__init__(fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(300)
        self.plot(votes, genres)

    def plot(self, votes, genres):
        pct = 100 * np.array(votes) / np.sum(votes)
        self.ax.clear()
        bars = self.ax.bar(genres, pct)
        bars[int(pct.argmax())].set_color("tab:red")
        for bar, v in zip(bars, pct):
            self.ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                         f"{v: <4.1f}%" if v > 10 else "",
                         ha='center', va='bottom', fontsize=7)
        self.ax.set_ylabel("possible (%)")
        self.ax.set_ylim(0, 100)
        self.ax.set_title("genre prediction votes")
        self.ax.tick_params(axis="x", rotation=45)
        self.figure.tight_layout()
        self.draw()


class BrokenBarhCanvas(FigureCanvas):
    def __init__(self, votes: list, genres: list = GENRES, parent=None):
        fig_width = max(5, len(votes) * 0.2)
        fig, self.ax = plt.subplots(figsize=(fig_width, 2), dpi=100)
        super().__init__(fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(200)
        self.plot(votes, genres)

    def plot(self, votes: list, genres: list, threshold: int = 5):
        self.ax.clear()
        sequence = [genres[i] for i in votes]

        segs, start = [], 0
        for i in range(1, len(sequence)):
            if sequence[i] != sequence[i - 1]:
                segs.append((start, i - start, sequence[start]))
                start = i
        segs.append((start, len(sequence) - start, sequence[start]))

        for s, length, label in segs:
            self.ax.broken_barh([(s, length)], (0, 1), facecolors=COLOR_MAP.get(label, 'gray'))
            if length >= threshold:
                mid = s + length / 2
                self.ax.text(mid, 1.06, f"{length}s", ha='center', va='bottom', fontsize=12)
                self.ax.text(mid, 0.5, label, ha='center', va='center', fontsize=10,
                             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.15'))

        tick_pos, start = [], 0
        for i in range(1, len(sequence) + 1):
            if i == len(sequence) or sequence[i] != sequence[start]:
                if (i - start) >= threshold:
                    tick_pos.append(start)
                start = i
        self.ax.set_xticks(tick_pos)
        self.ax.set_xticklabels([f"{t // 60: >2}:{t % 60:0>2}" for t in tick_pos])
        self.ax.set_xlim(0, len(sequence))
        self.ax.set_ylim(0, 1.35)
        self.ax.set_yticks([])
        self.ax.set_xlabel("Time", fontsize=10.5, loc='left')
        self.ax.tick_params(axis='x', labelsize=10)
        self.figure.tight_layout()
        self.draw()


class SpectrogramCanvas(FigureCanvas):
    def __init__(self, mel_db: np.ndarray, sr: int = None, parent=None):
        if sr is None:
            sr = audio_cfg["sample_rate"]
        fig, self.ax = plt.subplots(figsize=(5, 3), dpi=100)
        super().__init__(fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(300)
        self.plot(mel_db, sr)

    def plot(self, mel_db, sr):
        self.ax.clear()
        hop_length = audio_cfg["hop_length"]
        img = librosa.display.specshow(mel_db, sr=sr, hop_length=hop_length,
                                       y_axis='mel', x_axis='time', ax=self.ax)
        self.ax.set_title('log-mel spectrogram')
        step = max(1, len(mel_db.T) // 43 // 6)
        self.ax.set_xticks(np.arange(0, len(mel_db.T) // 43, step))
        self.figure.colorbar(img, ax=self.ax, format='%+2.0f dB')
        self.figure.tight_layout()
        self.draw()
