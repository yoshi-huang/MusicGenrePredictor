# -----------------------------------------------------------------------------
# Qt-embedded matplotlib canvas widgets (GUI only)
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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


def _lerp_oklab(c1_hex: str, c2_hex: str, steps: int) -> list:
    """Interpolate two colours in Oklab space for perceptually smooth gradients."""
    def srgb_to_linear(c):
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    def linear_to_srgb(c):
        return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)

    def rgb_to_oklab(rgb):
        r, g, b = srgb_to_linear(rgb).T
        l = 0.4122214708*r + 0.5363325363*g + 0.0514459929*b
        m = 0.2119034982*r + 0.6806995451*g + 0.1073969566*b
        s = 0.0883024619*r + 0.2817188376*g + 0.6299787005*b
        l_, m_, s_ = np.cbrt(l), np.cbrt(m), np.cbrt(s)
        return np.stack([
            0.2104542553*l_ + 0.7936177850*m_ - 0.0040720468*s_,
            1.9779984951*l_ - 2.4285922050*m_ + 0.4505937099*s_,
            0.0259040371*l_ + 0.7827717662*m_ - 0.8086757660*s_,
        ], axis=-1)

    def oklab_to_rgb(lab):
        L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
        l_ = L + 0.3963377774*a + 0.2158037573*b
        m_ = L - 0.1055613458*a - 0.0638541728*b
        s_ = L - 0.0894841775*a - 1.2914855480*b
        l, m, s = l_**3, m_**3, s_**3
        r =  4.0767416621*l - 3.3077115913*m + 0.2309699292*s
        g = -1.2684380046*l + 2.6097574011*m - 0.3413193965*s
        b_ = -0.0041960863*l - 0.7034186147*m + 1.7076147010*s
        return np.clip(linear_to_srgb(np.stack([r, g, b_], axis=-1)), 0, 1)

    rgb1 = np.array(mcolors.to_rgb(c1_hex), dtype=float).reshape(1, 3)
    rgb2 = np.array(mcolors.to_rgb(c2_hex), dtype=float).reshape(1, 3)
    lab1 = rgb_to_oklab(rgb1)[0]
    lab2 = rgb_to_oklab(rgb2)[0]
    result = []
    for k in range(steps):
        t = k / max(steps - 1, 1)
        lab = lab1 * (1 - t) + lab2 * t
        rgb = oklab_to_rgb(lab.reshape(1, 3))[0]
        result.append(tuple(rgb))
    return result


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
        total = len(votes)

        # build segments (including -1 transition runs)
        segs, s = [], 0
        for i in range(1, total):
            if votes[i] != votes[i - 1]:
                segs.append((s, i - s, votes[s]))
                s = i
        segs.append((s, total - s, votes[s]))

        for seg_start, length, idx in segs:
            if idx == -1:
                # find prev/next real genre for gradient
                prev_idx = next((votes[j] for j in range(seg_start - 1, -1, -1) if votes[j] != -1), None)
                next_idx = next((votes[j] for j in range(seg_start + length, total) if votes[j] != -1), None)
                c1 = COLOR_MAP.get(genres[prev_idx], 'gray') if prev_idx is not None else 'gray'
                c2 = COLOR_MAP.get(genres[next_idx], 'gray') if next_idx is not None else 'gray'
                # matplotlib doesn't support per-bar gradients natively; simulate with thin slices
                colors = _lerp_oklab(c1, c2, length * 10)
                for k, color in enumerate(colors):
                    self.ax.broken_barh([(seg_start + k / 10, 1 / 10)], (0, 1),
                                        facecolors=[color], edgecolors='none')
            else:
                label = genres[idx]
                self.ax.broken_barh([(seg_start, length)], (0, 1),
                                    facecolors=COLOR_MAP.get(label, 'gray'), edgecolors='none')
                if length >= threshold:
                    mid = seg_start + length / 2
                    self.ax.text(mid, 1.06, f"{length}s", ha='center', va='bottom', fontsize=12)
                    self.ax.text(mid, 0.5, label, ha='center', va='center', fontsize=10,
                                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.15'))

        # x-axis ticks at segment starts (non-transition, long enough)
        tick_pos = [s for s, l, idx in segs if idx != -1 and l >= threshold]
        self.ax.set_xticks(tick_pos)
        self.ax.set_xticklabels([f"{t // 60:>2}:{t % 60:02d}" for t in tick_pos])
        self.ax.set_xlim(0, total)
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
