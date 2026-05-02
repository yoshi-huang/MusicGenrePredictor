
# -----------------------------------------------------------------------------
# Pure-matplotlib canvas helpers (DCBOT / headless use)
# -----------------------------------------------------------------------------

import os
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe to call from any thread
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import librosa.display

from core.audio import GENRES
from core.config import paths, audio_cfg, dcbot_cfg

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
        c = np.maximum(c, 0)  # guard against tiny negatives from float arithmetic
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
        window = self.votes[start_sec:end_sec]
        total_w = len(window)
        fig_width = max(5, total_w * 0.2)
        fig, ax = plt.subplots(figsize=(fig_width, 2), dpi=100)

        # build segments
        segs, s = [], 0
        for i in range(1, total_w):
            if window[i] != window[i - 1]:
                segs.append((s, i - s, window[s]))
                s = i
        segs.append((s, total_w - s, window[s]))

        for seg_s, length, idx in segs:
            abs_s = start_sec + seg_s
            if idx == -1:
                prev_idx = next((window[j] for j in range(seg_s - 1, -1, -1) if window[j] != -1), None)
                next_idx = next((window[j] for j in range(seg_s + length, total_w) if window[j] != -1), None)
                c1 = COLOR_MAP.get(self.genres[prev_idx], 'gray') if prev_idx is not None else 'gray'
                c2 = COLOR_MAP.get(self.genres[next_idx], 'gray') if next_idx is not None else 'gray'
                steps = max(length * 10, 2)
                colors = _lerp_oklab(c1, c2, steps)
                for k, color in enumerate(colors):
                    ax.broken_barh([(abs_s + k * length / steps, length / steps)], (0, 1),
                                   facecolors=[color], edgecolors='none')
            else:
                label = self.genres[idx]
                ax.broken_barh([(abs_s, length)], (0, 1),
                               facecolors=COLOR_MAP.get(label, 'gray'), edgecolors='none')
                if length >= self.threshold:
                    mid = abs_s + length / 2
                    ax.text(mid, 1.06, f"{length}s", ha='center', va='bottom', fontsize=12)
                    ax.text(mid, 0.5, label, ha='center', va='center', fontsize=10,
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.15'))

        tick_pos = [start_sec + s for s, l, idx in segs if idx != -1 and l >= self.threshold]
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
