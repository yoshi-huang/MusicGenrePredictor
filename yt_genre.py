#!/usr/bin/env python3
"""
yt-genre  —  Music Genre Predictor CLI
Usage:
    python yt_genre.py <youtube_url>
    yt-genre <youtube_url>          # after: pip install -e .

Outputs:
    - Vote table + dominant genre printed to terminal
    - One matplotlib window with three subplots:
        1. Bar chart  (vote % per genre)
        2. Log-Mel Spectrogram
        3. Full-length BrokenBarh timeline
"""

import sys
import os
import argparse
import tempfile
import shutil

# ── resolve project root so imports work regardless of cwd ──────────────────
ROOT    = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(ROOT, "backend")
sys.path.insert(0, ROOT)
sys.path.insert(0, BACKEND)

# ── colour helpers ───────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
DIM    = "\033[2m"

GENRE_LABELS = [
    "blues", "classical", "country", "disco",
    "hiphop", "jazz", "metal", "pop", "reggae", "rock",
]
GENRE_ZH = {
    "blues": "藍調", "classical": "古典", "country": "鄉村",
    "disco": "舞廳", "hiphop": "嘻哈",  "jazz": "爵士",
    "metal": "重金屬", "pop": "流行",   "reggae": "雷鬼", "rock": "搖滾",
}

BAR_WIDTH = 30


def _bar(count: int, total: int) -> str:
    filled = int(BAR_WIDTH * count / total) if total else 0
    return f"[{'█' * filled}{'░' * (BAR_WIDTH - filled)}]"


def print_banner():
    print(f"\n{CYAN}{BOLD}{'─' * 50}")
    print("  yt-genre  |  Music Genre Predictor")
    print(f"{'─' * 50}{RESET}\n")


def print_results(votes: list, filtered: list):
    total        = sum(votes)
    dominant_idx = votes.index(max(votes))

    print(f"{BOLD}{'Genre':<12} {'ZH':<6} {'Votes':>6}  {'Distribution'}{RESET}")
    print("─" * 60)
    for i, (label, vote) in enumerate(zip(GENRE_LABELS, votes)):
        bar   = _bar(vote, total)
        pct   = f"{vote / total * 100:5.1f}%" if total else "  0.0%"
        zh    = GENRE_ZH[label]
        arrow = f"  {GREEN}◀ dominant{RESET}" if i == dominant_idx else ""
        print(f"  {label:<12}{zh:<6}{vote:>6}  {bar} {pct}{arrow}")

    print("─" * 60)
    dominant = GENRE_LABELS[dominant_idx]
    print(f"\n{BOLD}Dominant genre : {GREEN}{dominant}{RESET}  "
          f"({GENRE_ZH[dominant]})  "
          f"{DIM}[{votes[dominant_idx]}/{total} votes]{RESET}")

    if filtered:
        unique = list(dict.fromkeys(g for g in filtered if g != -1))
        parts  = [
            f"{GREEN}{GENRE_LABELS[g]}{RESET}" if g == dominant_idx else GENRE_LABELS[g]
            for g in unique
        ]
        print(f"{BOLD}Timeline       :{RESET} {' → '.join(parts)}\n")


# ── chart helpers (pure matplotlib, no Qt) ──────────────────────────────────

COLOR_MAP = {
    "blues": "dodgerblue", "classical": "orange",  "country": "peru",
    "disco": "y",          "hiphop": "yellowgreen", "jazz": "steelblue",
    "metal": "slategray",  "pop": "darkviolet",     "reggae": "green",
    "rock":  "firebrick",
}


def _lerp_oklab(c1_hex: str, c2_hex: str, steps: int) -> list:
    import numpy as np
    import matplotlib.colors as mcolors

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
        r  =  4.0767416621*l - 3.3077115913*m + 0.2309699292*s
        g  = -1.2684380046*l + 2.6097574011*m - 0.3413193965*s
        b_ = -0.0041960863*l - 0.7034186147*m + 1.7076147010*s
        return np.clip(linear_to_srgb(np.stack([r, g, b_], axis=-1)), 0, 1)

    rgb1 = np.array(mcolors.to_rgb(c1_hex), dtype=float).reshape(1, 3)
    rgb2 = np.array(mcolors.to_rgb(c2_hex), dtype=float).reshape(1, 3)
    lab1 = rgb_to_oklab(rgb1)[0]
    lab2 = rgb_to_oklab(rgb2)[0]
    result = []
    for k in range(steps):
        t   = k / max(steps - 1, 1)
        lab = lab1 * (1 - t) + lab2 * t
        result.append(tuple(oklab_to_rgb(lab.reshape(1, 3))[0]))
    return result


def _draw_bar(ax, votes: list):
    import numpy as np
    genres = GENRE_LABELS
    pct    = 100 * np.array(votes) / np.sum(votes)
    bars   = ax.bar(genres, pct)
    bars[int(pct.argmax())].set_color("tab:red")
    for bar, v in zip(bars, pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                f"{v:<4.1f}%" if v > 10 else "",
                ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("possible (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Genre Prediction Votes")
    ax.tick_params(axis="x", rotation=45)


def _draw_spectrogram(ax, fig, mel_db):
    import numpy as np
    import librosa.display
    from core.config import audio_cfg
    sr         = audio_cfg["sample_rate"]
    hop_length = audio_cfg["hop_length"]
    img = librosa.display.specshow(mel_db, sr=sr, hop_length=hop_length,
                                   y_axis="mel", x_axis="time", ax=ax)
    ax.set_title("Log-Mel Spectrogram")
    step = max(1, mel_db.shape[1] // 43 // 6)
    ax.set_xticks(range(0, mel_db.shape[1] // 43, step))
    fig.colorbar(img, ax=ax, format="%+2.0f dB")


def _draw_brokenbarh(ax, tagged: list, threshold: int = 5):
    """Full-length timeline — mirrors BrokenBarhCanvas.plot() from charts_qt.py."""
    genres = GENRE_LABELS
    total  = len(tagged)

    # build run-length segments
    segs, s = [], 0
    for i in range(1, total):
        if tagged[i] != tagged[s]:
            segs.append((s, i - s, tagged[s]))
            s = i
    segs.append((s, total - s, tagged[s]))

    for seg_start, length, idx in segs:
        if idx == -1:
            prev_idx = next((tagged[j] for j in range(seg_start - 1, -1, -1) if tagged[j] != -1), None)
            next_idx = next((tagged[j] for j in range(seg_start + length, total) if tagged[j] != -1), None)
            c1 = COLOR_MAP.get(genres[prev_idx], "gray") if prev_idx is not None else "gray"
            c2 = COLOR_MAP.get(genres[next_idx], "gray") if next_idx is not None else "gray"
            colors = _lerp_oklab(c1, c2, length * 10)
            for k, color in enumerate(colors):
                ax.broken_barh([(seg_start + k / 10, 1 / 10)], (0, 1),
                               facecolors=[color], edgecolors="none")
        else:
            label = genres[idx]
            ax.broken_barh([(seg_start, length)], (0, 1),
                           facecolors=COLOR_MAP.get(label, "gray"), edgecolors="none")
            if length >= threshold:
                mid = seg_start + length / 2
                ax.text(mid, 1.06, f"{length}s", ha="center", va="bottom", fontsize=9)
                ax.text(mid, 0.5, label, ha="center", va="center", fontsize=8,
                        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none",
                                  boxstyle="round,pad=0.15"))

    tick_pos = [s for s, l, idx in segs if idx != -1 and l >= threshold]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([f"{t // 60}:{t % 60:02d}" for t in tick_pos])
    ax.set_xlim(0, total)
    ax.set_ylim(0, 1.35)
    ax.set_yticks([])
    ax.set_xlabel("Time", fontsize=10, loc="left")
    ax.tick_params(axis="x", labelsize=9)
    ax.set_title("Genre Timeline")


def show_combined(votes: list, mel_db, tagged: list):
    """Three subplots in one window, then plt.show()."""
    import matplotlib
    matplotlib.use("TkAgg")   # use interactive backend (not Agg)
    import matplotlib.pyplot as plt

    total_secs = len(tagged)
    # timeline width scales with duration; min 8, max 24 inches
    bb_width   = max(6, min(24, total_secs * 0.15))

    fig = plt.figure(figsize=(bb_width, 4), constrained_layout=True)
    fig.canvas.manager.set_window_title("yt-genre  |  Music Genre Predictor")

    # GridSpec: bar + spectrogram side-by-side on top, timeline full-width below
    gs  = fig.add_gridspec(2, 2, height_ratios=[3, 1.5])
    ax_bar  = fig.add_subplot(gs[0, 0])
    ax_spec = fig.add_subplot(gs[0, 1])
    ax_bb   = fig.add_subplot(gs[1, :])

    _draw_bar(ax_bar, votes)
    _draw_spectrogram(ax_spec, fig, mel_db)
    _draw_brokenbarh(ax_bb, tagged)

    plt.show()


# ── main pipeline ────────────────────────────────────────────────────────────

def run(url: str, keep_charts: bool = False):
    print_banner()

    # 1. config
    print(f"{DIM}[1/5] Loading config …{RESET}")
    from core.config import paths, audio_cfg, dcbot_cfg  # noqa: F401

    # 2. model
    print(f"{DIM}[2/5] Loading model …{RESET}")
    from core.model import load_model
    model = load_model()

    # 3. download
    print(f"{DIM}[3/5] Downloading audio from YouTube …{RESET}")
    from core.audio import youtube_download, process_audio, runlength_filter, \
        apply_shift_and_transitions, GENRES

    tmp_dir = tempfile.mkdtemp(prefix="yt_genre_")
    tmp_mp3 = os.path.join(tmp_dir, "audio.mp3")

    try:
        youtube_download(url, tmp_mp3)

        # 4. inference  (mirrors worker.py)
        print(f"{DIM}[4/5] Running inference …{RESET}")
        import torch
        import torch.nn.functional as F

        segments, mel_db = process_audio(tmp_mp3)
        x = torch.tensor(segments).float()          # (N, seq_len, 128)

        with torch.no_grad():
            pred = model(x)                         # (N, 10)

        frame_labels = F.softmax(pred, dim=1).argmax(dim=1).tolist()
        votes_arr    = [0] * len(GENRES)
        for idx in frame_labels:
            votes_arr[idx] += 1

        filtered = runlength_filter(frame_labels, threshold=dcbot_cfg["runlength_threshold"])
        filtered, tagged = apply_shift_and_transitions(filtered)

        print_results(votes_arr, tagged)

        # 5. display
        print(f"{DIM}[5/5] Rendering charts …{RESET}")
        show_combined(votes_arr, mel_db, tagged)

        if keep_charts:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from core.charts_mpl import BarChartCanvas, SpectrogramCanvas, BrokenBarhCanvas
            dest   = os.path.join(os.getcwd(), "yt_genre_charts")
            bb_dir = os.path.join(dest, "windows")
            os.makedirs(bb_dir, exist_ok=True)
            BarChartCanvas(votes_arr,  save_path=os.path.join(dest, "BarChart.png"))
            SpectrogramCanvas(mel_db,  save_path=os.path.join(dest, "Spectrogram.png"))
            BrokenBarhCanvas(tagged,   output_dir=bb_dir + os.sep)
            print(f"\n{DIM}Charts saved to: {dest}{RESET}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        prog="yt-genre",
        description="Predict the music genre of a YouTube video.",
    )
    parser.add_argument("url", help="YouTube URL to analyse")
    parser.add_argument(
        "--keep-charts",
        action="store_true",
        help="Also save PNG charts to ./yt_genre_charts/",
    )
    args = parser.parse_args()

    try:
        run(args.url, keep_charts=args.keep_charts)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Interrupted.{RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{RED}Error: {e}{RESET}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
