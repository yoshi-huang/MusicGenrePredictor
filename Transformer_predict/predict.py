# -----------------------------------------------------------------------------
# hint
# -----------------------------------------------------------------------------
def hint_declaration():
    hint = \
    """
    Music Genre Classifier - Usage Guide

    This application allows you to classify the genre of a music track either by:
    1. 🔗 Entering a YouTube music URL and clicking "Analyze" to automatically download and analyze the audio.
    2. 📁 Uploading a local audio file (mp3 or wav) using the "Upload" button for direct analysis.

    Key Features:
    - 🎧 Playback of the downloaded/uploaded audio, including play/pause and seek functionality.
    - 📊 Displays analysis results:
        - A bar chart showing the number of votes for each music genre.
        - A log-mel spectrogram of the input audio.
        - A textual summary of the most likely predicted genre.
    - 📶 Progress tracking with a progress bar and live status updates.

    Notes:
    - Internet connection is required for YouTube downloads.
    - If download fails, check the URL format or try another music video.
    """
    print(hint)
    
hint_declaration()

import os
import sys
import math
print("import complete : os sys math ")

import torch
import torch.nn as nn
import torch.nn.functional as F
print("import complete : pytorch ")

import matplotlib
import numpy as np
from turtledemo.chaos import onscreenclick
matplotlib.use("Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
print("import complete : matplotlib numpy ")

import librosa
import librosa.display
print("import complete : librosa ")

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
    QLabel, QTextEdit, QProgressBar, QSizePolicy, QFileDialog, QSlider
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
print("import complete : QtUI ")

from yt_dlp import YoutubeDL
print("import complete : youtube download ")

# -----------------------------------------------------------------------------
# youtube downloader with progress callback (0‑100)
# -----------------------------------------------------------------------------
def youtube_download(url: str, output_path: str, progress_callback=None) -> None:
    """download the audio track with yt_dlp and report progress (0-100)."""

    def hook(d):
        if d["status"] == "downloading" and progress_callback:
            total_bytes = d.get("total_bytes") or d.get("_total_bytes_estimate")
            downloaded = d.get("downloaded_bytes", 0)
            if total_bytes:
                percent = int(downloaded / total_bytes * 100)
                progress_callback(percent, "downloading")
        elif d["status"] == "finished" and progress_callback:
            progress_callback(100, "downloading")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_path.split(".")[0]}',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'progress_hooks': [hook],
        'noplaylist': True
    }

    with YoutubeDL(ydl_opts) as ydl:
        try: os.remove(r"audio_temp_save\temp_audio.mp3")
        except: pass
        ydl.download([url])

        
# -----------------------------------------------------------------------------
# audio preprocessing with granular progress (0‑100)
# -----------------------------------------------------------------------------
def process_audio(path: str, segment_frames: int = 128, progress_callback=None) -> np.ndarray:
    """
    load an audio file and return a (frames, 128) log-mel array.

    progress_callback(percent:int, stage:str) is called at several points.
    """

    if progress_callback:
        progress_callback(0, "loading audio")
    y_audio, sr = librosa.load(path, sr=22050)

    if progress_callback:
        progress_callback(25, "computing mel spectrogram")
    mel = librosa.feature.melspectrogram(y=y_audio, sr=sr, hop_length=512)

    if progress_callback:
        progress_callback(50, "converting to log-mel")
    mel_db = librosa.power_to_db(mel, ref=np.max)

    segments = []
    for i in range(0, mel_db.shape[1] - segment_frames + 1, segment_frames):
        segment = mel_db[:, i:i+segment_frames]
        segments.append(segment)

    if progress_callback:
        progress_callback(75, "trimming silence")
    segments = np.array(segments)
    segments = np.transpose(segments, (0, 2, 1))

    if progress_callback:
        progress_callback(100, "processing audio done")
    # Transformer need (batch_size, seq_len, feature_dim)
    return list(segments), mel_db

# -----------------------------------------------------------------------------
# transformer encoder definition
# -----------------------------------------------------------------------------
class PositionEncoder(nn.Module):

    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10_000.0)/d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class TransformerEncoder(nn.Module):

    def __init__(self,
                 input_dim, embed_dim,
                 seq_len,
                 num_heads, num_layers,
                 FFN_dim,
                 dropout
                 ):
        super().__init__()

        # self.EmbeddingBlock = nn.Embedding(input_dim, embed_dim)
        self.EmbeddingBlock = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.Dropout(dropout)
            )
        self.PositionEncoder = PositionEncoder(embed_dim, seq_len, dropout)
        self.EncoderLayer = nn.TransformerEncoderLayer(
            d_model = embed_dim,
            nhead = num_heads,
            dim_feedforward = FFN_dim,
            dropout = dropout,
            activation = 'relu',
            batch_first = True
            )
        self.TransformerEncoder = nn.TransformerEncoder(self.EncoderLayer, num_layers=num_layers)
        self.Classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 10)
            )
        
    def forward(self, x):

        x = self.EmbeddingBlock(x)
        x = self.PositionEncoder(x)
        x = self.TransformerEncoder(x)
        x = x.mean(dim=1)

        return self.Classifier(x)

# -----------------------------------------------------------------------------
# matplotlib canvas helpers
# -----------------------------------------------------------------------------
class BarChartCanvas(FigureCanvas):
    def __init__(self, votes: np.ndarray, genres: list[str], parent=None):
        fig, self.ax = plt.subplots(figsize=(5, 3), dpi=100)
        super().__init__(fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(300)
        self.plot(votes, genres)

    def plot(self, votes, genres):
        votes = np.array(votes)
        votes = 100*votes/np.sum(votes)
        self.ax.clear()
        bars = self.ax.bar(genres, votes)
        winner = int(votes.argmax())
        bars[winner].set_color("tab:red")

        for bar, value in zip(bars, votes):
            height = bar.get_height()
            self.ax.text(bar.get_x() + bar.get_width() / 2, height + 3,
                    f"{value: <4.1f}%" if value>0 else "", ha='center', va='bottom', fontsize=7)
        
        self.ax.set_ylabel("possible (%)")
        self.ax.set_ylim(0,100)
        self.ax.set_title("genre prediction votes")
        self.ax.tick_params(axis="x", rotation=45)
        self.figure.tight_layout()
        self.draw()

class SpectrogramCanvas(FigureCanvas):
    def __init__(self, mel_db: np.ndarray, sr: int = 22050, parent=None):
        fig, self.ax = plt.subplots(figsize=(5, 3), dpi=100)
        super().__init__(fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(300)
        self.plot(mel_db, sr)

    def plot(self, mel_db, sr):
        self.ax.clear()
        img = librosa.display.specshow(mel_db, sr=sr, hop_length=512,
                                       y_axis='mel', x_axis='time', ax=self.ax)
        self.ax.set_title('log-mel spectrogram')
        self.figure.colorbar(img, ax=self.ax, format='%+2.0f dB')
        self.figure.tight_layout()
        self.draw()

# -----------------------------------------------------------------------------
# worker thread
# -----------------------------------------------------------------------------
class AnalysisWorker(QThread):
    progress = pyqtSignal(int, str)                     # overall percent 0‑100, stage label
    finished = pyqtSignal(str, np.ndarray, np.ndarray)  # result text
    error = pyqtSignal(str)                             # error message

    def __init__(self, source_path: str, model: torch.nn.Module, is_file: bool):
        super().__init__()
        self.source_path = source_path
        self.is_file = is_file
        self.model = model
        self.tmp_path = "audio_temp_save/temp_audio.mp3"
        os.makedirs(os.path.dirname(self.tmp_path), exist_ok=True)
        self.genres = ["blues", "classical", "country", "disco", "hiphop",
                       "jazz", "metal", "pop", "reggae", "rock"]

    # helper to map stage‑specific percent to global percent
    def _emit_download(self, p, stage):
        # map 0‑100 -> 0‑70
        self.progress.emit(int(p * 0.7), stage)

    def _emit_process(self, p, stage):
        # map 0‑100 -> 70‑100
        self.progress.emit(int(70 + p * 0.3), stage)

    def run(self):

        # download youtube url or upload file
        if self.is_file:
            audio_path = self.source_path
        else:
            youtube_download(self.source_path, self.tmp_path, progress_callback=self._emit_download)
            audio_path = self.tmp_path

        mel, mel_db = process_audio(audio_path, progress_callback=self._emit_process)

        x = torch.tensor(mel).float()  # (1, T, 128)

        with torch.no_grad():
            pred = self.model(x)

        probs = F.softmax(pred, dim=1)
        votes = probs.argmax(dim=1)  # (frames,)
        total_votes = np.bincount(votes.numpy(),minlength=10)
        winner = total_votes.argmax()

        result_lines = ["prediction result:"]
        for i in range(len(self.genres)):
            v = int(total_votes[i])
            marker = "<-- predicted" if i == winner else ""
            result_lines.append(f"{self.genres[i]:10s} : {v:3d} votes {marker}")
        result_lines.append("")
        result_lines.append(f"most likely genre: {self.genres[winner]}")
        self.progress.emit(100, "done")
        # transpose for display
        self.finished.emit("\n".join(result_lines), total_votes, np.transpose(mel_db))

# -----------------------------------------------------------------------------
# main application
# -----------------------------------------------------------------------------
class GenreClassifierApp(QWidget):
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("music genre classifier")
        self.setGeometry(500, 50, 900, 900)
        font = QFont("Consolas", 10)
        self.player = QMediaPlayer()
        self.audio_path = None
        self.setFont(font)
        self.model = self._load_model()
        self._build_ui(font)

    def _load_model(self):
        model = TransformerEncoder(128, 128, 128, 4, 6, 256, 0.0).to("cpu")
        model.load_state_dict(torch.load("transformer_parms.pth", map_location="cpu", weights_only=True))
        
        model.eval()
        return model

    def _build_ui(self, font):
        self.url_input = QLineEdit(self)
        self.url_input.setPlaceholderText("enter youtube music url")

        self.upload_button = QPushButton("upload file", self)
        self.upload_button.clicked.connect(self._upload_file)

        self.analyze_button = QPushButton("analyze", self)
        self.analyze_button.clicked.connect(self._start_analysis)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setVisible(False)

        self.status_label = QLabel(self)
        self.status_label.setVisible(True)
        self.status_label.setAlignment(Qt.AlignCenter)

        self.play_button = QPushButton("play", self)
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self._toggle_play)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderReleased.connect(lambda: self._set_position(self.slider.value()))

        self.result_area = QTextEdit(self)
        self.result_area.setReadOnly(True)
        self.result_area.setFont(font)

        self.vis_layout = QVBoxLayout()

        # top layout: url and button
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.url_input)

        # player layout: music player
        player_layout = QHBoxLayout()
        player_layout.addWidget(self.play_button, 1)
        player_layout.addWidget(self.slider, 4)

        # left middle: visuals
        left_middle_layout = QVBoxLayout()
        left_middle_layout.addLayout(player_layout)
        left_middle_layout.addWidget(QLabel("result:"))

        
        # right middle: visuals
        right_middle_layout = QVBoxLayout()
        right_middle_layout.addWidget(self.upload_button)
        right_middle_layout.addWidget(QLabel("votes chart and spectrogram:"))

        middle_layout = QHBoxLayout()
        middle_layout.addLayout(left_middle_layout, 1)
        middle_layout.addLayout(right_middle_layout, 1)

        # left bottom: result
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.result_area)

        # right bottom: visuals
        right_layout = QVBoxLayout()
        right_layout.addLayout(self.vis_layout)

        bottom_layout = QHBoxLayout()
        bottom_layout.addLayout(left_layout, 1)
        bottom_layout.addLayout(right_layout, 1)

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.analyze_button)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.progress_bar)
        main_layout.addLayout(middle_layout)
        main_layout.addLayout(bottom_layout)

        self.setLayout(main_layout)
    
    def _upload_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "select audio file", "", "Audio Files (*.mp3 *.wav)")
        if file_path:
            self.url_input.clear()
            self.audio_path = file_path
            self.status_label.setText(f"selected: {os.path.basename(file_path)}")
            self.status_label.setVisible(True)

    def _start_analysis(self):
        
        if self.url_input.text().strip():
            is_file = False
            source = self.url_input.text().strip() 
        else:
            is_file = True
            source = self.audio_path

        print(self.url_input.text().strip(), source)
        if not source :
            self.result_area.setPlainText("Error: please enter a valid youtube url or upload a file.")
            return
        if self.url_input.text().strip() and (not "https://" in source):
            self.result_area.setPlainText("Error: please enter a valid youtube url or upload a file.")
            return
        
        self._clear_visuals()
        self.play_button.setEnabled(False)
        self.slider.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.status_label.setVisible(True)
        self.status_label.setText(" ")
        self.analyze_button.setVisible(False)
        self.status_label.setText("preparing...")
        self.result_area.setPlainText("analyzing... please wait")
        self.analyze_button.setEnabled(False)
        self.upload_button.setEnabled(False)

        self.worker = AnalysisWorker(source, self.model, is_file)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, percent, stage):
        self.progress_bar.setValue(percent)
        self.status_label.setText(f"{stage}... {percent}%")

    def _on_finished(self, text, votes, mel_db):
        self.result_area.setPlainText(text)
        self._add_visuals(votes, mel_db)
        self._reset_ui()
        # enable play
        if self.audio_path or os.path.exists(r"audio_temp_save/temp_audio.mp3"):
            path = r"audio_temp_save/temp_audio.mp3" if self.url_input.text().strip() else self.audio_path
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(os.path.abspath(path))))
            self.play_button.setEnabled(True)
            self.slider.setEnabled(True)

    def _on_error(self, msg):
        self.result_area.setPlainText(f"Error: {msg}")
        self._reset_ui()

    def _toggle_play(self):
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
            self.play_button.setText("play")
        else:
            self.player.play()
            self.play_button.setText("pause")

    def _update_slider(self, position):
        self.slider.setValue(position)

    def _set_slider_range(self, duration):
        self.slider.setRange(0, duration)

    def _set_position(self, position):
        self.player.setPosition(position*1000)

    def _clear_visuals(self):
        while self.vis_layout.count():
            child = self.vis_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def _add_visuals(self, votes, mel_db):
        bar_canvas = BarChartCanvas(votes=votes, genres=[
            "blues", "classical", "country", "disco", "hiphop",
            "jazz", "metal", "pop", "reggae", "rock"])
        spec_canvas = SpectrogramCanvas(mel_db)
        self.vis_layout.addWidget(bar_canvas)
        self.vis_layout.addWidget(QLabel("input log-mel spectrogram:"))
        self.vis_layout.addWidget(spec_canvas)

    def _reset_ui(self):
        self.progress_bar.setVisible(False)
        self.status_label.setVisible(True)
        self.status_label.setText(" ")
        self.analyze_button.setVisible(True)
        self.analyze_button.setEnabled(True)
        self.upload_button.setEnabled(True)

# -----------------------------------------------------------------------------
# entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GenreClassifierApp()
    window.show()
    print("system is running...")
    sys.exit(app.exec_())