# -----------------------------------------------------------------------------
# Music Genre Classifier — GUI entry point
# -----------------------------------------------------------------------------

import os, sys
from math import floor

# Must be set before importing torch or numba to avoid LLVM SVML symbol conflict
os.environ.setdefault("NUMBA_DISABLE_INTEL_SVML", "1")

# Add backend to path so shared core modules are importable
_HERE    = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.join(_HERE, "..", "backend")
sys.path.insert(0, os.path.abspath(_BACKEND))

print("import complete : os sys math")

import numpy as np
print("import complete : numpy")

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
    QLabel, QTextEdit, QProgressBar, QFileDialog, QSlider, QScrollArea
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
print("import complete : QtUI")

from core.model import load_model
from core.audio import GENRES
from ui.charts_qt import BarChartCanvas, BrokenBarhCanvas, SpectrogramCanvas
from ui.worker import AnalysisWorker
from core.config import paths
print("import complete : core modules")

# -----------------------------------------------------------------------------
# main application
# -----------------------------------------------------------------------------

class GenreClassifierApp(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("music genre classifier")
        self.setGeometry(500, 150, 900, 775)
        font = QFont("Consolas", 10)
        self.player = QMediaPlayer()
        self.audio_path = None
        self.setFont(font)
        self.model = load_model("transformer_parms.pth")
        self._build_ui(font)

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
        self.slider.sliderMoved.connect(self._preview_position)
        self.slider.sliderReleased.connect(self._seek_position)
        self.slider_is_pressed = False
        self.slider.sliderPressed.connect(lambda: setattr(self, 'slider_is_pressed', True))
        self.slider.sliderReleased.connect(lambda: setattr(self, 'slider_is_pressed', False))
        self.player.positionChanged.connect(self._update_slider)
        self.player.durationChanged.connect(self._set_slider_range)
        self.player.mediaStatusChanged.connect(self._on_media_status_changed)

        self.result_area = QTextEdit(self)
        self.result_area.setReadOnly(True)
        self.result_area.setFont(font)
        self.result_area.setPlainText("""\
Music Genre Classifier - Usage Guide

This application allows you to classify the genre of a music track either by:
1. Entering a YouTube music URL and clicking "Analyze" to automatically download and analyze the audio.
2. Uploading a local audio file (mp3 or wav) using the "Upload" button for direct analysis.

Key Features:
- Playback of the downloaded/uploaded audio, including play/pause and seek functionality.
- Displays analysis results:
    - A bar chart showing the number of votes for each music genre.
    - A log-mel spectrogram of the input audio.
- Progress tracking with a progress bar and live status updates.

Notes:
- Internet connection is required for YouTube downloads.
- If download fails, check the URL format or try another music video.\
""")

        self.duratio_text = QLabel(self)
        self.duratio_text.setVisible(True)
        self.duratio_text.setAlignment(Qt.AlignCenter)

        self.chart_layout = QHBoxLayout()
        self.vis_layout   = QHBoxLayout()

        top_layout = QHBoxLayout()
        top_layout.addWidget(self.url_input)

        player_layout = QHBoxLayout()
        player_layout.addWidget(self.play_button, 1)
        player_layout.addWidget(self.slider, 5)
        player_layout.addWidget(self.duratio_text, 1)

        left_middle = QVBoxLayout()
        left_middle.addLayout(player_layout)
        right_middle = QVBoxLayout()
        right_middle.addWidget(self.upload_button)

        middle_layout = QHBoxLayout()
        middle_layout.addLayout(left_middle, 1)
        middle_layout.addLayout(right_middle, 1)

        self.main_layout = QVBoxLayout()
        self.main_layout.addLayout(top_layout)
        self.main_layout.addWidget(self.analyze_button)
        self.main_layout.addWidget(self.status_label)
        self.main_layout.addWidget(self.progress_bar)
        self.main_layout.addLayout(middle_layout)
        self.main_layout.addLayout(self.chart_layout)
        self.main_layout.addLayout(self.vis_layout)
        self.main_layout.addWidget(self.result_area)
        self.setLayout(self.main_layout)

    # -------------------------------------------------------------------------
    # UI actions
    # -------------------------------------------------------------------------

    def _upload_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "select audio file", "",
                                              "Audio Files (*.mp3 *.wav)")
        if path:
            self.url_input.clear()
            self.audio_path = path
            self.status_label.setText(f"selected: {os.path.basename(path)}")

    def _start_analysis(self):
        self.result_area.setVisible(True)
        url_text = self.url_input.text().strip()

        if url_text:
            if "https://" not in url_text:
                self.result_area.setPlainText("Error: please enter a valid youtube url or upload a file.")
                return
            is_file, source = False, url_text
        else:
            if not self.audio_path:
                self.result_area.setPlainText("Error: please enter a valid youtube url or upload a file.")
                return
            is_file, source = True, self.audio_path

        self._clear_layout(self.chart_layout)
        self._clear_layout(self.vis_layout)

        self.play_button.setEnabled(False)
        self.slider.setEnabled(False)
        self.player.pause()
        self.play_button.setText("play")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.status_label.setVisible(True)
        self.status_label.setText("preparing...")
        self.analyze_button.setVisible(False)
        self.analyze_button.setEnabled(False)
        self.upload_button.setEnabled(False)
        self.result_area.setPlainText("analyzing... please wait")

        self.worker = AnalysisWorker(source, self.model, is_file)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    # -------------------------------------------------------------------------
    # Worker callbacks
    # -------------------------------------------------------------------------

    def _on_progress(self, percent, stage):
        self.progress_bar.setValue(percent)
        self.status_label.setText(f"{stage}... {percent}%")

    def _on_finished(self, votes, mel_db, tagged):
        self._add_visuals(votes, mel_db, tagged)
        self._reset_ui()

        tmp = paths["audio_temp_file"]
        path = tmp if self.url_input.text().strip() else self.audio_path
        if path and os.path.exists(path):
            self.play_button.setEnabled(False)
            self.slider.setEnabled(False)
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(os.path.abspath(path))))

    def _on_error(self, msg):
        self.result_area.setPlainText(f"Error: {msg}")
        self._reset_ui()

    # -------------------------------------------------------------------------
    # Player controls
    # -------------------------------------------------------------------------

    def _on_media_status_changed(self, status):
        if status == QMediaPlayer.LoadedMedia:
            self.play_button.setEnabled(True)
            self.slider.setEnabled(True)

    def _toggle_play(self):
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
            self.play_button.setText("play")
        else:
            self.player.play()
            self.play_button.setText("pause")

    def _update_slider(self, position):
        t = floor(position / 1000)
        self.duratio_text.setText(f"{t // 60: >2}:{t % 60:0>2}")
        if not self.slider_is_pressed:
            self.slider.setValue(position)

    def _set_slider_range(self, duration):
        self.slider.setRange(0, duration)

    def _preview_position(self, position):
        # 只更新時間顯示，不實際 seek，避免拖動時卡頓
        t = floor(position / 1000)
        self.duratio_text.setText(f"{t // 60: >2}:{t % 60:0>2}")

    def _seek_position(self):
        self.player.setPosition(self.slider.value())

    def _set_position(self, position):
        self.player.setPosition(position)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
            elif item.layout():
                self._clear_layout(item.layout())

    def _add_visuals(self, votes, mel_db, tagged):
        total_votes = np.bincount(votes, minlength=10)
        bar_canvas  = BarChartCanvas(votes=total_votes, genres=GENRES)
        spec_canvas = SpectrogramCanvas(mel_db)

        left = QVBoxLayout()
        left.addWidget(QLabel("votes chart:"))
        left.addWidget(bar_canvas)

        right = QVBoxLayout()
        right.addWidget(QLabel("log-mel spectrogram:"))
        right.addWidget(spec_canvas)

        self.chart_layout.addLayout(left, 1)
        self.chart_layout.addLayout(right, 1)

        seg_canvas = BrokenBarhCanvas(votes=tagged, genres=GENRES)
        seg_canvas.setMinimumWidth(len(tagged) * 10)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setWidget(seg_canvas)
        self.vis_layout.addWidget(scroll)
        self.result_area.setVisible(False)

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
