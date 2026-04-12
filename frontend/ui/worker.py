# -----------------------------------------------------------------------------
# Background analysis worker thread (GUI only)
# -----------------------------------------------------------------------------

import os
import numpy as np
import torch
import torch.nn.functional as F

from PyQt5.QtCore import QThread, pyqtSignal

from core.audio import youtube_download, process_audio
from core.config import paths


class AnalysisWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(np.ndarray, np.ndarray)
    error    = pyqtSignal(str)

    TMP_PATH = None

    def __init__(self, source_path: str, model: torch.nn.Module, is_file: bool):
        super().__init__()
        self.source_path = source_path
        self.is_file     = is_file
        self.model       = model
        self.TMP_PATH    = paths["audio_temp_file"]
        os.makedirs(os.path.dirname(self.TMP_PATH), exist_ok=True)

    def _emit_download(self, p, stage):
        if "converting" in stage:
            self.progress.emit(45, stage)
        else:
            self.progress.emit(int(p * 0.4), stage)

    def _emit_process(self, p, stage):
        self.progress.emit(int(50 + p * 0.45), stage)

    def run(self):
        try:
            if self.is_file:
                audio_path = self.source_path
            else:
                youtube_download(self.source_path, self.TMP_PATH,
                                 progress_callback=self._emit_download)
                audio_path = self.TMP_PATH

            mel, mel_db = process_audio(audio_path, progress_callback=self._emit_process)
            x = torch.tensor(mel).float()

            with torch.no_grad():
                pred = self.model(x)

            votes = F.softmax(pred, dim=1).argmax(dim=1)
            self.progress.emit(100, "done")
            self.finished.emit(votes.numpy(), mel_db)
        except Exception as e:
            self.error.emit(str(e))
