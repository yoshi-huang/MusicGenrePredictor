# -----------------------------------------------------------------------------
# Config loader — reads config.yaml from backend root
# -----------------------------------------------------------------------------

import os
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH = os.path.join(_ROOT, "config.yaml")


def _load() -> dict:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


cfg = _load()

# Resolve all path values relative to backend root
def _abs(p):
    return os.path.join(_ROOT, p) if not os.path.isabs(p) else p

_raw_paths = cfg["paths"]
paths = {k: _abs(v) for k, v in _raw_paths.items()}
model_cfg = cfg["model"]
audio_cfg = cfg["audio"]
dcbot_cfg = cfg["dcbot"]
