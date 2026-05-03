# -----------------------------------------------------------------------------
# Config loader
#
# Search order for config.yaml:
#   1. Same directory as this file  (pip install . → site-packages/core/)
#   2. One level up                 (source tree   → backend/config.yaml)
#   3. Two levels up                (fallback)
# -----------------------------------------------------------------------------

import os
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))

_CONFIG_CANDIDATES = [
    os.path.join(_HERE, "config.yaml"),               # installed
    os.path.join(_HERE, "..", "config.yaml"),          # source tree
    os.path.join(_HERE, "..", "..", "config.yaml"),    # extra fallback
]


def _find(candidates: list, name: str) -> str:
    for p in candidates:
        if os.path.isfile(p):
            return os.path.abspath(p)
    raise FileNotFoundError(
        f"{name} not found. Searched:\n" +
        "\n".join(f"  {os.path.abspath(p)}" for p in candidates)
    )


_CONFIG_PATH = _find(_CONFIG_CANDIDATES, "config.yaml")
_CONFIG_DIR  = os.path.dirname(_CONFIG_PATH)


def _load() -> dict:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


cfg = _load()


def _abs(p: str) -> str:
    """Resolve a path relative to the config file's directory."""
    return os.path.join(_CONFIG_DIR, p) if not os.path.isabs(p) else p


_raw_paths = cfg["paths"]
paths     = {k: _abs(v) for k, v in _raw_paths.items()}
model_cfg = cfg["model"]
audio_cfg = cfg["audio"]
dcbot_cfg = cfg["dcbot"]
