import yaml
from pathlib import Path

def load_config(path: str = "configs/config.yaml") -> dict:
    with open(Path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
