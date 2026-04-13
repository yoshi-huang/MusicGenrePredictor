import logging
from pathlib import Path

def init_logger(level=logging.INFO, log_dir: str = None, filename: str = "train.log"):
    fmt = "%(asctime)s \33[36m[%(levelname)s] \33[0m%(message)s"
    datefmt = "\33[35m%Y-%m-%d \33[32m%H:%M:%S"
    handlers = [logging.StreamHandler()]

    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(Path(log_dir) / filename, mode="w", encoding="utf-8"))
        # 檔案 handler 用純文字格式（不含 ANSI 色碼）
        handlers[-1].setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        ))

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)
    logging.info("log initialization complete")
