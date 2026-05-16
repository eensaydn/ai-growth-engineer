"""Shared utility helpers."""
import logging
import sys
from pathlib import Path

import numpy as np


def get_logger(name: str = "ikas") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                            datefmt="%H:%M:%S")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logger


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize(value, lo: float, hi: float):
    """Linearly map value from [lo, hi] to [0, 1], clipped."""
    if hi == lo:
        return np.zeros_like(value, dtype=float) if hasattr(value, "__len__") else 0.0
    arr = (np.asarray(value, dtype=float) - lo) / (hi - lo)
    return np.clip(arr, 0.0, 1.0)
