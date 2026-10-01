"""Seeding, logging and small shared helpers."""
from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

_LOGGER_NAME = "gfb"


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG the pipeline touches.

    Called once per model run so that model A and model B start from identical
    conditions. `torch` is imported lazily so that data-only tooling does not need it.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:  # data inspection without torch installed
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> "Any":
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_device() -> str:
    try:
        import torch
    except ImportError:
        return "torch not installed"
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return (
            f"cuda:0 {props.name} "
            f"({props.total_memory / 1024 ** 3:.1f} GB, torch {torch.__version__})"
        )
    return f"cpu (torch {torch.__version__})"


def setup_logging(log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    # Windows consoles are frequently cp1252; never let a German umlaut kill a run.
    if hasattr(stream.stream, "reconfigure"):
        try:
            stream.stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    logger.addHandler(stream)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(_LOGGER_NAME)


def section(title: str) -> None:
    log = get_logger()
    log.info("")
    log.info("=" * 78)
    log.info(title.upper())
    log.info("=" * 78)


class _NpEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def save_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, cls=_NpEncoder), encoding="utf-8"
    )


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def environment_report() -> dict[str, Any]:
    """Captured into every run so results can be traced to an environment."""
    import platform

    rep: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    for mod in ("numpy", "pandas", "sklearn", "torch", "transformers"):
        try:
            m = __import__(mod)
            rep[mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            rep[mod] = "not installed"
    try:
        import torch

        rep["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            rep["cuda_device"] = torch.cuda.get_device_name(0)
            rep["cuda_version"] = torch.version.cuda
    except ImportError:
        rep["cuda_available"] = False
    return rep
