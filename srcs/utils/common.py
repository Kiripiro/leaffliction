from __future__ import annotations

import logging
import sys

__all__ = ["setup_logging", "get_logger"]


def get_logger(name: str) -> logging.Logger:
    if not logging.getLogger().handlers:
        setup_logging()
    return logging.getLogger(name)


def setup_logging(level: str = "INFO") -> None:
    """Minimal colored logging.

    level: logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    Suppresses noisy third-party DEBUG logs.
    """
    root = logging.getLogger()
    if root.handlers:
        return

    RESET = "\033[0m"
    COLORS = {
        logging.DEBUG: "\033[36m",  # Cyan
        logging.INFO: "\033[32m",  # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",  # Red
        logging.CRITICAL: "\033[1;37;41m",  # Bold white on red background
    }

    class _Fmt(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            if record.levelno in COLORS:
                record.levelname = f"{COLORS[record.levelno]}{record.levelname}{RESET}"
            return super().format(record)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        _Fmt("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.addHandler(handler)

    for noisy in ("matplotlib", "matplotlib.font_manager", "PIL", "fontTools"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
