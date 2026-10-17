"""Logging setup.

User messages are scripted demo examples only, and are never logged at INFO. The
rule from docs/DECIDER.md section 7 is that real user text does not get recorded;
the `--demo`/`--interactive` flows print to the terminal and that is all.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_LOGGER_NAME = "crisis_triage"


def setup_logging(
    level: str | int | None = None,
    *,
    log_file: Path | None = None,
    quiet: bool = False,
) -> logging.Logger:
    """Configure the package logger once and return it."""
    logger = logging.getLogger(_LOGGER_NAME)

    if level is None:
        level = os.environ.get("CRISIS_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")

    if not quiet:
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(formatter)
        # Routing decisions go to stdout for the demo; details to stderr.
        console.setLevel(level)
        logger.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

    return logger


def get_logger(suffix: str | None = None) -> logging.Logger:
    return logging.getLogger(f"{_LOGGER_NAME}.{suffix}" if suffix else _LOGGER_NAME)