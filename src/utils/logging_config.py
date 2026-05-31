"""Unified logging configuration used across scripts, API, and tests."""
from __future__ import annotations

import logging
import os
import sys


def get_logger(name: str, level: str | None = None) -> logging.Logger:
    """
    Return a configured logger.

    The log level is taken from the LOG_LEVEL environment variable
    (default: INFO). Output goes to stdout in a structured single-line format,
    which Cloud Run captures and indexes automatically.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    level_str = level or os.environ.get("LOG_LEVEL", "INFO")
    logger.setLevel(getattr(logging, level_str.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    return logger
