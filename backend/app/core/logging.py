"""Logging configuration."""

from __future__ import annotations

import logging

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once, using a readable single-line format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=LOG_FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    # Third-party loggers that are noisy in development.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Small helper so modules do not import ``logging`` conventions ad hoc."""
    return logging.getLogger(name)
