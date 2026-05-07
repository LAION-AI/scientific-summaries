"""Utility helpers for retries, logging, text normalization, and formatting."""

from __future__ import annotations

import logging
import re
import sys
import time
from pathlib import Path
from typing import Callable, TypeVar


T = TypeVar("T")


class RetryableError(RuntimeError):
    """Raised when a transient operation exhausts all retry attempts."""


def setup_logging(log_path: Path) -> None:
    """Configure root logging to both stdout and a log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def retry_call(
    func: Callable[[], T],
    *,
    max_retries: int,
    base_delay_seconds: float,
    operation_name: str,
) -> T:
    """Retry transient operations with exponential backoff.

    Broad exception handling is intentionally used because transient network or
    remote service failures may surface as different exception types depending
    on dependency versions and the runtime environment.
    """
    attempt = 0

    while True:
        try:
            return func()
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            attempt += 1
            if attempt > max_retries:
                raise RetryableError(
                    f"{operation_name} failed after {max_retries} retries: {exc}"
                ) from exc

            delay = base_delay_seconds * (2 ** (attempt - 1))
            logging.warning(
                "%s failed on attempt %s/%s (%s). Retrying in %.1f seconds...",
                operation_name,
                attempt,
                max_retries,
                exc,
                delay,
            )
            time.sleep(delay)


def slugify(value: str) -> str:
    """Create a filesystem-safe slug from an arbitrary string."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    slug = re.sub(r"_+", "_", slug)
    return slug.strip("._") or "unknown"


def normalize_text(value: object) -> str:
    """Normalize any value into a safe string representation."""
    if value is None:
        return ""
    return str(value)


def render_chatml(user_prompt: str, assistant_response: str) -> str:
    """Render a user-assistant exchange in ChatML format."""
    return (
        "<|im_start|>user\n"
        f"{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{assistant_response}<|im_end|>"
    )