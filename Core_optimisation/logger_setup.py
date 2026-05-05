"""Shared logger configuration for the restoration optimisation pipeline."""

import logging
import os
from datetime import datetime


def setup_logger(log_dir="logs", run_label=None):
    """
    Configure the 'resto_prio' named logger.

    - FileHandler at DEBUG level  → logs/<timestamp>[_label].log
    - StreamHandler at WARNING level → console only

    Safe to call multiple times; duplicate handlers are not added.

    Args:
        log_dir:   Directory for log files (created if absent).
        run_label: Optional label appended to the log filename.

    Returns:
        str | None: Absolute path to the log file, or None if already configured.
    """
    logger = logging.getLogger("resto_prio")

    # Avoid adding duplicate handlers if already configured
    if logger.handlers:
        return None

    logger.setLevel(logging.DEBUG)

    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{run_label}" if run_label else ""
    log_path = os.path.join(log_dir, f"run_{timestamp}{suffix}.log")

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return log_path
