from __future__ import annotations

import logging
import os
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger with consistent formatting.

    The log level is read from the ``PERCEPTION_LOG_LEVEL`` environment
    variable (default: ``INFO``). Accepted values: ``DEBUG``, ``INFO``,
    ``WARNING``, ``ERROR``, ``CRITICAL``.

    Args:
        name: Usually ``__name__`` of the calling module.

    Returns:
        A ready-to-use :class:`logging.Logger` instance.
    """
    level_name = os.environ.get("PERCEPTION_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        fmt = (
            "[%(asctime)s.%(msecs)03d] %(levelname)-8s %(name)s "
            "- %(message)s"
        )
        formatter = logging.Formatter(fmt, datefmt="%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
