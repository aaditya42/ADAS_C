from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from typing import Any, Callable, Generator

from src.utils.logger import get_logger


@contextmanager
def timer(name: str = "block") -> Generator[None, None, None]:
    """Context manager that logs the elapsed time (ms) of a code block.

    Args:
        name: A label for the timed block, used in the log message.
    """
    logger = get_logger("timer")
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = (time.perf_counter() - start) * 1000.0
        logger.debug("%s took %.1f ms", name, elapsed)


def timed(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that logs the elapsed time (ms) of a function call.

    The log message includes both the function name and the measured duration.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = get_logger(f"timer.{func.__name__}")
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000.0
        logger.debug("%s took %.1f ms", func.__qualname__, elapsed)
        return result

    return wrapper
