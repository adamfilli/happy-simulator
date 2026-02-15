"""Logging configuration utilities for happysimulator.

This module provides helper functions to configure logging for the library.
By default, happysimulator is silent (uses NullHandler). Users must explicitly
enable logging using the functions provided here.

Example usage:
    import happysimulator

    # Simple console logging
    happysimulator.enable_console_logging(level="DEBUG")

    # Rotating file logging (prevents disk space issues)
    happysimulator.enable_file_logging("simulation.log", max_bytes=10_000_000)

    # JSON logging for log aggregation (ELK, Datadog, etc.)
    happysimulator.enable_json_logging()

    # Configure from environment variables
    happysimulator.configure_from_env()

Environment variables:
    HS_LOGGING: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    HS_LOG_FILE: Path to log file (enables rotating file logging)
    HS_LOG_JSON: Set to "1" for JSON output
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Literal

__all__ = [
    "configure_from_env",
    "disable_logging",
    "enable_console_logging",
    "enable_file_logging",
    "enable_json_file_logging",
    "enable_json_logging",
    "enable_timed_file_logging",
    "set_level",
    "set_module_level",
]

# Default format strings
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Default rotating file handler settings
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_BACKUP_COUNT = 5

# Logger name for the library
LOGGER_NAME = "happysimulator"

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class JsonFormatter(logging.Formatter):
    """Formats log records as JSON for structured logging.

    Output is compatible with ELK stack, Datadog, and other log aggregators.

    Example output:
        {"timestamp": "2024-01-15T10:30:00.123456Z", "level": "INFO",
         "logger": "happysimulator.core.simulation", "message": "Simulation started"}
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a JSON string."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields if present
        if hasattr(record, "extra"):
            log_data["extra"] = record.extra

        return json.dumps(log_data)


def _get_level(level: str | int) -> int:
    """Convert a level string or int to a logging level constant."""
    if isinstance(level, int):
        return level
    return getattr(logging, level.upper(), logging.INFO)


def _get_logger() -> logging.Logger:
    """Get the happysimulator root logger."""
    return logging.getLogger(LOGGER_NAME)


def _clear_handlers() -> None:
    """Remove and close all handlers from the happysimulator logger except NullHandler."""
    logger = _get_logger()
    for handler in logger.handlers[:]:
        if not isinstance(handler, logging.NullHandler):
            logger.removeHandler(handler)
            handler.close()


def enable_console_logging(
    level: LogLevel | int = "INFO",
    format: str = DEFAULT_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
) -> logging.StreamHandler:
    """Enable console (stderr) logging for happysimulator.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) or int.
        format: Log message format string.
        date_format: Date format string for %(asctime)s.

    Returns:
        The created StreamHandler.

    Example:
        >>> import happysimulator
        >>> happysimulator.enable_console_logging(level="DEBUG")
    """
    logger = _get_logger()
    logger.setLevel(_get_level(level))

    handler = logging.StreamHandler()
    handler.setLevel(_get_level(level))
    handler.setFormatter(logging.Formatter(format, date_format))

    logger.addHandler(handler)
    return handler


def enable_file_logging(
    path: str | Path,
    level: LogLevel | int = "INFO",
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    format: str = DEFAULT_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
) -> RotatingFileHandler:
    """Enable rotating file logging for happysimulator.

    Uses RotatingFileHandler to prevent unbounded disk usage. When the log file
    reaches max_bytes, it is renamed with a numeric suffix and a new file is
    created. Up to backup_count old files are kept.

    Args:
        path: Path to the log file. Parent directories are created automatically.
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) or int.
        max_bytes: Maximum size of each log file in bytes. Default 10 MB.
        backup_count: Number of backup files to keep. Default 5.
        format: Log message format string.
        date_format: Date format string for %(asctime)s.

    Returns:
        The created RotatingFileHandler.

    Example:
        >>> import happysimulator
        >>> happysimulator.enable_file_logging(
        ...     "logs/simulation.log",
        ...     max_bytes=50_000_000,  # 50 MB
        ...     backup_count=10,
        ... )
    """
    logger = _get_logger()
    logger.setLevel(_get_level(level))

    # Create parent directories if needed
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        path,
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    handler.setLevel(_get_level(level))
    handler.setFormatter(logging.Formatter(format, date_format))

    logger.addHandler(handler)
    return handler


def enable_timed_file_logging(
    path: str | Path,
    level: LogLevel | int = "INFO",
    when: str = "midnight",
    interval: int = 1,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    format: str = DEFAULT_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
) -> TimedRotatingFileHandler:
    """Enable time-based rotating file logging for happysimulator.

    Uses TimedRotatingFileHandler to rotate logs at specified intervals.

    Args:
        path: Path to the log file. Parent directories are created automatically.
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) or int.
        when: Rotation interval type. Options:
            - 'S': Seconds
            - 'M': Minutes
            - 'H': Hours
            - 'D': Days
            - 'midnight': Roll over at midnight
            - 'W0'-'W6': Roll over on weekday (0=Monday)
        interval: Interval count (e.g., interval=2, when='D' means every 2 days).
        backup_count: Number of backup files to keep. Default 5.
        format: Log message format string.
        date_format: Date format string for %(asctime)s.

    Returns:
        The created TimedRotatingFileHandler.

    Example:
        >>> import happysimulator
        >>> happysimulator.enable_timed_file_logging(
        ...     "logs/simulation.log",
        ...     when="midnight",
        ...     backup_count=30,  # Keep 30 days
        ... )
    """
    logger = _get_logger()
    logger.setLevel(_get_level(level))

    # Create parent directories if needed
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    handler = TimedRotatingFileHandler(
        path,
        when=when,
        interval=interval,
        backupCount=backup_count,
    )
    handler.setLevel(_get_level(level))
    handler.setFormatter(logging.Formatter(format, date_format))

    logger.addHandler(handler)
    return handler


def enable_json_logging(
    level: LogLevel | int = "INFO",
) -> logging.StreamHandler:
    """Enable JSON console logging for happysimulator.

    Outputs structured JSON logs to stderr, suitable for log aggregation
    systems like ELK stack, Datadog, Splunk, etc.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) or int.

    Returns:
        The created StreamHandler with JsonFormatter.

    Example:
        >>> import happysimulator
        >>> happysimulator.enable_json_logging()
        # Output: {"timestamp": "2024-01-15T10:30:00Z", "level": "INFO", ...}
    """
    logger = _get_logger()
    logger.setLevel(_get_level(level))

    handler = logging.StreamHandler()
    handler.setLevel(_get_level(level))
    handler.setFormatter(JsonFormatter())

    logger.addHandler(handler)
    return handler


def enable_json_file_logging(
    path: str | Path,
    level: LogLevel | int = "INFO",
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
) -> RotatingFileHandler:
    """Enable JSON rotating file logging for happysimulator.

    Combines rotating file handler with JSON formatting for structured logs
    that can be processed by log aggregation pipelines.

    Args:
        path: Path to the log file. Parent directories are created automatically.
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) or int.
        max_bytes: Maximum size of each log file in bytes. Default 10 MB.
        backup_count: Number of backup files to keep. Default 5.

    Returns:
        The created RotatingFileHandler with JsonFormatter.

    Example:
        >>> import happysimulator
        >>> happysimulator.enable_json_file_logging("logs/simulation.json")
    """
    logger = _get_logger()
    logger.setLevel(_get_level(level))

    # Create parent directories if needed
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        path,
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    handler.setLevel(_get_level(level))
    handler.setFormatter(JsonFormatter())

    logger.addHandler(handler)
    return handler


def configure_from_env() -> None:
    """Configure logging from environment variables.

    Reads the following environment variables:
        - HS_LOGGING: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        - HS_LOG_FILE: Path to log file (enables rotating file logging)
        - HS_LOG_JSON: Set to "1" for JSON output format

    If no environment variables are set, this function does nothing.

    Example:
        # In shell:
        export HS_LOGGING=DEBUG
        export HS_LOG_FILE=simulation.log

        # In Python:
        >>> import happysimulator
        >>> happysimulator.configure_from_env()
    """
    level = os.environ.get("HS_LOGGING", "").upper()
    log_file = os.environ.get("HS_LOG_FILE", "")
    use_json = os.environ.get("HS_LOG_JSON", "") == "1"

    if not level and not log_file:
        return

    level = level or "INFO"

    if use_json:
        if log_file:
            enable_json_file_logging(log_file, level=level)
        else:
            enable_json_logging(level=level)
    else:
        if log_file:
            enable_file_logging(log_file, level=level)
        else:
            enable_console_logging(level=level)


def set_level(level: LogLevel | int) -> None:
    """Set the global log level for happysimulator.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) or int.

    Example:
        >>> import happysimulator
        >>> happysimulator.enable_console_logging()
        >>> happysimulator.set_level("DEBUG")
    """
    _get_logger().setLevel(_get_level(level))


def set_module_level(module: str, level: LogLevel | int) -> None:
    """Set the log level for a specific happysimulator submodule.

    Allows fine-grained control over which parts of the library produce logs.

    Args:
        module: Module name relative to happysimulator (e.g., "core.simulation").
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) or int.

    Example:
        >>> import happysimulator
        >>> happysimulator.enable_console_logging(level="INFO")
        >>> happysimulator.set_module_level("core.simulation", "DEBUG")  # Verbose sim
        >>> happysimulator.set_module_level("distributions", "WARNING")  # Quiet dists
    """
    full_name = f"{LOGGER_NAME}.{module}"
    logging.getLogger(full_name).setLevel(_get_level(level))


def disable_logging() -> None:
    """Completely disable logging for happysimulator.

    Removes all handlers and sets level to CRITICAL+1 to silence all output.

    Example:
        >>> import happysimulator
        >>> happysimulator.enable_console_logging()
        >>> # ... some logging happens ...
        >>> happysimulator.disable_logging()  # Silence
    """
    logger = _get_logger()
    _clear_handlers()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.CRITICAL + 1)
