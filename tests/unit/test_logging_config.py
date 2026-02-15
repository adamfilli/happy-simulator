"""Unit tests for happysimulator logging configuration."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from unittest import mock

import happysimulator
from happysimulator.logging_config import (
    LOGGER_NAME,
    JsonFormatter,
    _clear_handlers,
    _get_level,
    _get_logger,
)


class TestSilentByDefault:
    """Tests that the library is silent by default."""

    def test_import_produces_no_log_output(self, capfd):
        """Importing happysimulator should not produce any log output."""
        # Force a reload to test import behavior
        import importlib

        importlib.reload(happysimulator)

        captured = capfd.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_logger_has_null_handler(self):
        """The logger should have a NullHandler by default."""
        logger = logging.getLogger(LOGGER_NAME)
        null_handlers = [h for h in logger.handlers if isinstance(h, logging.NullHandler)]
        assert len(null_handlers) >= 1


class TestEnableConsoleLogging:
    """Tests for enable_console_logging function."""

    def test_adds_stream_handler(self):
        """Should add a StreamHandler to the logger."""
        happysimulator.enable_console_logging()

        logger = _get_logger()
        stream_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
        assert len(stream_handlers) >= 1

    def test_sets_level(self):
        """Should set the logger level."""
        happysimulator.enable_console_logging(level="DEBUG")

        logger = _get_logger()
        assert logger.level == logging.DEBUG

    def test_outputs_to_stderr(self, capfd):
        """Should output log messages to stderr."""
        happysimulator.enable_console_logging(level="INFO")

        logger = logging.getLogger(f"{LOGGER_NAME}.test")
        logger.info("test message")

        captured = capfd.readouterr()
        assert "test message" in captured.err

    def test_custom_format(self, capfd):
        """Should respect custom format string."""
        happysimulator.enable_console_logging(level="INFO", format="[CUSTOM] %(message)s")

        logger = logging.getLogger(f"{LOGGER_NAME}.test")
        logger.info("hello")

        captured = capfd.readouterr()
        assert "[CUSTOM] hello" in captured.err


class TestEnableFileLogging:
    """Tests for enable_file_logging function."""

    def test_creates_rotating_file_handler(self, tmp_path):
        """Should create a RotatingFileHandler."""
        log_file = tmp_path / "test.log"
        happysimulator.enable_file_logging(log_file)

        from logging.handlers import RotatingFileHandler

        logger = _get_logger()
        rotating_handlers = [h for h in logger.handlers if isinstance(h, RotatingFileHandler)]
        assert len(rotating_handlers) >= 1

    def test_creates_parent_directories(self, tmp_path):
        """Should create parent directories if they don't exist."""
        log_file = tmp_path / "subdir" / "nested" / "test.log"
        happysimulator.enable_file_logging(log_file)

        assert log_file.parent.exists()

    def test_writes_to_file(self, tmp_path):
        """Should write log messages to file."""
        log_file = tmp_path / "test.log"
        happysimulator.enable_file_logging(log_file, level="INFO")

        logger = logging.getLogger(f"{LOGGER_NAME}.test")
        logger.info("file test message")

        # Flush handlers
        for handler in _get_logger().handlers:
            handler.flush()

        content = log_file.read_text()
        assert "file test message" in content

    def test_respects_max_bytes(self, tmp_path):
        """Should create handler with specified max_bytes."""
        log_file = tmp_path / "test.log"
        handler = happysimulator.enable_file_logging(log_file, max_bytes=1024, backup_count=3)

        assert handler.maxBytes == 1024
        assert handler.backupCount == 3


class TestEnableTimedFileLogging:
    """Tests for enable_timed_file_logging function."""

    def test_creates_timed_rotating_handler(self, tmp_path):
        """Should create a TimedRotatingFileHandler."""
        log_file = tmp_path / "test.log"
        happysimulator.enable_timed_file_logging(log_file)

        from logging.handlers import TimedRotatingFileHandler

        logger = _get_logger()
        timed_handlers = [h for h in logger.handlers if isinstance(h, TimedRotatingFileHandler)]
        assert len(timed_handlers) >= 1

    def test_respects_when_parameter(self, tmp_path):
        """Should create handler with specified 'when' parameter."""
        log_file = tmp_path / "test.log"
        handler = happysimulator.enable_timed_file_logging(log_file, when="H", interval=6)

        assert handler.when == "H"
        assert handler.interval == 6 * 60 * 60  # Converted to seconds


class TestEnableJsonLogging:
    """Tests for enable_json_logging function."""

    def test_outputs_valid_json(self, capfd):
        """Should output valid JSON."""
        happysimulator.enable_json_logging(level="INFO")

        logger = logging.getLogger(f"{LOGGER_NAME}.test")
        logger.info("json test")

        captured = capfd.readouterr()
        # Parse the JSON output
        log_line = captured.err.strip()
        data = json.loads(log_line)

        assert data["message"] == "json test"
        assert data["level"] == "INFO"
        assert "timestamp" in data
        assert "logger" in data

    def test_json_includes_exception(self, capfd):
        """Should include exception info in JSON."""
        happysimulator.enable_json_logging(level="INFO")

        logger = logging.getLogger(f"{LOGGER_NAME}.test")
        try:
            raise ValueError("test error")
        except ValueError:
            logger.exception("caught error")

        captured = capfd.readouterr()
        data = json.loads(captured.err.strip())

        assert "exception" in data
        assert "ValueError" in data["exception"]


class TestEnableJsonFileLogging:
    """Tests for enable_json_file_logging function."""

    def test_writes_json_to_file(self, tmp_path):
        """Should write JSON formatted logs to file."""
        log_file = tmp_path / "test.json"
        happysimulator.enable_json_file_logging(log_file, level="INFO")

        logger = logging.getLogger(f"{LOGGER_NAME}.test")
        logger.info("json file test")

        # Flush handlers
        for handler in _get_logger().handlers:
            handler.flush()

        content = log_file.read_text().strip()
        data = json.loads(content)

        assert data["message"] == "json file test"


class TestConfigureFromEnv:
    """Tests for configure_from_env function."""

    def test_respects_hs_logging_env(self, capfd):
        """Should configure level from HS_LOGGING env var."""
        with mock.patch.dict(os.environ, {"HS_LOGGING": "DEBUG"}, clear=False):
            happysimulator.configure_from_env()

        logger = _get_logger()
        assert logger.level == logging.DEBUG

    def test_respects_hs_log_file_env(self, tmp_path):
        """Should configure file logging from HS_LOG_FILE env var."""
        log_file = tmp_path / "env_test.log"
        with mock.patch.dict(
            os.environ,
            {"HS_LOGGING": "INFO", "HS_LOG_FILE": str(log_file)},
            clear=False,
        ):
            happysimulator.configure_from_env()

        from logging.handlers import RotatingFileHandler

        logger = _get_logger()
        rotating_handlers = [h for h in logger.handlers if isinstance(h, RotatingFileHandler)]
        assert len(rotating_handlers) >= 1

    def test_respects_hs_log_json_env(self, capfd):
        """Should enable JSON logging when HS_LOG_JSON=1."""
        with mock.patch.dict(os.environ, {"HS_LOGGING": "INFO", "HS_LOG_JSON": "1"}, clear=False):
            happysimulator.configure_from_env()

        logger = logging.getLogger(f"{LOGGER_NAME}.test")
        logger.info("json env test")

        captured = capfd.readouterr()
        data = json.loads(captured.err.strip())
        assert data["message"] == "json env test"

    def test_does_nothing_when_no_env_vars(self):
        """Should not add handlers when no env vars are set."""
        initial_count = len(_get_logger().handlers)

        with mock.patch.dict(os.environ, {}, clear=True):
            # Remove relevant env vars
            os.environ.pop("HS_LOGGING", None)
            os.environ.pop("HS_LOG_FILE", None)
            os.environ.pop("HS_LOG_JSON", None)
            happysimulator.configure_from_env()

        # No new handlers added (just NullHandler)
        assert len(_get_logger().handlers) == initial_count


class TestSetLevel:
    """Tests for set_level function."""

    def test_sets_level_by_string(self):
        """Should set level from string."""
        happysimulator.set_level("WARNING")
        assert _get_logger().level == logging.WARNING

    def test_sets_level_by_int(self):
        """Should set level from int."""
        happysimulator.set_level(logging.ERROR)
        assert _get_logger().level == logging.ERROR


class TestSetModuleLevel:
    """Tests for set_module_level function."""

    def test_sets_submodule_level(self):
        """Should set level for submodule."""
        happysimulator.set_module_level("core.simulation", "DEBUG")

        sublogger = logging.getLogger(f"{LOGGER_NAME}.core.simulation")
        assert sublogger.level == logging.DEBUG

    def test_independent_from_root_level(self, capfd):
        """Submodule level can be set to filter more strictly than root."""
        # Enable console with DEBUG level on root
        happysimulator.enable_console_logging(level="DEBUG")
        # But silence distributions module
        happysimulator.set_module_level("quiet_module", "CRITICAL")

        quiet_logger = logging.getLogger(f"{LOGGER_NAME}.quiet_module")
        noisy_logger = logging.getLogger(f"{LOGGER_NAME}.noisy_module")

        quiet_logger.warning("quiet warning")  # Should be filtered
        noisy_logger.debug("noisy debug")  # Should appear

        captured = capfd.readouterr()
        # quiet_module filters at CRITICAL, so WARNING is suppressed
        assert "quiet warning" not in captured.err
        # noisy_module inherits DEBUG from root
        assert "noisy debug" in captured.err


class TestDisableLogging:
    """Tests for disable_logging function."""

    def test_silences_all_output(self, capfd):
        """Should silence all log output."""
        happysimulator.enable_console_logging(level="DEBUG")
        happysimulator.disable_logging()

        logger = logging.getLogger(f"{LOGGER_NAME}.test")
        logger.critical("this should not appear")

        captured = capfd.readouterr()
        assert "this should not appear" not in captured.err

    def test_removes_non_null_handlers(self):
        """Should remove all non-NullHandler handlers."""
        happysimulator.enable_console_logging()
        happysimulator.enable_file_logging(Path(tempfile.gettempdir()) / "test.log")

        happysimulator.disable_logging()

        logger = _get_logger()
        non_null = [h for h in logger.handlers if not isinstance(h, logging.NullHandler)]
        assert len(non_null) == 0


class TestJsonFormatter:
    """Tests for JsonFormatter class."""

    def test_format_basic_record(self):
        """Should format basic log record as JSON."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert data["message"] == "test message"
        assert data["level"] == "INFO"
        assert data["logger"] == "test.logger"
        assert "timestamp" in data

    def test_format_with_exception(self):
        """Should include exception info."""
        formatter = JsonFormatter()
        try:
            raise RuntimeError("test error")
        except RuntimeError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="error occurred",
            args=(),
            exc_info=exc_info,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert "exception" in data
        assert "RuntimeError" in data["exception"]


class TestHelperFunctions:
    """Tests for internal helper functions."""

    def test_get_level_from_string(self):
        """Should convert level string to int."""
        assert _get_level("DEBUG") == logging.DEBUG
        assert _get_level("info") == logging.INFO
        assert _get_level("WARNING") == logging.WARNING

    def test_get_level_from_int(self):
        """Should pass through int levels."""
        assert _get_level(logging.ERROR) == logging.ERROR

    def test_get_level_default_for_invalid(self):
        """Should default to INFO for invalid level strings."""
        assert _get_level("INVALID") == logging.INFO

    def test_clear_handlers_keeps_null_handler(self):
        """_clear_handlers should keep NullHandler."""
        logger = _get_logger()
        logger.addHandler(logging.StreamHandler())

        _clear_handlers()

        null_handlers = [h for h in logger.handlers if isinstance(h, logging.NullHandler)]
        assert len(null_handlers) >= 1
