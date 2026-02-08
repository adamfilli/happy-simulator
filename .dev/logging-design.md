# Logging System Improvement Plan

## Problem Summary

The current logging setup in `happysimulator/__init__.py` violates Python library best practices:

1. **Forces configuration at import** - Calls `basicConfig()` when library is imported
2. **No log rotation** - Uses `FileHandler` without size limits (disk space risk)
3. **Hardcoded output** - Always writes to `happysimulator.log` + console
4. **No NullHandler** - Libraries should be silent by default
5. **Limited flexibility** - No JSON format, no per-module control

## Solution Overview

Follow the pattern used by `urllib3`, `requests`, and other well-designed Python libraries:

- **Silent by default** - Add `NullHandler` to root logger
- **User controls configuration** - Remove `basicConfig()` from import
- **Easy helpers** - Provide `enable_console_logging()`, `enable_file_logging()`, etc.
- **Env var support** - `configure_from_env()` for easy shell-based configuration

## Files to Modify/Create

| File | Action |
|------|--------|
| `happysimulator/__init__.py` | Remove `basicConfig()`, add `NullHandler`, export config functions |
| `happysimulator/logging_config.py` | **NEW** - All logging configuration utilities |
| `tests/unit/test_logging_config.py` | **NEW** - Unit tests for logging config |
| `tests/conftest.py` | Add logging reset fixture |
| `CLAUDE.md` | Update logging documentation |

## Implementation Details

### 1. Modify `happysimulator/__init__.py`

**Remove** (lines 5-26):
```python
import logging
import os

level = os.environ.get("HS_LOGGING", "INFO")
# ... get_logging_level function ...
logging.basicConfig(...)
```

**Replace with**:
```python
import logging

# Library best practice: silent by default
logging.getLogger("happysimulator").addHandler(logging.NullHandler())
```

**Add exports** at end of file:
```python
from happysimulator.logging_config import (
    enable_console_logging,
    enable_file_logging,
    enable_json_logging,
    configure_from_env,
    set_level,
    set_module_level,
    disable_logging,
)
```

### 2. Create `happysimulator/logging_config.py`

New module providing:

| Function | Purpose |
|----------|---------|
| `enable_console_logging(level="INFO")` | Stream to stderr |
| `enable_file_logging(path, max_bytes=10MB, backup_count=5)` | Rotating file handler |
| `enable_timed_file_logging(path, when="midnight")` | Time-based rotation |
| `enable_json_logging()` | Structured JSON for log aggregators |
| `enable_json_file_logging(path)` | JSON to rotating file |
| `configure_from_env()` | Configure via env vars (`HS_LOGGING`, `HS_LOG_FILE`, `HS_LOG_JSON`) |
| `set_level(level)` | Set global level |
| `set_module_level(module, level)` | Per-module control |
| `disable_logging()` | Silence completely |

Key features:
- `RotatingFileHandler` with configurable size limits (default 10MB, 5 backups)
- `TimedRotatingFileHandler` for daily rotation
- `JsonFormatter` class for structured logging (ELK, Datadog compatible)
- Creates parent directories automatically

### 3. Usage Examples

**Simple console logging:**
```python
import happysimulator
happysimulator.enable_console_logging(level="DEBUG")
```

**Rotating file (prevents disk issues):**
```python
happysimulator.enable_file_logging(
    "logs/simulation.log",
    max_bytes=50_000_000,  # 50 MB
    backup_count=10,
)
```

**JSON for log aggregation:**
```python
happysimulator.enable_json_logging()
# Output: {"timestamp": "2024-01-15T10:30:00Z", "level": "INFO", ...}
```

**Environment variable configuration:**
```bash
export HS_LOGGING=DEBUG
export HS_LOG_FILE=simulation.log
```
```python
happysimulator.configure_from_env()
```

**Per-module debugging:**
```python
happysimulator.enable_console_logging(level="INFO")
happysimulator.set_module_level("core.simulation", "DEBUG")  # Debug only sim loop
happysimulator.set_module_level("distributions", "WARNING")  # Silence distributions
```

### 4. Update `tests/conftest.py`

Add fixture to reset logging between tests:

```python
@pytest.fixture(autouse=True)
def reset_happysimulator_logging():
    """Reset logging state before each test."""
    import logging
    logger = logging.getLogger("happysimulator")
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.NOTSET)
    yield
```

### 5. Update `CLAUDE.md`

Replace the debugging section with:

```markdown
## Logging

By default, happysimulator is silent. Enable logging explicitly:

```python
import happysimulator

# Console logging
happysimulator.enable_console_logging(level="DEBUG")

# Rotating file logging (prevents disk space issues)
happysimulator.enable_file_logging("simulation.log", max_bytes=10_000_000)

# Or from environment variables
happysimulator.configure_from_env()
```

Environment variables:
- `HS_LOGGING`: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `HS_LOG_FILE`: Path to log file
- `HS_LOG_JSON`: Set to "1" for JSON output
```

## Verification

1. **Import test**: Verify importing `happysimulator` produces no log output
2. **Console test**: `enable_console_logging()` shows logs on stderr
3. **File rotation test**: Write enough logs to trigger rotation, verify backup files created
4. **JSON test**: Verify output is valid JSON parseable by `json.loads()`
5. **Env var test**: `configure_from_env()` respects `HS_LOGGING` variable
6. **Existing tests**: Run `pytest -q` to ensure no regressions
7. **Example scripts**: Run `python examples/m_m_1_queue.py` with explicit logging enabled

## Breaking Changes (Acceptable)

- Importing `happysimulator` no longer configures logging automatically
- No `happysimulator.log` file created by default
- Users must explicitly call `enable_*()` functions to see logs
