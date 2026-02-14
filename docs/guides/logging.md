# Logging

happy-simulator is silent by default. Enable logging explicitly.

## Console Logging

```python
import happysimulator

happysimulator.enable_console_logging(level="DEBUG")
```

## File Logging

```python
happysimulator.enable_file_logging("sim.log", max_bytes=10_000_000)
happysimulator.enable_timed_file_logging("sim.log")
```

## JSON Logging

```python
happysimulator.enable_json_logging()           # to console
happysimulator.enable_json_file_logging("sim.json")  # to file
```

## Environment Variables

```python
happysimulator.configure_from_env()
```

| Variable | Description |
|----------|-------------|
| `HS_LOGGING` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `HS_LOG_FILE` | Log file path |
| `HS_LOG_JSON` | Set to `1` for JSON format |

## Module-Level Control

```python
happysimulator.set_module_level("core.simulation", "DEBUG")
happysimulator.set_level("WARNING")
```

## Disabling

```python
happysimulator.disable_logging()
```

## Next Steps

- [Testing Patterns](testing-patterns.md) — debugging with deterministic simulations
- [Visual Debugger](visual-debugger.md) — browser-based log viewer
