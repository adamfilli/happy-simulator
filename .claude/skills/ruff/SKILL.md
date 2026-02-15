---
name: ruff
description: Run ruff linter and formatter on the project
disable-model-invocation: true
---

# Ruff

Run the ruff linter and/or formatter on the project. Configuration is in `pyproject.toml` under `[tool.ruff]`.

## Instructions

1. Determine what the user wants:
   - **No arguments or "check"**: lint check only
   - **"fix"**: lint with auto-fix
   - **"format"**: format code
   - **"all"**: lint fix + format (the full cleanup)
   - **A specific path**: scope to that path

2. Run the appropriate commands using `.venv/Scripts/python.exe -m ruff`:
   - Lint check: `.venv/Scripts/python.exe -m ruff check .`
   - Lint fix: `.venv/Scripts/python.exe -m ruff check --fix .`
   - Format check: `.venv/Scripts/python.exe -m ruff format --check .`
   - Format apply: `.venv/Scripts/python.exe -m ruff format .`
   - Full cleanup: `.venv/Scripts/python.exe -m ruff check --fix . && .venv/Scripts/python.exe -m ruff format .`

3. If a specific path is provided, replace `.` with that path in all commands.

4. Report the results:
   - Number of violations found
   - Number of auto-fixed violations (if --fix was used)
   - Number of files reformatted (if format was used)
   - List any remaining violations that require manual attention

5. Do NOT auto-fix violations unless the user asks for "fix", "all", or explicitly requests fixes.
