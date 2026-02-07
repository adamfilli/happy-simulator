---
name: run-tests
description: Run pytest tests for the project
disable-model-invocation: true
---

# Run Tests

Run the project's test suite using pytest.

## Instructions

1. If the user specifies a test file or directory, run that: `.venv/Scripts/python.exe -m pytest <path> -q`
2. If the user specifies a keyword or marker, pass it through (e.g., `-k "test_name"`, `-m "slow"`)
3. If no arguments are given, run the full test suite: `.venv/Scripts/python.exe -m pytest -q`
4. Report the results: number of passed, failed, skipped, and errors
5. If there are failures, summarize each failing test with the relevant assertion or error message
6. Do NOT attempt to fix failing tests unless the user explicitly asks
