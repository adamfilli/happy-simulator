# Developer Guide

This document covers setup and development workflows for contributing to happy-simulator.

## Prerequisites

- Python 3.13 or higher
- Git

## Installation

### Using pip (standard)

```bash
# Clone the repository
git clone https://github.com/adamfilli/happy-simulator.git
cd happy-simulator

# Create and activate virtual environment
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows Command Prompt
.\.venv\Scripts\activate.bat

# Linux/macOS
source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Using uv (faster alternative)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer written in Rust.

```bash
# Install uv (if not already installed)
# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/adamfilli/happy-simulator.git
cd happy-simulator

# Create venv and install (uv handles both)
uv venv --python 3.13
uv pip install -e ".[dev]"
```

### Using pipx (isolated global install)

For installing as a standalone tool without affecting your system Python:

```bash
pipx install git+https://github.com/adamfilli/happy-simulator.git
```

## Development Commands

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/integration/test_queue.py

# Run with coverage
pytest --cov=happysimulator --cov-report=html
```

### Code Quality

```bash
# Type checking
mypy happysimulator

# Linting and formatting check
ruff check .

# Auto-fix linting issues
ruff check --fix .

# Format code
ruff format .
```

### Running Examples

```bash
python examples/basic_client_server.py
```

### Debugging

Set the logging level via environment variable:

```powershell
# PowerShell
$env:HS_LOGGING='DEBUG'
python examples/basic_client_server.py
```

```bash
# Bash
HS_LOGGING=DEBUG python examples/basic_client_server.py
```

Logs are written to `happysimulator.log` in the repository root.

## Project Structure

```
happy-simulator/
├── happysimulator/          # Main package
│   ├── __init__.py          # Package exports and version
│   ├── core/                # Core simulation engine
│   ├── components/          # Reusable components (queues, resources)
│   ├── distributions/       # Statistical distributions
│   ├── instrumentation/     # Probes and data collection
│   ├── load/                # Load generation (sources, providers)
│   ├── modules/             # Higher-level modules
│   └── utils/               # Utilities (Instant, etc.)
├── examples/                # Example simulations
├── tests/                   # Test suite
├── pyproject.toml           # Package configuration
└── DEV.md                   # This file
```

## Building Distribution Packages

```bash
# Install build tool
pip install build

# Build wheel and sdist
python -m build

# Output will be in dist/
#   dist/happy_simulator-0.1.0-py3-none-any.whl
#   dist/happy_simulator-0.1.0.tar.gz
```

## Version Management

The package version is defined in two places (keep them in sync):
- `pyproject.toml`: `version = "0.1.0"`
- `happysimulator/__init__.py`: `__version__ = "0.1.0"`

## Publishing to PyPI

### Prerequisites

```powershell
# Install/upgrade build and twine
pip install --upgrade build twine
```

### Step 1: Clean and Build

```powershell
# Remove old builds
Remove-Item -Recurse -Force dist -ErrorAction SilentlyContinue

# Build wheel and sdist
python -m build
```

### Step 2: Upload to TestPyPI (Optional but Recommended)

Test the upload before publishing to production:

```powershell
python -m twine upload --repository testpypi dist/*
```

When prompted:
- Username: `__token__`
- Password: Your TestPyPI API token (starts with `pypi-`)

Verify at: https://test.pypi.org/project/happysim/

Test installation:
```powershell
pip install --index-url https://test.pypi.org/simple/ happysim
```

### Step 3: Upload to PyPI

```powershell
python -m twine upload dist/*
```

When prompted:
- Username: `__token__`
- Password: Your PyPI API token (starts with `pypi-`)

Verify at: https://pypi.org/project/happysim/

### API Token Setup

1. Create accounts at https://pypi.org and https://test.pypi.org
2. Go to Account Settings → API tokens
3. Create a token scoped to this project (or account-wide for first upload)
4. Store tokens securely (e.g., in a `.pypirc` file or password manager)

Optional `.pypirc` for automated uploads (place in `$HOME`):
```ini
[pypi]
username = __token__
password = pypi-YOUR_TOKEN_HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR_TEST_TOKEN_HERE
```
