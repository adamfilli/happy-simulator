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

## Development Commands

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/unit/test_event.py

# Run a specific test directory
pytest tests/integration/

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

Examples are organized into subdirectories by domain:

```bash
python -m examples.queuing.m_m_1_queue
python -m examples.distributed.chain_replication
python -m examples.industrial.bank_branch
```

### Documentation

```bash
# Local preview at localhost:8000
python -m mkdocs serve

# Build docs to site/
python -m mkdocs build
```

### Visual Debugger

Requires optional dependencies:

```bash
pip install -e ".[visual]"
python -m examples.visual.visual_debugger
```

### Debugging

Set the logging level via environment variable:

```bash
# Bash
HS_LOGGING=DEBUG python -m examples.queuing.m_m_1_queue
```

```powershell
# PowerShell
$env:HS_LOGGING='DEBUG'
python -m examples.queuing.m_m_1_queue
```

## Project Structure

```
happy-simulator/
├── happysimulator/              # Main package
│   ├── __init__.py              # Package exports and version
│   ├── core/                    # Simulation engine (event, entity, instant, sim_future)
│   │   └── control/             # Pause/step/breakpoints
│   ├── components/              # Reusable simulation building blocks
│   │   ├── behavior/            # Agent-based behavioral modeling
│   │   ├── client/              # Client entities
│   │   ├── consensus/           # Consensus protocols (Raft, Paxos)
│   │   ├── crdt/                # Conflict-free replicated data types
│   │   ├── datastore/           # Storage engine components
│   │   ├── deployment/          # Deployment strategies (canary, rolling)
│   │   ├── industrial/          # Manufacturing/service industry components
│   │   ├── infrastructure/      # CPU, disk, scheduling primitives
│   │   ├── load_balancer/       # Load balancing strategies
│   │   ├── messaging/           # Message queues and pub/sub
│   │   ├── microservice/        # Microservice patterns
│   │   ├── network/             # Network topology and partitions
│   │   ├── queue_policies/      # FIFO, priority, etc.
│   │   ├── rate_limiter/        # Rate limiting (token bucket, etc.)
│   │   ├── replication/         # Replication protocols
│   │   ├── resilience/          # Circuit breakers, retries, bulkheads
│   │   ├── scheduling/          # Job/task scheduling
│   │   ├── server/              # Server entities
│   │   ├── sketching/           # Probabilistic data structures
│   │   ├── storage/             # B-tree, LSM tree, WAL
│   │   ├── streaming/           # Stream processing
│   │   └── sync/                # Synchronization primitives
│   ├── analysis/                # Post-run analysis utilities
│   ├── distributions/           # Statistical distributions
│   ├── faults/                  # Fault injection
│   ├── instrumentation/         # Probes, trackers, data collection
│   ├── load/                    # Load generation (sources, profiles)
│   ├── numerics/                # Numerical utilities
│   ├── sketching/               # Core sketching algorithms
│   ├── visual/                  # Browser-based debugger (FastAPI + React)
│   └── utils/                   # Shared utilities
├── visual-frontend/             # React + TypeScript source for visual debugger
├── examples/                    # Example simulations (organized by domain)
│   ├── behavior/                # Agent-based modeling examples
│   ├── deployment/              # Deployment simulation examples
│   ├── distributed/             # Distributed systems examples
│   ├── industrial/              # Manufacturing/service examples
│   ├── infrastructure/          # Infrastructure simulation examples
│   ├── load-balancing/          # Load balancing examples
│   ├── performance/             # Performance analysis examples
│   ├── queuing/                 # Queuing theory examples
│   ├── storage/                 # Storage engine examples
│   └── visual/                  # Visual debugger example
├── tests/                       # Test suite
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   ├── regression/              # Regression tests
│   └── perf/                    # Performance tests
├── docs/                        # MkDocs Material documentation site
│   ├── guides/                  # User guides
│   ├── reference/               # Auto-generated API reference
│   ├── examples/                # Example gallery
│   └── design/                  # Design philosophy
├── .github/workflows/           # CI/CD
│   ├── tests.yml                # Daily test runs
│   ├── docs.yml                 # Docs deployment to GitHub Pages
│   └── publish-pypi.yml         # PyPI publish on GitHub release
├── pyproject.toml               # Package configuration
├── CLAUDE.md                    # AI assistant instructions
└── DEV.md                       # This file
```

## CI/CD

### Tests

Tests run daily via GitHub Actions and can be triggered manually. See `.github/workflows/tests.yml`.

### Documentation

Docs are auto-deployed to GitHub Pages on push to `main`. See `.github/workflows/docs.yml`.

### PyPI Publishing

Publishing is automated via GitHub Actions using trusted publishing (no API tokens needed):

1. Create a GitHub release (or use the `/release` workflow)
2. The `publish-pypi.yml` workflow builds and publishes to PyPI automatically

Manual publishing with twine is no longer needed for normal releases.

## Version Management

The package version is defined in two places (keep them in sync):
- `pyproject.toml`: `version = "..."` (used by PyPI)
- `happysimulator/__init__.py`: `__version__ = "..."` (used at runtime)

## Building Distribution Packages

```bash
# Install build tool
pip install build

# Build wheel and sdist
python -m build

# Output will be in dist/
```
