# Installation

## From PyPI

```bash
pip install happysim
```

## From Source

```bash
git clone https://github.com/adamfilli/happy-simulator.git
cd happy-simulator
pip install -e .
```

## Optional Extras

| Extra | Install Command | What It Adds |
|-------|----------------|--------------|
| `visual` | `pip install happysim[visual]` | Browser-based visual debugger (FastAPI, uvicorn) |
| `dev` | `pip install -e ".[dev]"` | Testing, linting, docs (pytest, mypy, ruff, mkdocs) |

## Verify Installation

```python
import happysimulator
print(happysimulator.__version__)
```

## Requirements

- Python 3.13 or later
- Core dependencies: `matplotlib`, `numpy`, `pandas`
