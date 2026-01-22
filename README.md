# happy-simulator

[![PyPI](https://img.shields.io/pypi/v/happysim)](https://pypi.org/project/happysim/)
[![Tests](https://github.com/adamfilli/happy-simulator/actions/workflows/tests.yml/badge.svg)](https://github.com/adamfilli/happy-simulator/actions/workflows/tests.yml)
[![Docs](https://github.com/adamfilli/happy-simulator/actions/workflows/docs.yml/badge.svg)](https://adamfilli.github.io/happy-simulator/)

A discrete-event simulation library for Python 3.13+, inspired by MATLAB SimEvents. Model systems using an event-driven architecture where a central `EventHeap` schedules and executes `Event` objects until the simulation ends.

## Installation

```bash
# Clone the repository
git clone https://github.com/adamfilli/happy-simulator.git
cd happy-simulator

# Create and activate a virtual environment (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```