# happy-simulator

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