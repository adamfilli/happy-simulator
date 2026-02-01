---
name: gen-test
description: Generate a pytest test following project conventions
disable-model-invocation: true
---

# Generate Test

Create pytest tests following happy-simulator project conventions.

## Project Test Patterns

### Timing
- Use `ConstantArrivalTimeProvider` for deterministic timing in tests
- Use `Instant.from_seconds()` for time values
- Use `Instant.Epoch` for time zero, `Instant.Infinity` for auto-termination

### File Organization
- Unit tests go in `tests/unit/`
- Integration tests go in `tests/integration/`
- File naming: `test_<feature>.py`
- Function naming: `test_<scenario>()`

### Common Imports

```python
import pytest
from happysimulator import Simulation, Event, Instant
from happysimulator.load import Source, ConstantArrivalTimeProvider
```

### Example Test Structure

```python
def test_example_scenario():
    """Test description explaining what behavior is verified."""
    # Arrange
    sim = Simulation(end_time=Instant.from_seconds(10))

    # Act
    sim.run()

    # Assert
    assert expected_condition
```

## Instructions

1. Ask what functionality needs testing if not specified
2. Determine if this is a unit test or integration test
3. Generate test following the patterns above
4. Run the test with `pytest tests/<path> -q` to verify it passes
