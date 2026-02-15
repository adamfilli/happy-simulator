# Testing Patterns

## Deterministic Simulations

Use constant arrival times and fixed seeds for reproducible tests:

```python
from happysimulator import Source, ConstantArrivalTimeProvider

# Constant rate for predictable timing
source = Source.constant(rate=10, target=server)

# Or use the full constructor
source = Source(
    name="Traffic",
    event_provider=provider,
    arrival_time_provider=ConstantArrivalTimeProvider(interval=0.1),
)
```

For stochastic components, set seeds:

```python
import random
random.seed(42)
# or use seed= parameter on distributions
```

## Test Structure

```
tests/
├── unit/           # fast, isolated tests
├── integration/    # full simulation tests
└── conftest.py     # shared fixtures
```

### Integration Tests

Integration tests need a `Simulation` to inject clocks into entities:

```python
from happysimulator import Simulation

def test_server_processes_request():
    sink = Sink()
    server = MyServer("server", downstream=sink)
    source = Source.constant(rate=1, target=server)

    sim = Simulation(
        entities=[source, server, sink],
        duration=10,
    )
    sim.run()

    assert sink.events_received > 0
```

## Fixtures

### `test_output_dir`

Per-test directory for artifacts (plots, logs):

```python
def test_with_output(test_output_dir):
    # test_output_dir is a Path to a unique directory
    plot_path = test_output_dir / "latency.png"
```

### Visualization Tests

```python
import pytest

def test_plot():
    plt = pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    # ... create plots ...
```

## Common Pitfalls

| Problem | Fix |
|---------|-----|
| Non-deterministic tests | Use `ConstantArrivalTimeProvider`, set `random.seed(42)` |
| `self.now` is None | Register entity in `Simulation(entities=[...])` |
| Queue grows forever | Ensure arrival rate < service rate |
| Source timing surprises | `Source.constant(rate=1)` first event at `t=1.0`, not `t=0` |

## Next Steps

- [Core Concepts](core-concepts.md) — Event, Entity, Simulation fundamentals
- [Logging](logging.md) — enable debug logging for test failures
