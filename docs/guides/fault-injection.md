# Fault Injection

Systematically test system resilience by scheduling faults.

## FaultSchedule

`FaultSchedule` injects node, network, and resource faults at specified times:

```python
from happysimulator.faults import FaultSchedule

faults = FaultSchedule(name="chaos")
```

Use fault schedules to automate failure scenarios — node crashes, network partitions, resource exhaustion — and verify that your system handles them correctly.

## Combining with Network Partitions

```python
from happysimulator import Network

network = Network(name="cluster")
# ... set up topology ...

# Manual partition
partition = network.partition([node_a], [node_c])
# ... observe system behavior ...
partition.heal()
```

## Combining with Breakdowns

For industrial simulations, use `BreakdownScheduler` for random UP/DOWN cycles:

```python
from happysimulator.components.industrial import BreakdownScheduler

breakdowns = BreakdownScheduler(
    name="machine_failures",
    target=machine,
    mean_time_to_failure=100.0,   # seconds
    mean_repair_time=10.0,
)
```

## Next Steps

- [Networking](networking.md) — network partitions
- [Distributed Systems](distributed-systems.md) — consensus under failures
- [Testing Patterns](testing-patterns.md) — deterministic fault testing
