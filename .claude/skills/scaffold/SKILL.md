---
name: scaffold
description: Generate a complete simulation from a high-level description
---

# Scaffold Simulation

Generate a complete, runnable happysimulator simulation from a user's description.

## Instructions

1. If the user hasn't described what they want to simulate, ask them. Get enough detail to choose the right components (e.g., "a hospital ER with triage" vs "a generic queue").

2. Read the project's CLAUDE.md for the full API reference, conventions, and component catalog. Use it as your source of truth for imports, patterns, and available components.

3. Generate a single `.py` file with this structure:

```python
"""<Title>: <one-line description of what this simulates>."""

from dataclasses import dataclass
# ... imports from happysimulator ...

@dataclass(frozen=True)
class Config:
    """Simulation parameters."""
    # All tunable knobs here with sensible defaults

# Entity classes with handle_event / handle_queued_event generators

def run(config: Config | None = None) -> None:
    config = config or Config()
    # Build pipeline: sink ← server(s) ← source
    # Create Simulation with sources, entities, end_time
    # Run and print summary

if __name__ == "__main__":
    run()
```

4. Follow these conventions strictly:
   - **Time**: Always use `Instant.from_seconds(n)`, never raw floats for Event times
   - **Events**: Every Event must have a `target`. Use `Event.once()` for function-based dispatch
   - **Generators**: `yield <float>` for delays, `yield <float>, [events]` for delay + side-effects, `yield <future>` to park, `return [events]` on completion
   - **Entity registration**: Every entity must appear in `Simulation(entities=[...])`
   - **QueuedResource**: Override `has_capacity()` if you want the queue to actually build up
   - **Results**: Include a `Sink` or `Counter` and print `sink.latency_stats()` or `counter.total` at the end
   - **Determinism**: Use `seed=42` on distributions and `random.seed(42)` for reproducibility

5. Prefer built-in components over custom entities when possible:
   - `Source.poisson(rate=N, target=server)` for stochastic arrivals
   - `Source.constant(rate=N, target=server)` for deterministic arrivals
   - `QueuedResource` for anything with a queue + processing
   - `Sink` / `Counter` for collecting results
   - Industrial components (`ConveyorBelt`, `BatchProcessor`, `ShiftSchedule`, etc.) for operations research
   - `Network` + link conditions for distributed systems
   - `Agent` + `Population` for behavioral modeling

6. Run the generated file with `python <file>` to verify it works. Fix any errors.

7. Briefly explain the simulation structure to the user: what entities exist, how events flow, and what metrics are printed.

## Component Quick Reference

| Domain | Key Components |
|--------|---------------|
| Queuing | `QueuedResource`, `FIFOQueue`, `PriorityQueue`, `Sink`, `Counter` |
| Networking | `Network`, `datacenter_network()`, `internet_network()`, `partition()` |
| Rate limiting | `RateLimitedEntity`, `TokenBucketPolicy`, `Inductor` |
| Resilience | `CircuitBreaker`, `Bulkhead`, `TimeoutWrapper`, `Fallback`, `Hedge` |
| Industrial | `ConveyorBelt`, `InspectionStation`, `BatchProcessor`, `ShiftSchedule`, `BreakdownScheduler`, `InventoryBuffer`, `BalkingQueue`, `RenegingQueuedResource`, `GateController`, `SplitMerge`, `PreemptibleResource` |
| Storage | `KVStore`, `LSMTree`, `WriteAheadLog`, `TransactionManager` |
| Distributed | `RaftNode`, `PaxosNode`, `CRDTStore`, `DistributedLock` |
| Behavioral | `Agent`, `Population`, `Environment`, `SocialGraph` |
| Resources | `Resource` (contended capacity with `acquire`/`release`) |
| Load balancing | `ConsistentHashRing`, `RoundRobinBalancer`, `LoadBalancer` |
