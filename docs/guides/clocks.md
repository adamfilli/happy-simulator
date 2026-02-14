# Clocks

## Per-Node Clocks

In distributed systems, each node may perceive time differently. `NodeClock` transforms the *read* side of the clock — events are still ordered by true simulation time.

```python
from happysimulator import NodeClock, FixedSkew, LinearDrift, Duration

# Constant offset
clock = NodeClock(FixedSkew(Duration.from_seconds(-0.05)))

# Accumulating drift (1000 parts per million)
clock = NodeClock(LinearDrift(rate_ppm=1000))
```

In an entity, use `self.now` for scheduling (true time) and `self.local_now` for decisions (perceived time).

### Clock Models

| Model | Behavior |
|-------|----------|
| `FixedSkew(offset)` | Constant offset from true time |
| `LinearDrift(rate_ppm)` | Clock runs fast/slow by `rate_ppm` parts per million |

## Logical Clocks

Pure algorithms (not entities) for establishing causal ordering. Store as entity fields.

### Lamport Clock

```python
from happysimulator import LamportClock

clock = LamportClock()
clock.tick()                # local event
ts = clock.send()           # before sending a message
clock.receive(remote_ts)    # on receiving a message
```

### Vector Clock

```python
from happysimulator import VectorClock

vc = VectorClock("node-1", ["node-1", "node-2", "node-3"])
vc.tick()
ts = vc.send()
vc.receive(remote_ts)

# Causal ordering
vc.happened_before(ts_a, ts_b)
vc.is_concurrent(ts_a, ts_b)
```

### Hybrid Logical Clock

Combines physical time with logical ordering:

```python
from happysimulator import HybridLogicalClock

hlc = HybridLogicalClock("node-1", physical_clock=node_clock)
ts = hlc.now()
ts = hlc.send()
hlc.receive(remote_ts)
```

## Next Steps

- [Networking](networking.md) — network topology and partitions
- [Distributed Systems](distributed-systems.md) — consensus protocols that use these clocks
