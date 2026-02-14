# Queuing & Resources

## Queue and QueuedResource

`QueuedResource` combines a queue with processing logic. Events are buffered when the entity is busy and dequeued when capacity is available.

```python
from happysimulator import QueuedResource, FIFOQueue, Event

class MyServer(QueuedResource):
    def __init__(self, name, downstream, concurrency=1):
        super().__init__(name, policy=FIFOQueue())
        self.downstream = downstream
        self.concurrency = concurrency
        self._in_flight = 0

    def has_capacity(self) -> bool:
        return self._in_flight < self.concurrency

    def handle_queued_event(self, event):
        self._in_flight += 1
        try:
            yield 0.1  # processing time
        finally:
            self._in_flight -= 1
        return [Event(time=self.now, event_type="Done", target=self.downstream)]
```

### Queue Policies

| Policy | Behavior |
|--------|----------|
| `FIFOQueue()` | First in, first out |
| `LIFOQueue()` | Last in, first out (stack) |
| `PriorityQueue(key=...)` | Ordered by priority key |

## Resource (Shared Capacity)

`Resource` models contended capacity (CPU cores, connections, permits) with acquire/release semantics.

```python
from happysimulator import Resource

cpu = Resource("cpu_cores", capacity=8)

# In an entity's handle_event():
grant = yield cpu.acquire(amount=2)   # blocks via SimFuture if unavailable
yield 0.1                              # do work
grant.release()                        # idempotent, wakes FIFO waiters

# Non-blocking alternative
grant = cpu.try_acquire(amount=2)      # Returns Grant | None
```

**Properties:** `cpu.available`, `cpu.utilization`, `cpu.waiters`, `cpu.stats`

!!! note
    Register `Resource` in `Simulation(entities=[cpu, ...])` so it receives clock injection.

## Rate Limiter

Rate limiting uses a **Policy** (pure algorithm) wrapped in a **RateLimitedEntity** (Entity with queue).

### Policies

| Policy | Algorithm |
|--------|-----------|
| `TokenBucketPolicy(rate, capacity)` | Refill tokens at fixed rate |
| `LeakyBucketPolicy(rate)` | Process at constant rate |
| `SlidingWindowPolicy(rate, window_s)` | Count in sliding window |
| `FixedWindowPolicy(rate, window_s)` | Count in fixed windows |
| `AdaptivePolicy(initial_rate, ...)` | Adjusts based on feedback |

### Usage

```python
from happysimulator import RateLimitedEntity, TokenBucketPolicy

limiter = RateLimitedEntity(
    name="RateLimiter",
    policy=TokenBucketPolicy(rate=100, capacity=10),
    downstream=server,
)
```

## Inductor (Burst Suppression)

EWMA-based smoothing that resists rate *changes* without capping throughput.

```python
from happysimulator import Inductor

inductor = Inductor(
    name="Smoother",
    downstream=server,
    time_constant=1.0,       # τ in seconds — higher = more smoothing
    queue_capacity=10000,
)
```

**Properties:** `inductor.stats`, `inductor.estimated_rate`, `inductor.queue_depth`

## Next Steps

- [Observability](observability.md) — data collection and analysis
- [Simulation Control](simulation-control.md) — pause, step, and breakpoints
