# Cold Start Simulation Example - Design Document

## Overview

Create `examples/cold_start.py` demonstrating cache behavior during cold start scenarios:
- Cache warmup from empty state
- Mid-simulation cache reset to trigger cold start
- Visualization of hit rate recovery, datastore load spikes, and latency impact

## Motivation

Cold starts are a critical performance concern in cached systems:
- Deployment or restart clears cache
- Cache invalidation events (TTL expiry, consistency requirements)
- Failover to cold replica

This example enables visualization and analysis of:
- How quickly cache warms up under different traffic patterns
- Impact of Zipf vs uniform customer distributions on cache efficiency
- Datastore load spikes during cold start recovery
- End-to-end latency degradation

## Requirements

### Functional Requirements

1. **Configurable traffic generation:**
   - Arrival rate (requests/second)
   - Customer ID distribution (Zipf or Uniform)
   - Number of unique customers

2. **Configurable cache:**
   - Cache capacity (max entries)
   - TTL (optional, for time-based expiration)
   - Cache read latency

3. **Configurable delays:**
   - Network delay: customer → cached server
   - Network delay: cached server → datastore (included in datastore latency)
   - Datastore read latency

4. **Cache reset capability:**
   - Trigger cache invalidation at specified time
   - Observe cold start recovery behavior

5. **Comprehensive metrics:**
   - Cache hit/miss rates over time
   - Datastore load over time
   - End-to-end latency
   - Queue depth

### Non-Functional Requirements

- Deterministic testing with constant arrivals and fixed seeds
- Visualization of all key metrics
- CLI interface for parameter configuration

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       COLD START SIMULATION ARCHITECTURE                         │
└─────────────────────────────────────────────────────────────────────────────────┘

    LOAD GENERATION
    ─────────────────────────────────────────────────────────────────────────────

    ┌─────────────────┐      Zipf/Uniform Distribution
    │   Source        │      (customer_id sampling)
    │ (Poisson/Const) │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Network Delay  │◄───── Configurable ingress latency
    │  (via yield)    │
    └────────┬────────┘
             │
             ▼
    ─────────────────────────────────────────────────────────────────────────────
    CACHED SERVER (Compositional Entity)
    ─────────────────────────────────────────────────────────────────────────────

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                           CachedServer                                   │
    │                                                                          │
    │   ┌─────────────────┐         ┌─────────────────────────────────────┐   │
    │   │  Request Queue  │         │           CachedStore               │   │
    │   │    (FIFO)       │────────►│  ┌─────────────┐  ┌──────────────┐  │   │
    │   └─────────────────┘         │  │   L1 Cache  │  │   Eviction   │  │   │
    │          ▲                    │  │  (LRU/TTL)  │  │    Policy    │  │   │
    │          │                    │  └──────┬──────┘  └──────────────┘  │   │
    │     depth probe               │         │                            │   │
    │                               │         │ miss                       │   │
    │                               │         ▼                            │   │
    │                               │  ┌─────────────────┐                 │   │
    │                               │  │    KVStore      │◄─── datastore   │   │
    │                               │  │   (datastore)   │     latency     │   │
    │                               │  └─────────────────┘                 │   │
    │                               └─────────────────────────────────────┘   │
    │                                                                          │
    │   reset_cache() ──► cache.invalidate_all()                              │
    │                                                                          │
    └──────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
    ┌─────────────────┐
    │ LatencyTracking │◄───── Records end-to-end latency
    │      Sink       │       from created_at context
    └─────────────────┘

    ─────────────────────────────────────────────────────────────────────────────
    PROBES (Metric Collection)
    ─────────────────────────────────────────────────────────────────────────────

    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
    │  Hit Rate Probe │ │ Miss Rate Probe │ │ Queue Depth     │ │ Datastore Load  │
    │  (hit_rate)     │ │ (miss_rate)     │ │ Probe (depth)   │ │ Probe (reads)   │
    └─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘
```

**Timeline:**
```
    t=0          t=warmup     t=reset      t=end
    |--------------|------------|------------|
      Cache fills    Steady      Cold start
      (warmup)       state       recovery

    Hit Rate:
    1.0 │                  ╭──────────────────╮
        │                 ╱                    \
    0.5 │               ╱                       \           ╭──────────
        │             ╱                          \        ╱
      0 │─────────╱                                ╲──╱
        └─────────────────────────────────────────────────────────────→ t
                                                   ▲
                                            Cache Reset
```

## Design

### Configuration (`ColdStartConfig`)

```python
@dataclass(frozen=True)
class ColdStartConfig:
    # Traffic parameters
    arrival_rate: float = 50.0          # requests per second
    num_customers: int = 1000           # unique customer count
    distribution_type: str = "zipf"     # "zipf" or "uniform"
    zipf_s: float = 1.0                 # Zipf exponent

    # Cache parameters
    cache_capacity: int = 100           # max cached entries
    cache_read_latency_s: float = 0.0001  # 100us cache hit
    cache_ttl_s: float | None = None    # None = LRU only

    # Network/latency parameters
    ingress_latency_s: float = 0.005    # 5ms customer → server
    datastore_read_latency_s: float = 0.010  # 10ms (includes network)

    # Timing
    cold_start_time_s: float = 90.0     # when to reset cache
    total_duration_s: float = 180.0     # total simulation time
    probe_interval_s: float = 0.5       # metric sampling interval

    # Reproducibility
    seed: int | None = 42
    use_poisson: bool = True            # False for deterministic
```

### CachedServer Entity

Extends `QueuedResource` for queue management with internal cache composition:

```python
class CachedServer(QueuedResource):
    def __init__(self, name: str, config: ColdStartConfig, downstream: Entity):
        super().__init__(name, policy=FIFOQueue())

        # Backing datastore (pre-populated)
        self.datastore = KVStore(
            name=f"{name}.datastore",
            read_latency=config.datastore_read_latency_s,
        )
        for i in range(config.num_customers):
            self.datastore.put_sync(f"customer_{i}", {"id": i, "data": f"value_{i}"})

        # Eviction policy
        if config.cache_ttl_s is not None:
            eviction = TTLEviction(
                ttl=config.cache_ttl_s,
                clock_func=lambda: self.now.to_seconds(),
            )
        else:
            eviction = LRUEviction()

        # Cache layer
        self.cache = CachedStore(
            name=f"{name}.cache",
            backing_store=self.datastore,
            cache_capacity=config.cache_capacity,
            eviction_policy=eviction,
            cache_read_latency=config.cache_read_latency_s,
        )

    def reset_cache(self) -> None:
        """Trigger cold start by invalidating all cache entries."""
        self.cache.invalidate_all()

    @property
    def hit_rate(self) -> float:
        return self.cache.hit_rate

    @property
    def datastore_reads(self) -> int:
        return self.datastore.stats.reads

    def handle_queued_event(self, event: Event) -> Generator[float, None, list[Event]]:
        # Network delay (customer → server)
        yield self.config.ingress_latency_s

        # Cache lookup (hit: fast, miss: goes to datastore)
        customer_id = event.context.get("customer_id", 0)
        value = yield from self.cache.get(f"customer_{customer_id}")

        # Forward response to sink
        return [Event(
            time=self.now,
            event_type="Response",
            target=self.downstream,
            context=event.context,
        )]
```

### Cache Reset Mechanism

Uses callback-style event scheduled mid-simulation:

```python
def create_cache_reset_event(server: CachedServer, reset_time: Instant) -> Event:
    def reset_callback(event: Event) -> list[Event]:
        server.reset_cache()
        return []

    return Event(
        time=reset_time,
        event_type="CacheReset",
        callback=reset_callback,
    )

# In simulation setup:
reset_event = create_cache_reset_event(server, Instant.from_seconds(config.cold_start_time_s))
sim.schedule(reset_event)
```

### Probes Configuration

| Metric | Target Property | Purpose |
|--------|-----------------|---------|
| `hit_rate` | `server.hit_rate` | Cache effectiveness over time |
| `miss_rate` | `server.miss_rate` | Inverse view for visualization |
| `depth` | `server.depth` | Request queue buildup |
| `datastore_reads` | `server.datastore_reads` | Backing store pressure |
| `cache_size` | `server.cache_size` | Cache fill level |

### Visualization

**Figure 1: Overview (2x2 grid)**
- Top-left: Hit/Miss rate with vertical line at cache reset
- Top-right: Cache size over time
- Bottom-left: Datastore load rate (derivative of cumulative reads)
- Bottom-right: Queue depth

**Figure 2: Latency Analysis (1x2)**
- Left: Bucketed latency over time (avg and p99)
- Right: Histogram comparing steady-state vs post-reset latency

## File Organization

```
examples/
└── cold_start.py              # Main example file

tests/integration/
└── test_cold_start.py         # Integration tests
```

## CLI Interface

```bash
python examples/cold_start.py \
  --rate 50.0 \
  --customers 1000 \
  --cache-size 100 \
  --distribution zipf \
  --zipf-s 1.0 \
  --reset-time 90.0 \
  --duration 180.0 \
  --seed 42 \
  --output output/cold_start
```

## Testing Strategy

### Integration Tests

1. **`test_cache_warms_up_over_time`**: Verify hit rate improves during warmup phase
2. **`test_cache_reset_drops_hit_rate`**: Verify hit rate drops immediately after reset
3. **`test_datastore_load_spikes_after_reset`**: Verify backing store sees increased load
4. **`test_generates_visualization_files`**: Verify PNG files are created

### Deterministic Testing

Use constant arrivals and fixed seed for reproducible tests:

```python
config = ColdStartConfig(
    use_poisson=False,  # ConstantArrivalTimeProvider
    seed=42,            # Fixed random seed
)
```

## Implementation Plan

1. Create `ColdStartConfig` dataclass with all parameters
2. Implement `CachedServer(QueuedResource)` with internal cache composition
3. Create `LatencyTrackingSink` for end-to-end latency tracking
4. Implement `run_cold_start_simulation()` main function
5. Add cache reset event scheduling mechanism
6. Create `visualize_results()` function with matplotlib
7. Add CLI entry point with argparse
8. Write integration tests

## Verification

1. **Run example:** `python examples/cold_start.py`
2. **Expected console output:**
   - Warmup phase shows hit rate improving (0% → ~80%)
   - Hit rate drops after reset
   - Final statistics printed
3. **Expected files:**
   - `output/cold_start/cold_start_overview.png`
   - `output/cold_start/cold_start_latency.png`
4. **Run tests:** `pytest tests/integration/test_cold_start.py -v`

## References

- Existing example: `examples/m_m_1_queue.py` (pattern reference)
- Cache components: `happysimulator/components/datastore/cached_store.py`
- Eviction policies: `happysimulator/components/datastore/eviction_policies.py`
- Queue patterns: `happysimulator/components/queued_resource.py`
