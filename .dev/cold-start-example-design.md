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
   - Network delay: customer → cached server (ingress)
   - Network delay: cached server → datastore (explicit round-trip)
   - Datastore read latency (processing time at datastore)

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

The datastore (KVStore) is a **separate entity** from the CachedServer, representing a remote
database like DynamoDB, Redis, or PostgreSQL. The server has a local in-memory cache and
communicates with the remote datastore over the network on cache misses.

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
    │  Network Delay  │◄───── Configurable ingress latency (customer → server)
    │  (ingress)      │
    └────────┬────────┘
             │
             ▼
    ─────────────────────────────────────────────────────────────────────────────
    CACHED SERVER (Application Server with Local Cache)
    ─────────────────────────────────────────────────────────────────────────────

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                           CachedServer                                   │
    │                                                                          │
    │   ┌─────────────────┐         ┌─────────────────────────────────────┐   │
    │   │  Request Queue  │         │      Local Cache (CachedStore)      │   │
    │   │    (FIFO)       │────────►│  ┌─────────────┐  ┌──────────────┐  │   │
    │   └─────────────────┘         │  │   L1 Cache  │  │   Eviction   │  │   │
    │          ▲                    │  │  (LRU/TTL)  │  │    Policy    │  │   │
    │          │                    │  └──────┬──────┘  └──────────────┘  │   │
    │     depth probe               │         │                            │   │
    │                               │         │ cache miss                 │   │
    │                               └─────────┼────────────────────────────┘   │
    └─────────────────────────────────────────┼────────────────────────────────┘
                                              │
                                              ▼
    ─────────────────────────────────────────────────────────────────────────────
    NETWORK (Server ↔ Datastore)
    ─────────────────────────────────────────────────────────────────────────────

    ┌─────────────────┐
    │  Network Delay  │◄───── Configurable db_network_latency (round-trip)
    │  (db network)   │       e.g., 2-5ms for same-region, 50ms+ cross-region
    └────────┬────────┘
             │
             ▼
    ─────────────────────────────────────────────────────────────────────────────
    REMOTE DATASTORE (Separate Entity - e.g., DynamoDB, Redis, PostgreSQL)
    ─────────────────────────────────────────────────────────────────────────────

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                           Datastore (KVStore)                            │
    │                                                                          │
    │   Pre-populated with customer data at simulation start                   │
    │   ┌─────────────────────────────────────────────────────────────────┐   │
    │   │   customer:0 → {id: 0, balance: 100.0}                          │   │
    │   │   customer:1 → {id: 1, balance: 101.0}                          │   │
    │   │   ...                                                            │   │
    │   │   customer:999 → {id: 999, balance: 1099.0}                     │   │
    │   └─────────────────────────────────────────────────────────────────┘   │
    │                                                                          │
    │   Latency: datastore_read_latency (processing time at DB)               │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘
             │
             ▼
    ─────────────────────────────────────────────────────────────────────────────
    RESPONSE PATH
    ─────────────────────────────────────────────────────────────────────────────

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

**Key Design Decision: Separate Datastore Entity**

The datastore is created as a separate `KVStore` entity and passed to the `CachedServer`.
This models real-world architectures where:
- The application server and database are separate services
- Network latency exists between them
- The datastore may be shared by multiple application servers
- The datastore's load can be monitored independently

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

    # Network/latency parameters
    ingress_latency_s: float = 0.005    # 5ms customer → server
    db_network_latency_s: float = 0.002 # 2ms server ↔ datastore RTT
    datastore_read_latency_s: float = 0.001  # 1ms DB processing time

    # Timing
    cold_start_time_s: float = 90.0     # when to reset cache
    duration_s: float = 180.0           # total simulation time
    probe_interval_s: float = 1.0       # metric sampling interval

    # Reproducibility
    seed: int | None = 42
    use_poisson: bool = True            # False for deterministic
```

### Datastore Entity (External)

The datastore is created separately and passed to the server:

```python
# Create remote datastore (separate entity)
datastore = KVStore(
    name="Datastore",
    read_latency=config.db_network_latency_s + config.datastore_read_latency_s,
    write_latency=config.db_network_latency_s + config.datastore_read_latency_s,
)

# Pre-populate with customer data
for i in range(config.num_customers):
    datastore.put_sync(f"customer:{i}", {"id": i, "balance": 100.0 + i})

# Create server with reference to external datastore
server = CachedServer(
    name="Server",
    datastore=datastore,  # External dependency
    cache_capacity=config.cache_capacity,
    ...
)
```

### CachedServer Entity

Extends `QueuedResource` with a local cache that wraps an external datastore:

```python
class CachedServer(QueuedResource):
    def __init__(
        self,
        name: str,
        *,
        datastore: KVStore,  # External datastore dependency
        cache_capacity: int,
        cache_read_latency_s: float,
        ingress_latency_s: float,
        downstream: Entity | None = None,
    ):
        super().__init__(name, policy=FIFOQueue())
        self._datastore = datastore  # Reference to external datastore
        self._ingress_latency_s = ingress_latency_s

        # Local cache layer wrapping the external datastore
        self._cache = CachedStore(
            name=f"{name}_cache",
            backing_store=datastore,
            cache_capacity=cache_capacity,
            eviction_policy=LRUEviction(),
            cache_read_latency=cache_read_latency_s,
        )
        ...

    def reset_cache(self) -> None:
        """Trigger cold start by invalidating local cache."""
        self._cache.invalidate_all()
        # Reset windowed stats tracking
        ...

    @property
    def datastore_reads(self) -> int:
        """Total reads from external datastore."""
        return self._datastore.stats.reads

    def handle_queued_event(self, event: Event) -> Generator[float, None, list[Event]]:
        # Network delay (customer → server)
        yield self._ingress_latency_s

        # Cache lookup (hit: fast local, miss: goes to remote datastore)
        customer_id = event.context.get("customer_id", 0)
        value = yield from self._cache.get(f"customer:{customer_id}")

        # Forward response to sink
        return [Event(...)]
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

### Simulation Setup

The datastore is created first and registered as an entity:

```python
def run_cold_start_simulation(config: ColdStartConfig) -> SimulationResult:
    # Create external datastore
    datastore = KVStore(
        name="Datastore",
        read_latency=config.db_network_latency_s + config.datastore_read_latency_s,
    )
    # Pre-populate datastore
    for i in range(config.num_customers):
        datastore.put_sync(f"customer:{i}", {"id": i, "balance": 100.0 + i})

    # Create server with reference to external datastore
    sink = LatencyTrackingSink(name="Sink")
    server = CachedServer(
        name="Server",
        datastore=datastore,
        cache_capacity=config.cache_capacity,
        cache_read_latency_s=config.cache_read_latency_s,
        ingress_latency_s=config.ingress_latency_s,
        downstream=sink,
    )

    # Run simulation with both entities
    sim = Simulation(
        ...
        entities=[datastore, server, sink],  # Datastore is a separate entity
        ...
    )
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
  --db-network-latency 0.002 \
  --datastore-latency 0.001 \
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
2. Create external `KVStore` as separate datastore entity
3. Implement `CachedServer(QueuedResource)` with local cache wrapping external datastore
4. Create `LatencyTrackingSink` for end-to-end latency tracking
5. Implement `run_cold_start_simulation()` main function
6. Add cache reset event scheduling mechanism
7. Create `visualize_results()` function with matplotlib
8. Add CLI entry point with argparse
9. Write integration tests

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
