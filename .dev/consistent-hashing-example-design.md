# Consistent Hashing Load Balancing Simulation

## Overview

Create a comprehensive simulation demonstrating consistent hashing benefits in a distributed caching scenario. Output location: `./examples/load-balancing/`

## Architecture

```
Source ──> LoadBalancer ──┬──> Server 0 (Cache 0) ──┐
  (customer_ids)          ├──> Server 1 (Cache 1) ──┼──> Shared Datastore
                          ├──> Server 2 (Cache 2) ──┤
                          └──> Server N (Cache N) ──┘
```

Each server has a local TTL-based cache. With consistent hashing, the same customer always routes to the same server, maximizing cache hits.

---

## File Structure

```
examples/load-balancing/
├── __init__.py
├── common.py                        # Shared components
├── consistent_hashing_basics.py     # Scenario A: ConsistentHash vs RoundRobin
├── fleet_change_comparison.py       # Scenario B: ConsistentHash vs ModuloHash during fleet changes
├── vnodes_analysis.py               # Scenario C: Virtual node distribution analysis
├── zipf_effect.py                   # Scenario D: Zipf distribution impact
└── README.md                        # Overview documentation

tests/integration/
└── test_load_balancing_examples.py  # Integration tests
```

---

## Existing Components to Reuse

| Component | File | Usage |
|-----------|------|-------|
| `LoadBalancer` | `components/load_balancer/load_balancer.py` | Core routing with `add_backend()`/`remove_backend()` |
| `ConsistentHash` | `components/load_balancer/strategies.py:336` | Consistent hashing with virtual nodes |
| `RoundRobin` | `components/load_balancer/strategies.py:49` | Baseline comparison |
| `IPHash` | `components/load_balancer/strategies.py:294` | Modulo-based hashing (hash % N) |
| `CachedStore` | `components/datastore/cached_store.py` | Cache with eviction policies |
| `KVStore` | `components/datastore/kv_store.py` | Shared backing datastore |
| `TTLEviction` | `components/datastore/eviction_policies.py:160` | TTL-based cache expiration |
| `ZipfDistribution` | `distributions/zipf.py` | Skewed customer ID distribution |
| `UniformDistribution` | `distributions/uniform.py` | Constant/uniform distribution |
| `DistributedFieldProvider` | `load/providers/distributed_field.py` | Sample customer_id per request |
| `Probe`, `Data` | `instrumentation/` | Metrics collection |

---

## New Components (common.py)

### 1. CachingServer Entity

```python
class CachingServer(QueuedResource):
    """Server with local TTL-based cache backed by shared datastore.

    Probed metrics: hit_rate, miss_rate, cache_size, requests_processed
    """
    def __init__(
        self,
        name: str,
        server_id: int,
        datastore: KVStore,
        cache_capacity: int,
        cache_ttl_s: float,
        cache_read_latency_s: float = 0.0001,
        processing_latency_s: float = 0.001,
    ): ...
```

Key design:
- Wraps `CachedStore` with `TTLEviction` policy
- Injects simulation clock into TTLEviction via `clock_func`
- Exposes `hit_rate`, `miss_rate`, `cache_size` properties for probing
- Extracts `customer_id` from event context to use as cache key

### 2. Helper Functions

- `collect_aggregate_metrics(servers)` - Aggregate stats across all servers
- `compute_key_distribution(strategy, backends, keys)` - Analyze key placement
- `plot_hit_rate_comparison(...)` - Compare strategies over time
- `plot_key_distribution(...)` - Bar chart of keys per server
- `plot_fleet_change_impact(...)` - Visualize cache invalidation

---

## Scenarios

### Scenario A: Consistent Hashing vs Round Robin (`consistent_hashing_basics.py`)

**Goal:** Show consistent hashing achieves ~80-90% hit rate vs ~10-20% for round robin.

**Config:**
```python
@dataclass(frozen=True)
class BasicConfig:
    arrival_rate: float = 500.0
    num_customers: int = 1000
    duration_s: float = 60.0
    num_servers: int = 5
    cache_capacity: int = 100
    cache_ttl_s: float = 30.0
```

**Metrics:**
- Per-server and aggregate cache hit rate over time
- Datastore read rate (lower = better)
- Key distribution uniformity

**Visualization:** 2-panel plot comparing hit rates and datastore load for both strategies.

---

### Scenario B: Fleet Changes (`fleet_change_comparison.py`)

**Goal:** Demonstrate that modulo hashing causes catastrophic cache invalidation when servers change, while consistent hashing only shifts ~1/N keys.

**Timeline:**
```
t=0          t=30s           t=60s
|------------|---------------|
  5 servers   Add 6th server
              (observe impact)
```

**Comparison:**
- **ModuloHash (IPHash):** `hash(key) % N` - when N changes, ALL keys potentially shift
- **ConsistentHash:** Virtual node ring - only ~1/N keys shift

**Metrics:**
- Hit rate drop immediately after fleet change
- Datastore read spike magnitude
- Recovery time to steady-state
- Percentage of keys that changed servers

**Visualization:** Time series showing hit rate drop and recovery for both strategies.

---

### Scenario C: V-Nodes Analysis (`vnodes_analysis.py`)

**Goal:** Show empirically that more virtual nodes = more uniform key distribution.

**Approach:** Analytical (no full simulation needed)
1. Create ConsistentHash rings with varying vnode counts: 1, 5, 10, 50, 100, 200, 500
2. Assign 10,000 keys to 5 backends
3. Measure distribution variance

**Metrics:**
- Coefficient of variation (std/mean) per vnode count
- Max/min ratio of key counts

**Expected Results:**
| V-nodes | CoV | Max/Min |
|---------|-----|---------|
| 1 | ~0.8 | ~3-4x |
| 100 | ~0.1 | ~1.2x |

**Visualization:** Line plot of CoV vs vnode count, bar chart of distribution.

---

### Scenario D: Zipf Distribution Effect (`zipf_effect.py`)

**Goal:** Show that Zipf-distributed access patterns cause uneven server load even with perfect consistent hashing.

**Config:**
```python
@dataclass(frozen=True)
class ZipfConfig:
    num_customers: int = 1000
    distribution_type: str = "zipf"  # or "uniform"
    zipf_s: float = 1.5
```

**Key Insight:** With Zipf, 20% of keys get 80% of traffic. The servers assigned to hot keys are overwhelmed, even though key *assignment* is uniform.

**Metrics:**
- Per-server request counts (load imbalance)
- Per-server cache hit rates (hot servers hit more)
- Datastore reads per server

**Visualization:** Side-by-side bar charts comparing uniform vs Zipf load distribution.

---

## Implementation Phases

### Phase 1: Foundation
1. Create `examples/load-balancing/` directory
2. Implement `common.py` with `CachingServer` entity
3. Implement `consistent_hashing_basics.py` (Scenario A)
4. Write basic tests

**Deliverable:** Working comparison showing consistent hash >> round robin

### Phase 2: Fleet Changes
1. Implement `fleet_change_comparison.py` (Scenario B)
2. Use callback event to add/remove server at t=30s
3. Write tests verifying ~1/N shift for consistent hashing

**Deliverable:** Visualization of cache invalidation impact

### Phase 3: V-Nodes Analysis
1. Implement `vnodes_analysis.py` (Scenario C)
2. Purely analytical - iterate ConsistentHash with different vnode counts

**Deliverable:** Empirical proof of vnode → uniformity relationship

### Phase 4: Zipf Effect
1. Implement `zipf_effect.py` (Scenario D)
2. Compare uniform vs Zipf with consistent hashing

**Deliverable:** Demonstration of real-world load skew

### Phase 5: Polish
1. Add `README.md` with scenario overview
2. Add argparse CLI to all scripts
3. Ensure consistent visualization style
4. Final test coverage

---

## Verification Plan

### Manual Testing
```bash
# Run each scenario
python examples/load-balancing/consistent_hashing_basics.py
python examples/load-balancing/fleet_change_comparison.py
python examples/load-balancing/vnodes_analysis.py
python examples/load-balancing/zipf_effect.py

# Check output in examples/load-balancing/output/
```

### Automated Tests
```bash
pytest tests/integration/test_load_balancing_examples.py -v
```

### Key Assertions
- Scenario A: `consistent_hash_hit_rate > round_robin_hit_rate * 3`
- Scenario B: `consistent_hash_keys_shifted < modulo_keys_shifted * 0.3`
- Scenario C: `cov(vnodes=100) < cov(vnodes=1) * 0.2`
- Scenario D: `max_server_load / min_server_load > 2` with Zipf

---

## Critical Implementation Notes

1. **TTLEviction clock injection:** Must pass simulation clock to TTLEviction:
   ```python
   TTLEviction(ttl=30.0, clock_func=lambda: self.now.to_seconds())
   ```

2. **Customer ID extraction:** ConsistentHash looks for key in `context["metadata"]["customer_id"]`. Ensure DistributedFieldProvider places it there.

3. **Fleet change event:** Schedule callback at t=30s to call `lb.add_backend(new_server)`:
   ```python
   Event(time=Instant.from_seconds(30), event_type="AddServer", callback=add_server_callback)
   ```

4. **IPHash as ModuloHash:** The existing `IPHash` strategy uses `hash % len(backends)` which is exactly modulo hashing. No new strategy needed.
