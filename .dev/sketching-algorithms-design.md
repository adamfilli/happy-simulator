# Sketching Algorithm Component Library Design

> **Status:** Approved
> **Created:** 2026-02-01

## Overview

Add a streaming/sketching algorithm library to happy-simulator that provides approximate statistics over event streams. The library separates pure mathematical algorithms from simulation entity wrappers, following established codebase patterns.

## Motivation

Streaming algorithms (sketches) compute approximate statistics over data streams using bounded memory. They're essential for simulating real-world systems that need to:
- Track top-K heavy hitters (e.g., most frequent API endpoints)
- Estimate latency percentiles (p50, p99, p999)
- Count distinct elements (unique users)
- Test set membership (bloom filters)

## File Organization

```
happysimulator/
├── sketching/                           # Pure math algorithms (no simulation deps)
│   ├── __init__.py                      # Public exports
│   ├── base.py                          # Base protocols: Sketch, FrequencySketch, etc.
│   ├── topk.py                          # Top-K (Space-Saving algorithm)
│   ├── count_min_sketch.py              # Count-Min Sketch for frequency estimation
│   ├── tdigest.py                       # T-Digest for quantile estimation
│   ├── hyperloglog.py                   # HyperLogLog for cardinality
│   ├── bloom_filter.py                  # Bloom Filter for membership
│   └── reservoir.py                     # Reservoir Sampling
│
├── components/
│   └── sketching/                       # Entity wrappers (simulation integration)
│       ├── __init__.py
│       ├── sketch_collector.py          # Base wrapper for any sketch
│       ├── topk_collector.py            # Top-K specific collector
│       └── quantile_estimator.py        # T-Digest wrapper for latency tracking

tests/
├── unit/sketching/                      # Unit tests for pure algorithms
│   ├── test_topk.py
│   ├── test_count_min_sketch.py
│   ├── test_tdigest.py
│   ├── test_hyperloglog.py
│   ├── test_bloom_filter.py
│   └── test_reservoir.py
│
└── integration/
    └── test_sketch_vs_ideal.py          # Compare sketch accuracy to exact statistics
```

## Architecture

### Layer 1: Pure Algorithms (happysimulator/sketching/)

No simulation dependencies. Each algorithm provides:
- `add(item, count=1)` - Process stream element
- `query methods` - Algorithm-specific queries
- `merge(other)` - Combine two sketches
- `memory_bytes` - Memory estimate
- `clear()` - Reset state
- Seeded RNG for reproducibility

**Base Protocols:**
```python
class Sketch(ABC):
    def add(self, item, count=1): ...
    def merge(self, other): ...
    @property
    def memory_bytes(self) -> int: ...
    @property
    def item_count(self) -> int: ...
    def clear(self): ...

class FrequencySketch(Sketch):      # TopK, Count-Min Sketch
    def estimate(self, item) -> int: ...

class QuantileSketch(Sketch):       # T-Digest
    def quantile(self, q) -> float: ...
    def cdf(self, value) -> float: ...

class CardinalitySketch(Sketch):    # HyperLogLog
    def cardinality(self) -> int: ...

class MembershipSketch(Sketch):     # Bloom Filter
    def contains(self, item) -> bool: ...
```

### Layer 2: Entity Wrappers (happysimulator/components/sketching/)

Integrate algorithms with simulation via `Entity.handle_event()`:

```python
class SketchCollector(Entity, Generic[S]):
    def __init__(self, name, sketch, value_extractor, weight_extractor=None):
        self._sketch = sketch
        self._value_extractor = value_extractor  # Event -> value

    def handle_event(self, event) -> list[Event]:
        value = self._value_extractor(event)
        self._sketch.add(value)
        return []
```

## Algorithms

| Algorithm | Use Case | Key Parameters | Error Guarantee |
|-----------|----------|----------------|-----------------|
| **TopK** | Heavy hitters | k (items to track) | Error ≤ N/k per item |
| **Count-Min Sketch** | Frequency estimation | width, depth | ≤ εN with prob ≥ 1-δ |
| **T-Digest** | Quantile estimation | compression | Higher accuracy at tails |
| **HyperLogLog** | Cardinality | precision (4-16) | ~1.04/√m standard error |
| **Bloom Filter** | Set membership | size_bits, num_hashes | Configurable FP rate |
| **Reservoir Sampler** | Uniform sampling | size | Exact uniform sample |

### TopK (Space-Saving Algorithm)

Maintains k counters tracking the most frequent items. When a new item arrives and all counters are full, it replaces the minimum counter.

**Properties:**
- No false negatives: All items with count > N/K are tracked
- Bounded error: Overestimation ≤ N/K per item
- Memory: O(K) counters

```python
topk = TopK[str](k=100, seed=42)
for request in stream:
    topk.add(request.endpoint)

for item in topk.top(10):
    print(f"{item.item}: {item.count} (±{item.error})")
```

### T-Digest

Cluster-based compression for quantile estimation. Maintains smaller clusters near the tails for better accuracy at extreme percentiles.

```python
td = TDigest(compression=100)
for latency in stream:
    td.add(latency)

print(f"p50: {td.quantile(0.5)}")
print(f"p99: {td.quantile(0.99)}")
print(f"p999: {td.quantile(0.999)}")
```

### Count-Min Sketch

Uses d hash functions and w counters per function. Provides frequency estimates with one-sided error (always overestimates).

```python
# ~0.1% error with 99.9% confidence
cms = CountMinSketch.from_error_rate(epsilon=0.001, delta=0.001)
for item in stream:
    cms.add(item)

print(f"Estimated count: {cms.estimate('foo')}")
```

### HyperLogLog

Estimates distinct element count using O(log log n) memory.

```python
hll = HyperLogLog(precision=14)  # ~0.8% error
for user_id in stream:
    hll.add(user_id)

print(f"Unique users: {hll.cardinality()}")
```

### Bloom Filter

Space-efficient probabilistic set membership with no false negatives.

```python
bf = BloomFilter.from_capacity(expected_items=10000, false_positive_rate=0.01)
bf.add("user:123")

print(bf.contains("user:123"))  # True
print(bf.contains("user:456"))  # False (probably)
```

### Reservoir Sampling

Uniform random sample from unbounded stream using O(k) memory.

```python
sampler = ReservoirSampler[Event](size=1000, seed=42)
for event in stream:
    sampler.add(event)

sample = sampler.sample()  # Uniform random sample
```

## Testing Strategy

### Unit Tests (Pure Algorithm Verification)

Test mathematical properties with seeded RNG:

```python
def test_topk_identifies_heavy_hitters():
    topk = TopK[int](k=20, seed=42)
    dist = ZipfDistribution(range(1000), s=1.0, seed=42)

    for _ in range(100000):
        topk.add(dist.sample())

    # True top 10 (ranks 0-9) should mostly be in sketch's top 10
    top_10 = {item.item for item in topk.top(10)}
    assert len(top_10 & set(range(10))) >= 8
```

### Integration Tests (Sketch vs Ideal Comparison)

Run full simulation with both sketch and exact counter:

```python
def test_topk_accuracy_vs_stream_size(test_output_dir):
    # Create both sketch and exact collector as event sinks
    ideal_sink = IdealStatsSink("ideal", key_field="customer_id")
    topk_collector = TopKCollector(name="topk", k=100, ...)

    # Route events to both
    router = Router([ideal_sink, topk_collector])

    # Run simulation with Zipf-distributed customer IDs
    sim = Simulation(sources=[source], entities=[router, ideal_sink, topk_collector])
    sim.run()

    # Compare precision/recall/error
    true_top_k = set(item for item, _ in ideal_sink.counts.most_common(k))
    sketch_top_k = set(item.item for item in topk_collector.top(k))

    precision = len(sketch_top_k & true_top_k) / k
    assert precision > 0.8

    # Visualize accuracy over time
    plt.savefig(test_output_dir / "topk_accuracy.png")
```

### Visualization Outputs

Each integration test produces:
- **topk_accuracy.png** - Precision/recall vs stream size
- **tdigest_accuracy.png** - Quantile error at p50, p95, p99, p999
- **sketch_error_over_time.png** - How error evolves as stream grows
- **memory_vs_accuracy.png** - Tradeoff analysis

## Implementation Plan

### Phase 1: Core Infrastructure
1. `happysimulator/sketching/base.py` - Protocols
2. `happysimulator/sketching/topk.py` - Space-Saving algorithm
3. `happysimulator/sketching/count_min_sketch.py`
4. `tests/unit/sketching/test_topk.py`
5. `tests/unit/sketching/test_count_min_sketch.py`

### Phase 2: Additional Algorithms
1. `happysimulator/sketching/tdigest.py`
2. `happysimulator/sketching/hyperloglog.py`
3. `happysimulator/sketching/bloom_filter.py`
4. `happysimulator/sketching/reservoir.py`
5. Unit tests for each

### Phase 3: Entity Wrappers
1. `happysimulator/components/sketching/sketch_collector.py`
2. `happysimulator/components/sketching/topk_collector.py`
3. `happysimulator/components/sketching/quantile_estimator.py`
4. Update `happysimulator/__init__.py` exports

### Phase 4: Integration Tests
1. `tests/integration/test_sketch_vs_ideal.py`
2. TopK accuracy test with visualization
3. T-Digest quantile accuracy test
4. Error-over-time visualization

## References

- Metwally, Agrawal, El Abbadi. "Efficient Computation of Frequent and Top-k Elements in Data Streams" (2005) - Space-Saving
- Cormode, Muthukrishnan. "An Improved Data Stream Summary: The Count-Min Sketch" (2005)
- Dunning, Ertl. "Computing Extremely Accurate Quantiles Using t-Digests" (2019)
- Flajolet, Fusy, Gandouet, Meunier. "HyperLogLog: the analysis of a near-optimal cardinality estimation algorithm" (2007)

## Key Files to Reference

| File | Pattern to Follow |
|------|-------------------|
| `happysimulator/distributions/zipf.py` | Pure algorithm with sample(), seed, properties |
| `happysimulator/distributions/value_distribution.py` | ABC base class design |
| `happysimulator/core/entity.py` | Entity base class for wrappers |
| `happysimulator/components/datastore/kv_store.py` | Stats dataclass pattern |
| `tests/unit/distributions/test_zipf.py` | Statistical test assertions |
| `tests/integration/test_zipf_distribution_visualization.py` | Ideal vs observed comparison |
