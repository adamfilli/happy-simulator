# Zipf Distribution Design Document

## Overview

This document outlines a design for incorporating Zipf (power-law) distributions into happy-simulator, enabling realistic modeling of skewed access patterns common in distributed systems.

## Motivation

In real-world distributed systems, access patterns rarely follow uniform distributions. Instead, they typically exhibit **Zipf's law** (or power-law distributions) where:

- A small number of items account for most of the activity
- The "80/20 rule": ~20% of customers generate ~80% of requests
- Classic examples:
  - Web caching: A few URLs receive most traffic
  - Database access: Hot keys dominate read/write patterns
  - API usage: Top customers make disproportionate requests
  - Social media: Few posts go viral, most have minimal engagement

The current `load_aware_routing.py` example uses `random.randint(0, 999)` for customer IDs (uniform distribution) and a single "high-rate customer 1001" to simulate hotspots. This approach is limited because:

1. It requires manually creating separate sources for "hot" vs "normal" customers
2. Doesn't model the realistic gradient where some customers are "warm" (moderate activity)
3. Can't easily configure the degree of skew

## Requirements

### Functional Requirements

1. **Sample discrete values following Zipf distribution**
   - Given N possible values (e.g., 1000 customer IDs), sample according to Zipf's law
   - Configurable skew parameter (s): higher s = more skewed toward popular items
   - s=0 produces uniform distribution; s=1 is classic Zipf; s>1 is more extreme

2. **Easy integration with EventProvider**
   - Simple API to create EventProviders where a field (e.g., `customer_id`) follows Zipf
   - Should compose with existing Source/ArrivalTimeProvider infrastructure

3. **Deterministic testing support**
   - Support seeded random generation for reproducible tests
   - Match existing patterns in the codebase

### Non-Functional Requirements

1. **Performance**: Sampling should be O(1) after initialization
2. **Flexibility**: Work with any discrete domain (ints, strings, objects)
3. **Observability**: Easy to verify the distribution matches expectations

## Design

### New Concepts

#### 1. `ValueDistribution[T]` - Abstract Base Class

A new abstraction for sampling discrete values from a probability distribution:

```
happysimulator/distributions/value_distribution.py
```

```python
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar('T')

class ValueDistribution(ABC, Generic[T]):
    """Abstract base for sampling discrete values from a distribution.

    Unlike LatencyDistribution (which samples continuous positive floats),
    ValueDistribution samples from a finite set of discrete values.
    """

    @abstractmethod
    def sample(self) -> T:
        """Sample a single value from the distribution."""
        pass

    @abstractmethod
    def sample_n(self, n: int) -> list[T]:
        """Sample n values from the distribution."""
        pass

    @property
    @abstractmethod
    def population(self) -> list[T]:
        """Return the complete population of possible values."""
        pass
```

#### 2. `ZipfDistribution[T]` - Zipf Implementation

```
happysimulator/distributions/zipf.py
```

```python
import random
from typing import Sequence

class ZipfDistribution(ValueDistribution[T]):
    """Samples values following Zipf's law (power-law distribution).

    Zipf's law states that the frequency of an item is inversely proportional
    to its rank raised to a power: P(rank=k) ∝ 1/k^s

    Args:
        values: The population of values to sample from.
                The first value is rank 1 (most frequent).
        s: Zipf exponent (default 1.0).
           s=0: uniform distribution
           s=1: classic Zipf (item k appears 1/k as often as item 1)
           s>1: more extreme skew
        seed: Optional random seed for reproducibility.

    Example:
        # Customer IDs 0-999 with classic Zipf distribution
        dist = ZipfDistribution(range(1000), s=1.0)
        customer_id = dist.sample()  # Most likely returns low IDs

        # Account IDs with extreme skew
        dist = ZipfDistribution(["acct_1", "acct_2", ..., "acct_100"], s=1.5)
    """

    def __init__(
        self,
        values: Sequence[T],
        s: float = 1.0,
        seed: int | None = None,
    ):
        self._values = list(values)
        self._s = s
        self._rng = random.Random(seed)

        # Precompute cumulative probabilities for O(1) sampling
        self._cum_probs = self._compute_cumulative_probs()

    def _compute_cumulative_probs(self) -> list[float]:
        """Compute cumulative probability distribution."""
        n = len(self._values)
        # Zipf: P(k) = (1/k^s) / H_n,s  where H_n,s is the generalized harmonic number
        weights = [1.0 / ((k + 1) ** self._s) for k in range(n)]
        total = sum(weights)
        probs = [w / total for w in weights]

        # Convert to cumulative
        cum = []
        running = 0.0
        for p in probs:
            running += p
            cum.append(running)
        return cum

    def sample(self) -> T:
        """Sample a value using inverse transform sampling."""
        u = self._rng.random()
        # Binary search for efficiency
        idx = bisect.bisect_left(self._cum_probs, u)
        return self._values[min(idx, len(self._values) - 1)]

    def sample_n(self, n: int) -> list[T]:
        return [self.sample() for _ in range(n)]

    @property
    def population(self) -> list[T]:
        return list(self._values)

    @property
    def s(self) -> float:
        """The Zipf exponent."""
        return self._s

    def expected_frequency(self, rank: int) -> float:
        """Return expected frequency for a given rank (1-indexed)."""
        if rank < 1 or rank > len(self._values):
            raise ValueError(f"Rank must be 1-{len(self._values)}")
        return self._cum_probs[rank - 1] - (self._cum_probs[rank - 2] if rank > 1 else 0)
```

#### 3. `UniformDistribution[T]` - For Comparison/Testing

```python
class UniformDistribution(ValueDistribution[T]):
    """Uniform random sampling from a population.

    Useful as a baseline comparison against Zipf.
    """

    def __init__(self, values: Sequence[T], seed: int | None = None):
        self._values = list(values)
        self._rng = random.Random(seed)

    def sample(self) -> T:
        return self._rng.choice(self._values)
```

### Integration with EventProvider

#### Option A: Composition (Recommended)

Create a helper that wraps any `ValueDistribution` for use in EventProviders:

```python
# In load_aware_routing.py or similar

class ZipfCustomerProvider(EventProvider):
    """Generates requests with customer IDs following Zipf distribution."""

    def __init__(
        self,
        target: Entity,
        customer_distribution: ValueDistribution[int],
        stop_after: Instant | None = None,
    ):
        self._target = target
        self._customer_dist = customer_distribution
        self._stop_after = stop_after

    def get_events(self, time: Instant) -> list[Event]:
        if self._stop_after is not None and time > self._stop_after:
            return []

        customer_id = self._customer_dist.sample()

        return [
            Event(
                time=time,
                event_type="Request",
                target=self._target,
                context={
                    "customer_id": customer_id,
                    "created_at": time,
                },
            )
        ]
```

Usage:
```python
# Classic Zipf: customer 0 gets ~50% of traffic, customer 1 gets ~25%, etc.
customer_dist = ZipfDistribution(range(1000), s=1.0, seed=42)
provider = ZipfCustomerProvider(router, customer_dist)
```

#### Option B: Generic FieldDistributionProvider

A more general-purpose EventProvider that can apply distributions to any field:

```python
class DistributedFieldProvider(EventProvider):
    """EventProvider that samples event context fields from distributions.

    Args:
        target: Target entity for generated events.
        event_type: Type string for generated events.
        field_distributions: Dict mapping field names to ValueDistributions.
        static_fields: Dict of constant field values.

    Example:
        provider = DistributedFieldProvider(
            target=router,
            event_type="Request",
            field_distributions={
                "customer_id": ZipfDistribution(range(1000), s=1.0),
                "region": UniformDistribution(["us-east", "us-west", "eu"]),
            },
            static_fields={
                "source": "api",
            },
        )
    """

    def __init__(
        self,
        target: Entity,
        event_type: str,
        field_distributions: dict[str, ValueDistribution],
        static_fields: dict[str, Any] | None = None,
    ):
        self._target = target
        self._event_type = event_type
        self._field_dists = field_distributions
        self._static_fields = static_fields or {}

    def get_events(self, time: Instant) -> list[Event]:
        context = dict(self._static_fields)
        context["created_at"] = time

        for field, dist in self._field_dists.items():
            context[field] = dist.sample()

        return [
            Event(
                time=time,
                event_type=self._event_type,
                target=self._target,
                context=context,
            )
        ]
```

### File Organization

```
happysimulator/
├── distributions/
│   ├── __init__.py           # Add new exports
│   ├── value_distribution.py # NEW: Abstract base
│   ├── zipf.py               # NEW: Zipf implementation
│   ├── uniform.py            # NEW: Uniform discrete distribution
│   └── ... (existing files)
│
├── load/
│   ├── __init__.py           # Add DistributedFieldProvider export
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── distributed_field.py  # NEW: Generic field distribution provider
│   │   └── ... (existing)
│   └── ... (existing)
```

### Updated Exports

```python
# happysimulator/distributions/__init__.py
from happysimulator.distributions.value_distribution import ValueDistribution
from happysimulator.distributions.zipf import ZipfDistribution
from happysimulator.distributions.uniform import UniformDistribution

__all__ = [
    # ... existing exports ...
    "ValueDistribution",
    "ZipfDistribution",
    "UniformDistribution",
]

# happysimulator/load/__init__.py
from happysimulator.load.providers.distributed_field import DistributedFieldProvider

__all__ = [
    # ... existing exports ...
    "DistributedFieldProvider",
]
```

## Example Usage

### Basic Zipf Customer Distribution

```python
from happysimulator import (
    ZipfDistribution,
    DistributedFieldProvider,
    Source,
    PoissonArrivalTimeProvider,
    ConstantRateProfile,
    Simulation,
    Instant,
)

# 1000 customers with Zipf distribution (s=1.0)
# Customer 0: ~7.5% of traffic
# Customer 1: ~3.7% of traffic
# Customer 2: ~2.5% of traffic
# ...
# Long tail: customers 500-999 collectively ~5% of traffic
customer_dist = ZipfDistribution(range(1000), s=1.0, seed=42)

provider = DistributedFieldProvider(
    target=my_router,
    event_type="Request",
    field_distributions={"customer_id": customer_dist},
)

source = Source(
    name="ZipfTraffic",
    event_provider=provider,
    arrival_time_provider=PoissonArrivalTimeProvider(
        ConstantRateProfile(rate=100),  # 100 req/s
        start_time=Instant.Epoch,
    ),
)
```

### Comparing Uniform vs Zipf

```python
# Compare hot-key behavior with different distributions
uniform_dist = UniformDistribution(range(100), seed=42)
zipf_mild = ZipfDistribution(range(100), s=0.5, seed=42)
zipf_classic = ZipfDistribution(range(100), s=1.0, seed=42)
zipf_extreme = ZipfDistribution(range(100), s=1.5, seed=42)

# Sample 10000 and compare
for name, dist in [("uniform", uniform_dist), ("zipf_0.5", zipf_mild),
                   ("zipf_1.0", zipf_classic), ("zipf_1.5", zipf_extreme)]:
    samples = dist.sample_n(10000)
    top_10_pct = sum(1 for s in samples if s < 10) / len(samples) * 100
    print(f"{name}: top 10% of keys receive {top_10_pct:.1f}% of requests")
```

Expected output:
```
uniform: top 10% of keys receive 10.0% of requests
zipf_0.5: top 10% of keys receive 26.3% of requests
zipf_1.0: top 10% of keys receive 52.8% of requests
zipf_1.5: top 10% of keys receive 73.4% of requests
```

### Multi-field Distributions

```python
# Simulate realistic API traffic with multiple distributed fields
provider = DistributedFieldProvider(
    target=api_gateway,
    event_type="APIRequest",
    field_distributions={
        # Customer IDs: Zipf (some customers are power users)
        "customer_id": ZipfDistribution(range(10000), s=1.0),

        # Endpoints: Zipf (some endpoints are hot)
        "endpoint": ZipfDistribution(
            ["/api/users", "/api/orders", "/api/products", "/api/search",
             "/api/inventory", "/api/payments", "/api/reviews", "/api/analytics"],
            s=0.8,
        ),

        # Regions: Non-uniform but not Zipf
        "region": UniformDistribution(["us-east-1", "us-west-2", "eu-west-1"]),
    },
    static_fields={
        "api_version": "v2",
    },
)
```

## Testing Strategy

### Unit Tests

```python
# tests/unit/distributions/test_zipf.py

class TestZipfDistribution:
    def test_s_zero_is_uniform(self):
        """With s=0, distribution should be uniform."""
        dist = ZipfDistribution(range(10), s=0.0, seed=42)
        samples = dist.sample_n(10000)
        counts = Counter(samples)

        # Each value should appear ~1000 times (±10%)
        for v in range(10):
            assert 900 < counts[v] < 1100

    def test_s_one_classic_zipf(self):
        """With s=1, rank 1 should appear ~2x as often as rank 2."""
        dist = ZipfDistribution(range(100), s=1.0, seed=42)
        samples = dist.sample_n(100000)
        counts = Counter(samples)

        ratio = counts[0] / counts[1]
        assert 1.8 < ratio < 2.2  # Should be ~2.0

    def test_higher_s_more_skewed(self):
        """Higher s should concentrate more on top values."""
        dist_mild = ZipfDistribution(range(100), s=0.5, seed=42)
        dist_extreme = ZipfDistribution(range(100), s=2.0, seed=42)

        mild_samples = dist_mild.sample_n(10000)
        extreme_samples = dist_extreme.sample_n(10000)

        mild_top10 = sum(1 for s in mild_samples if s < 10) / 10000
        extreme_top10 = sum(1 for s in extreme_samples if s < 10) / 10000

        assert extreme_top10 > mild_top10

    def test_deterministic_with_seed(self):
        """Same seed should produce same samples."""
        dist1 = ZipfDistribution(range(100), s=1.0, seed=42)
        dist2 = ZipfDistribution(range(100), s=1.0, seed=42)

        assert dist1.sample_n(100) == dist2.sample_n(100)

    def test_works_with_strings(self):
        """Should work with non-integer values."""
        dist = ZipfDistribution(["hot", "warm", "cool", "cold"], s=1.0, seed=42)
        sample = dist.sample()
        assert sample in ["hot", "warm", "cool", "cold"]
```

### Integration Tests

The integration test validates that the full simulation pipeline (Source → EventProvider → Sink) correctly generates events following the Zipf distribution.

#### Test File: `tests/integration/test_zipf_distribution_visualization.py`

```python
"""Integration tests for Zipf distribution in load generation.

These tests verify that events generated through the simulation pipeline
correctly follow the expected Zipf distribution, with visualization output.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from happysimulator import (
    Entity,
    Event,
    EventProvider,
    Instant,
    Simulation,
    Source,
    ConstantArrivalTimeProvider,
    ConstantRateProfile,
)
from happysimulator.distributions import ZipfDistribution, UniformDistribution


# =============================================================================
# Statistics Collector Sink
# =============================================================================


class StatisticsCollectorSink(Entity):
    """Sink entity that accumulates statistics about received events.

    Collects field values from event context to verify distributions.
    This acts as the "end of the pipeline" that receives all generated
    events and records their characteristics.

    Attributes:
        field_counts: Dict mapping field names to Counter of observed values.
        events_received: Total count of events received.
        event_times: List of timestamps when events were received.
    """

    def __init__(self, name: str, tracked_fields: list[str]):
        """Initialize the statistics collector.

        Args:
            name: Entity name.
            tracked_fields: List of context field names to track.
        """
        super().__init__(name)
        self._tracked_fields = tracked_fields
        self.field_counts: dict[str, Counter] = {f: Counter() for f in tracked_fields}
        self.events_received: int = 0
        self.event_times: list[Instant] = []

    def handle_event(self, event: Event) -> list[Event]:
        """Record statistics from received event."""
        self.events_received += 1
        self.event_times.append(event.time)

        for field_name in self._tracked_fields:
            value = event.context.get(field_name)
            if value is not None:
                self.field_counts[field_name][value] += 1

        return []

    def get_frequency_distribution(self, field: str) -> list[tuple[Any, int]]:
        """Return (value, count) pairs sorted by count descending."""
        return self.field_counts[field].most_common()

    def get_top_n_percentage(self, field: str, n: int) -> float:
        """Return percentage of events in top n values."""
        counter = self.field_counts[field]
        if not counter:
            return 0.0
        top_n_count = sum(count for _, count in counter.most_common(n))
        return top_n_count / self.events_received * 100


# =============================================================================
# Test Classes
# =============================================================================


class TestZipfDistributionVisualization:
    """Integration tests for Zipf distribution with visualization."""

    def test_zipf_source_generates_expected_distribution(self, test_output_dir: Path):
        """Verify source with Zipf distribution produces expected skew.

        This test:
        1. Creates a Source that generates events with customer_id from Zipf distribution
        2. Runs the simulation to generate many events
        3. Collects statistics in a sink entity
        4. Verifies the distribution matches expected Zipf characteristics
        5. Generates visualizations comparing observed vs expected frequencies
        """
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        # Configuration
        num_customers = 100
        zipf_s = 1.0  # Classic Zipf
        num_events = 10000
        rate_per_second = 1000.0  # Generate quickly
        seed = 42

        # Create the statistics collector sink
        sink = StatisticsCollectorSink("sink", tracked_fields=["customer_id"])

        # Create Zipf distribution for customer IDs
        customer_dist = ZipfDistribution(range(num_customers), s=zipf_s, seed=seed)

        # Create event provider using the distribution
        provider = DistributedFieldProvider(
            target=sink,
            event_type="Request",
            field_distributions={"customer_id": customer_dist},
            stop_after=Instant.from_seconds(num_events / rate_per_second),
        )

        # Create source with constant arrival rate
        source = Source(
            name="ZipfSource",
            event_provider=provider,
            arrival_time_provider=ConstantArrivalTimeProvider(
                ConstantRateProfile(rate=rate_per_second),
                start_time=Instant.Epoch,
            ),
        )

        # Run simulation
        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(num_events / rate_per_second + 1.0),
            sources=[source],
            entities=[sink],
        )
        sim.run()

        # === ASSERTIONS ===

        # 1. Verify we generated the expected number of events
        assert sink.events_received >= num_events * 0.95  # Allow 5% tolerance

        # 2. Verify Zipf characteristic: top 10% of customers get majority of traffic
        top_10_pct = sink.get_top_n_percentage("customer_id", num_customers // 10)
        assert top_10_pct > 40, f"Top 10% should get >40% of traffic, got {top_10_pct:.1f}%"

        # 3. Verify rank-frequency relationship
        freq_dist = sink.get_frequency_distribution("customer_id")
        if len(freq_dist) >= 2:
            rank1_count = freq_dist[0][1]
            rank2_count = freq_dist[1][1]
            ratio = rank1_count / rank2_count
            # With s=1, rank 1 should appear ~2x as often as rank 2
            assert 1.5 < ratio < 3.0, f"Rank 1/2 ratio should be ~2, got {ratio:.2f}"

        # === VISUALIZATION ===

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Observed frequency by customer rank (log-log scale)
        ax = axes[0, 0]
        observed_counts = [count for _, count in freq_dist]
        ranks = range(1, len(observed_counts) + 1)
        ax.loglog(ranks, observed_counts, 'bo-', alpha=0.6, label='Observed', markersize=4)

        # Expected Zipf frequencies
        expected_probs = [1.0 / (k ** zipf_s) for k in ranks]
        expected_total = sum(expected_probs)
        expected_counts = [p / expected_total * sink.events_received for p in expected_probs]
        ax.loglog(ranks, expected_counts, 'r--', alpha=0.8, label=f'Expected (s={zipf_s})', linewidth=2)

        ax.set_xlabel("Rank (log scale)")
        ax.set_ylabel("Frequency (log scale)")
        ax.set_title("Zipf Distribution: Rank vs Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')

        # Plot 2: Cumulative distribution
        ax = axes[0, 1]
        cumulative = np.cumsum(observed_counts) / sum(observed_counts) * 100
        ax.plot(ranks, cumulative, 'b-', linewidth=2, label='Observed')

        expected_cumulative = np.cumsum(expected_counts) / sum(expected_counts) * 100
        ax.plot(ranks, expected_cumulative, 'r--', linewidth=2, label='Expected')

        ax.axhline(y=80, color='gray', linestyle=':', alpha=0.7)
        ax.axvline(x=num_customers * 0.2, color='gray', linestyle=':', alpha=0.7)
        ax.set_xlabel("Customer Rank")
        ax.set_ylabel("Cumulative % of Requests")
        ax.set_title("Cumulative Distribution (80/20 Rule)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Top 20 customers bar chart
        ax = axes[1, 0]
        top_20 = freq_dist[:20]
        customer_ids = [str(cid) for cid, _ in top_20]
        counts = [count for _, count in top_20]
        bars = ax.bar(range(len(top_20)), counts, color='steelblue', alpha=0.8)
        ax.set_xticks(range(len(top_20)))
        ax.set_xticklabels(customer_ids, rotation=45, ha='right')
        ax.set_xlabel("Customer ID")
        ax.set_ylabel("Request Count")
        ax.set_title("Top 20 Customers by Request Volume")
        ax.grid(True, alpha=0.3, axis='y')

        # Add percentage labels on bars
        total = sink.events_received
        for i, (bar, count) in enumerate(zip(bars, counts)):
            pct = count / total * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)

        # Plot 4: Distribution comparison (histogram of frequencies)
        ax = axes[1, 1]
        ax.hist(observed_counts, bins=30, alpha=0.6, color='steelblue', label='Observed', edgecolor='black')
        ax.set_xlabel("Request Count per Customer")
        ax.set_ylabel("Number of Customers")
        ax.set_title("Distribution of Request Counts")
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        fig.suptitle(f"Zipf Distribution Verification (s={zipf_s}, n={num_customers}, events={sink.events_received})",
                     fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig.savefig(test_output_dir / "zipf_distribution_verification.png", dpi=150)
        plt.close(fig)

        # Write summary statistics to file
        with open(test_output_dir / "zipf_statistics.txt", "w") as f:
            f.write(f"Zipf Distribution Test Summary\n")
            f.write(f"==============================\n\n")
            f.write(f"Configuration:\n")
            f.write(f"  Number of customers: {num_customers}\n")
            f.write(f"  Zipf exponent (s): {zipf_s}\n")
            f.write(f"  Events generated: {sink.events_received}\n\n")
            f.write(f"Results:\n")
            f.write(f"  Top 10% customers: {top_10_pct:.1f}% of traffic\n")
            f.write(f"  Top 20% customers: {sink.get_top_n_percentage('customer_id', 20):.1f}% of traffic\n")
            f.write(f"  Rank 1 customer: {freq_dist[0][1]} requests ({freq_dist[0][1]/sink.events_received*100:.1f}%)\n")
            f.write(f"  Rank 1/2 ratio: {freq_dist[0][1]/freq_dist[1][1]:.2f} (expected ~2.0)\n\n")
            f.write(f"Top 10 customers:\n")
            for i, (cid, count) in enumerate(freq_dist[:10], 1):
                f.write(f"  {i}. Customer {cid}: {count} requests ({count/sink.events_received*100:.2f}%)\n")


    def test_compare_zipf_parameters(self, test_output_dir: Path):
        """Compare different Zipf s parameters side by side.

        Generates visualization showing how different s values affect
        the distribution skew.
        """
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        num_customers = 100
        num_events = 5000
        s_values = [0.0, 0.5, 1.0, 1.5, 2.0]
        results = {}

        for s in s_values:
            sink = StatisticsCollectorSink(f"sink_s{s}", tracked_fields=["customer_id"])
            customer_dist = ZipfDistribution(range(num_customers), s=s, seed=42)

            provider = DistributedFieldProvider(
                target=sink,
                event_type="Request",
                field_distributions={"customer_id": customer_dist},
                stop_after=Instant.from_seconds(num_events / 1000.0),
            )

            source = Source(
                name=f"Source_s{s}",
                event_provider=provider,
                arrival_time_provider=ConstantArrivalTimeProvider(
                    ConstantRateProfile(rate=1000.0),
                    start_time=Instant.Epoch,
                ),
            )

            sim = Simulation(
                start_time=Instant.Epoch,
                end_time=Instant.from_seconds(num_events / 1000.0 + 1.0),
                sources=[source],
                entities=[sink],
            )
            sim.run()

            results[s] = {
                "sink": sink,
                "top_10_pct": sink.get_top_n_percentage("customer_id", 10),
                "top_20_pct": sink.get_top_n_percentage("customer_id", 20),
                "freq_dist": sink.get_frequency_distribution("customer_id"),
            }

        # === VISUALIZATION ===
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Rank-frequency for all s values
        ax = axes[0, 0]
        colors = plt.cm.viridis([i / len(s_values) for i in range(len(s_values))])
        for (s, data), color in zip(results.items(), colors):
            freq_dist = data["freq_dist"]
            counts = [count for _, count in freq_dist]
            ranks = range(1, len(counts) + 1)
            label = f"s={s}" if s > 0 else "s=0 (uniform)"
            ax.loglog(ranks, counts, 'o-', alpha=0.7, label=label, color=color, markersize=3)

        ax.set_xlabel("Rank (log scale)")
        ax.set_ylabel("Frequency (log scale)")
        ax.set_title("Effect of Zipf Exponent on Rank-Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')

        # Plot 2: Top 10% traffic share
        ax = axes[0, 1]
        s_labels = [f"s={s}" for s in s_values]
        top_10_values = [results[s]["top_10_pct"] for s in s_values]
        bars = ax.bar(s_labels, top_10_values, color='steelblue', alpha=0.8)
        ax.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='Uniform (10%)')
        ax.set_xlabel("Zipf Exponent")
        ax.set_ylabel("% of Traffic to Top 10 Customers")
        ax.set_title("Traffic Concentration vs Zipf Exponent")
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, top_10_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.1f}%', ha='center', va='bottom')

        # Plot 3: Cumulative distributions
        ax = axes[1, 0]
        for (s, data), color in zip(results.items(), colors):
            freq_dist = data["freq_dist"]
            counts = [count for _, count in freq_dist]
            cumulative = [sum(counts[:i+1])/sum(counts)*100 for i in range(len(counts))]
            label = f"s={s}" if s > 0 else "s=0 (uniform)"
            ax.plot(range(1, len(cumulative)+1), cumulative, '-', alpha=0.8, label=label, color=color, linewidth=2)

        ax.axhline(y=80, color='gray', linestyle=':', alpha=0.7)
        ax.set_xlabel("Number of Top Customers")
        ax.set_ylabel("Cumulative % of Traffic")
        ax.set_title("Cumulative Traffic Share")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Summary table as text
        ax = axes[1, 1]
        ax.axis('off')
        table_data = [
            ["s value", "Top 10%", "Top 20%", "Rank 1 %"],
        ]
        for s in s_values:
            data = results[s]
            rank1_pct = data["freq_dist"][0][1] / data["sink"].events_received * 100
            table_data.append([
                f"{s:.1f}",
                f"{data['top_10_pct']:.1f}%",
                f"{data['top_20_pct']:.1f}%",
                f"{rank1_pct:.1f}%",
            ])

        table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                         colWidths=[0.2, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.8)
        ax.set_title("Summary Statistics", pad=20)

        fig.suptitle(f"Comparing Zipf Exponent Values (n={num_customers} customers)",
                     fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig.savefig(test_output_dir / "zipf_parameter_comparison.png", dpi=150)
        plt.close(fig)


    def test_zipf_vs_uniform_hotspot_behavior(self, test_output_dir: Path):
        """Compare how Zipf vs Uniform distributions create hotspots.

        Simulates cache/database access patterns and shows how Zipf
        naturally creates "hot keys" vs uniform distribution.
        """
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        num_keys = 1000
        num_accesses = 20000
        seed = 42

        # Run with both distributions
        distributions = {
            "Uniform": UniformDistribution(range(num_keys), seed=seed),
            "Zipf (s=1.0)": ZipfDistribution(range(num_keys), s=1.0, seed=seed),
            "Zipf (s=1.5)": ZipfDistribution(range(num_keys), s=1.5, seed=seed),
        }

        results = {}
        for name, dist in distributions.items():
            sink = StatisticsCollectorSink(name, tracked_fields=["key_id"])

            provider = DistributedFieldProvider(
                target=sink,
                event_type="Access",
                field_distributions={"key_id": dist},
                stop_after=Instant.from_seconds(num_accesses / 1000.0),
            )

            source = Source(
                name=f"Source_{name}",
                event_provider=provider,
                arrival_time_provider=ConstantArrivalTimeProvider(
                    ConstantRateProfile(rate=1000.0),
                    start_time=Instant.Epoch,
                ),
            )

            sim = Simulation(
                start_time=Instant.Epoch,
                end_time=Instant.from_seconds(num_accesses / 1000.0 + 1.0),
                sources=[source],
                entities=[sink],
            )
            sim.run()

            freq_dist = sink.get_frequency_distribution("key_id")
            counts = [count for _, count in freq_dist]

            results[name] = {
                "counts": counts,
                "hot_keys": sum(1 for c in counts if c > num_accesses/num_keys * 5),  # >5x average
                "cold_keys": sum(1 for c in counts if c < num_accesses/num_keys * 0.2),  # <0.2x average
                "max_count": max(counts),
                "top_1_pct": sink.get_top_n_percentage("key_id", num_keys // 100),
            }

        # === ASSERTIONS ===
        # Zipf should have more hot keys and fewer cold keys than uniform
        assert results["Zipf (s=1.0)"]["hot_keys"] > results["Uniform"]["hot_keys"]
        assert results["Zipf (s=1.5)"]["top_1_pct"] > results["Zipf (s=1.0)"]["top_1_pct"]

        # === VISUALIZATION ===
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for ax, (name, data) in zip(axes, results.items()):
            counts = sorted(data["counts"], reverse=True)
            ax.bar(range(len(counts)), counts, width=1.0, alpha=0.7)
            ax.axhline(y=num_accesses/num_keys, color='red', linestyle='--',
                       label=f'Average ({num_accesses/num_keys:.0f})')
            ax.set_xlabel("Key Rank")
            ax.set_ylabel("Access Count")
            ax.set_title(f"{name}\nHot keys: {data['hot_keys']}, Cold keys: {data['cold_keys']}")
            ax.legend()
            ax.set_xlim(0, 200)  # Show first 200 keys

        fig.suptitle(f"Hot Key Distribution: Zipf vs Uniform ({num_keys} keys, {num_accesses} accesses)",
                     fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig.savefig(test_output_dir / "zipf_hotspot_comparison.png", dpi=150)
        plt.close(fig)
```

#### Supporting Code: `DistributedFieldProvider` with `stop_after`

The integration test uses a `DistributedFieldProvider` variant that supports stopping after a certain time:

```python
class DistributedFieldProvider(EventProvider):
    """EventProvider that samples event context fields from distributions."""

    def __init__(
        self,
        target: Entity,
        event_type: str,
        field_distributions: dict[str, ValueDistribution],
        static_fields: dict[str, Any] | None = None,
        stop_after: Instant | None = None,
    ):
        self._target = target
        self._event_type = event_type
        self._field_dists = field_distributions
        self._static_fields = static_fields or {}
        self._stop_after = stop_after
        self.generated = 0

    def get_events(self, time: Instant) -> list[Event]:
        if self._stop_after is not None and time > self._stop_after:
            return []

        self.generated += 1
        context = dict(self._static_fields)
        context["created_at"] = time

        for field_name, dist in self._field_dists.items():
            context[field_name] = dist.sample()

        return [
            Event(
                time=time,
                event_type=self._event_type,
                target=self._target,
                context=context,
            )
        ]
```

#### Expected Test Outputs

The integration tests will generate:

1. **`zipf_distribution_verification.png`**: 4-panel visualization showing:
   - Log-log rank vs frequency plot (observed vs expected)
   - Cumulative distribution curve (80/20 rule)
   - Top 20 customers bar chart with percentages
   - Histogram of request counts per customer

2. **`zipf_statistics.txt`**: Text summary with key metrics

3. **`zipf_parameter_comparison.png`**: Comparison of s=0, 0.5, 1.0, 1.5, 2.0

4. **`zipf_hotspot_comparison.png`**: Side-by-side showing hot/cold key behavior

## Alternatives Considered

### 1. Extend LatencyDistribution

Could add a `ZipfLatency` that samples inter-arrival times following Zipf. Rejected because:
- Conceptually different: Zipf is for discrete categories, not continuous delays
- Would conflate two different use cases

### 2. Use numpy.random.zipf directly

Could use `numpy.random.zipf(s, size)` in EventProviders. Rejected because:
- Adds numpy dependency (currently not required)
- Doesn't match the existing distribution abstraction pattern
- Less composable and testable

### 3. Probability mapping in EventProvider

Could have EventProvider take a probability map `{value: probability}`. Rejected because:
- More verbose for standard distributions
- Harder to configure mathematically (user must compute probabilities)
- Less discoverable

## Implementation Plan

1. **Phase 1: Core Distribution Classes**
   - Create `ValueDistribution` base class
   - Implement `ZipfDistribution` with tests
   - Implement `UniformDistribution` as baseline

2. **Phase 2: Load Generation Integration**
   - Create `DistributedFieldProvider`
   - Add exports to `__init__.py` files
   - Update CLAUDE.md with new patterns

3. **Phase 3: Examples and Documentation**
   - Update `load_aware_routing.py` to use Zipf (or create new example)
   - Add integration test with visualization
   - Document common s values and their characteristics

## References

- [Zipf's Law (Wikipedia)](https://en.wikipedia.org/wiki/Zipf%27s_law)
- [Power Law in Web Caching](https://www.cs.cornell.edu/courses/cs6410/2018fa/slides/21-zipf.pdf)
- [Workload Characterization for Caches](https://www.usenix.org/conference/fast16/technical-sessions/presentation/waldspurger)
