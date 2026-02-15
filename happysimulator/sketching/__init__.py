"""Streaming/sketching algorithms for approximate statistics.

This module provides space-efficient algorithms for computing approximate
statistics over data streams. These are useful in simulations for:
- Tracking heavy hitters (most frequent items)
- Estimating item frequencies
- Computing approximate quantiles/percentiles
- Estimating cardinality (distinct count)
- Testing set membership
- Uniform random sampling

All algorithms share common properties:
- Bounded memory usage (configurable)
- Single-pass processing (add items one at a time)
- Mergeable (combine sketches from parallel streams)
- Reproducible (optional seed for deterministic behavior)

Quick Reference:
    TopK: Heavy hitters (top-k most frequent items)
    CountMinSketch: Frequency estimation for any item
    TDigest: Quantile/percentile estimation (p50, p95, p99)
    HyperLogLog: Cardinality (distinct count) estimation
    BloomFilter: Probabilistic set membership testing
    ReservoirSampler: Uniform random sampling from streams

Example:
    from happysimulator.sketching import (
        TopK, CountMinSketch, TDigest,
        HyperLogLog, BloomFilter, ReservoirSampler,
    )

    # Track top 100 customers by request volume
    topk = TopK[int](k=100)
    for customer_id in requests:
        topk.add(customer_id)
    print(topk.top(10))  # Top 10 most active customers

    # Track latency percentiles
    td = TDigest(compression=100)
    for latency in latencies:
        td.add(latency)
    print(f"p99: {td.percentile(99)}")

    # Count unique visitors
    hll = HyperLogLog[str](precision=14)
    for visitor_id in visitors:
        hll.add(visitor_id)
    print(f"~{hll.cardinality()} unique visitors")
"""

# Base protocols
from happysimulator.sketching.base import (
    CardinalitySketch,
    FrequencyEstimate,
    FrequencySketch,
    MembershipSketch,
    QuantileSketch,
    SamplingSketch,
    Sketch,
)

# Membership testing
from happysimulator.sketching.bloom_filter import BloomFilter
from happysimulator.sketching.count_min_sketch import CountMinSketch

# Cardinality estimation
from happysimulator.sketching.hyperloglog import HyperLogLog

# Hash trees
from happysimulator.sketching.merkle_tree import KeyRange, MerkleNode, MerkleTree

# Sampling
from happysimulator.sketching.reservoir import ReservoirSampler

# Quantile estimation
from happysimulator.sketching.tdigest import TDigest

# Frequency estimation
from happysimulator.sketching.topk import TopK

__all__ = [
    # Membership testing
    "BloomFilter",
    "CardinalitySketch",
    "CountMinSketch",
    "FrequencyEstimate",
    "FrequencySketch",
    # Cardinality estimation
    "HyperLogLog",
    "KeyRange",
    "MembershipSketch",
    "MerkleNode",
    # Hash trees
    "MerkleTree",
    "QuantileSketch",
    # Sampling
    "ReservoirSampler",
    "SamplingSketch",
    # Protocols
    "Sketch",
    # Quantile estimation
    "TDigest",
    # Frequency estimation
    "TopK",
]
