"""T-Digest for quantile/percentile estimation.

T-Digest is a data structure for accurate estimation of quantiles (percentiles)
using bounded memory. It clusters streaming data points into centroids and
provides excellent accuracy at the tails (p99, p999) where it matters most.

Key properties:
- Space: O(compression) centroids
- Update: O(log n) amortized
- Query: O(log n)
- Accuracy: Better at extreme quantiles (p1, p99) than middle (p50)

This is ideal for:
- Latency percentile tracking (p50, p95, p99, p999)
- Response time distributions
- Any metric where tail accuracy matters

Reference:
    Dunning, Ertl. "Computing Extremely Accurate Quantiles Using t-Digests" (2019)
"""

from __future__ import annotations

import bisect
import math
import sys
from dataclasses import dataclass, field
from typing import Iterator

from happysimulator.sketching.base import QuantileSketch


@dataclass(slots=True)
class _Centroid:
    """A centroid (cluster) in the T-Digest.

    Represents a group of data points by their mean and count.
    """
    mean: float
    count: int

    def merge(self, other: "_Centroid") -> "_Centroid":
        """Merge two centroids into one."""
        total = self.count + other.count
        new_mean = (self.mean * self.count + other.mean * other.count) / total
        return _Centroid(mean=new_mean, count=total)


class TDigest(QuantileSketch):
    """T-Digest for streaming quantile estimation.

    Maintains a compressed representation of the data distribution using
    weighted centroids. Provides excellent accuracy at extreme quantiles
    (p1, p99, p999) while using bounded memory.

    Args:
        compression: Controls accuracy vs memory tradeoff (default 100).
            Higher values = more centroids = more accuracy = more memory.
            Typical values: 50-500.
        seed: Not used (algorithm is deterministic) but accepted for API consistency.

    Example:
        # Track latency distribution
        td = TDigest(compression=100)

        for latency in latencies:
            td.add(latency)

        # Get percentiles
        p50 = td.percentile(50)
        p95 = td.percentile(95)
        p99 = td.percentile(99)
        p999 = td.percentile(99.9)
    """

    def __init__(self, compression: float = 100.0, seed: int | None = None):
        """Initialize T-Digest.

        Args:
            compression: Controls number of centroids. Must be > 0.
            seed: Unused (algorithm is deterministic). Accepted for API consistency.

        Raises:
            ValueError: If compression <= 0.
        """
        if compression <= 0:
            raise ValueError(f"compression must be positive, got {compression}")

        self._compression = compression
        self._centroids: list[_Centroid] = []
        self._total_count = 0
        self._min_value: float | None = None
        self._max_value: float | None = None

        # Buffer for batch processing (better performance)
        self._buffer: list[float] = []
        self._buffer_size = int(compression * 2)

    @property
    def compression(self) -> float:
        """Compression factor (higher = more accuracy)."""
        return self._compression

    def _k_scale(self, q: float) -> float:
        """Scale function for T-Digest (k_1 from the paper).

        Maps quantile q to a scale value. The scale function determines
        centroid size limits at different quantiles.
        """
        # k_1(q) = (δ/2π) * arcsin(2q - 1)
        # Scaled by compression factor
        return self._compression * (math.asin(2 * q - 1) / math.pi + 0.5)

    def _max_size(self, q: float) -> float:
        """Maximum centroid size at quantile q.

        Centroids near q=0 or q=1 are kept smaller for better tail accuracy.
        Based on the derivative of k_scale: allows more points in the middle.
        """
        # The derivative of asin(2q-1)/π + 0.5 is 2/(π * sqrt(1 - (2q-1)²))
        # = 2/(π * sqrt(4q - 4q²)) = 2/(π * 2*sqrt(q*(1-q))) = 1/(π*sqrt(q*(1-q)))
        # This gets large at q=0.5 (middle) and small near q=0 or q=1 (tails)
        q = max(0.0001, min(0.9999, q))  # Avoid division by zero
        return self._total_count * 4 / (self._compression * math.pi * math.sqrt(q * (1 - q)))

    def add(self, value: float, count: int = 1) -> None:
        """Add a value to the digest.

        Args:
            value: The value to add.
            count: Number of times to add this value (default 1).

        Raises:
            ValueError: If count is negative.
        """
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        if count == 0:
            return

        # Track min/max
        if self._min_value is None or value < self._min_value:
            self._min_value = value
        if self._max_value is None or value > self._max_value:
            self._max_value = value

        self._total_count += count

        # Add to buffer for batch processing
        for _ in range(count):
            self._buffer.append(value)

        # Flush buffer if full
        if len(self._buffer) >= self._buffer_size:
            self._flush()

    def _flush(self) -> None:
        """Flush buffered values into centroids."""
        if not self._buffer:
            return

        # Sort buffer and merge into centroids
        self._buffer.sort()

        # Add buffered values as new centroids
        for value in self._buffer:
            self._centroids.append(_Centroid(mean=value, count=1))

        self._buffer.clear()

        # Compress centroids
        self._compress()

    def _compress(self) -> None:
        """Compress centroids to stay within memory bounds."""
        if len(self._centroids) <= 1:
            return

        # Sort centroids by mean
        self._centroids.sort(key=lambda c: c.mean)

        # Merge adjacent centroids that are small enough
        compressed: list[_Centroid] = []
        running_count = 0

        for centroid in self._centroids:
            if not compressed:
                compressed.append(centroid)
                running_count = centroid.count
                continue

            # Calculate quantile position
            q = (running_count + centroid.count / 2) / self._total_count
            max_size = self._max_size(q)

            # Check if we can merge with last centroid
            last = compressed[-1]
            if last.count + centroid.count <= max_size:
                compressed[-1] = last.merge(centroid)
            else:
                compressed.append(centroid)

            running_count += centroid.count

        self._centroids = compressed

    def quantile(self, q: float) -> float:
        """Estimate the value at a given quantile.

        Args:
            q: Quantile to estimate (0.0 to 1.0).

        Returns:
            Estimated value at the quantile.

        Raises:
            ValueError: If q is not in [0, 1] or digest is empty.
        """
        if not 0 <= q <= 1:
            raise ValueError(f"Quantile must be in [0, 1], got {q}")

        # Flush any buffered values
        self._flush()

        if not self._centroids:
            raise ValueError("Cannot compute quantile of empty digest")

        if q == 0:
            return self._min_value if self._min_value is not None else self._centroids[0].mean
        if q == 1:
            return self._max_value if self._max_value is not None else self._centroids[-1].mean

        # Find the target count
        target_count = q * self._total_count

        # Walk through centroids
        running_count = 0.0
        for i, centroid in enumerate(self._centroids):
            # Calculate weight boundaries
            if i == 0:
                left_weight = 0.0
            else:
                left_weight = running_count

            if i == len(self._centroids) - 1:
                right_weight = self._total_count
            else:
                right_weight = running_count + centroid.count

            # Check if target is in this centroid's range
            if left_weight <= target_count <= right_weight:
                # Interpolate within this centroid
                if i == 0:
                    # First centroid: interpolate from min
                    if self._min_value is not None and target_count < centroid.count / 2:
                        t = target_count / (centroid.count / 2)
                        return self._min_value + t * (centroid.mean - self._min_value)
                    return centroid.mean
                elif i == len(self._centroids) - 1:
                    # Last centroid: interpolate to max
                    if self._max_value is not None:
                        remaining = self._total_count - running_count
                        if target_count > running_count + remaining / 2:
                            t = (target_count - running_count - remaining / 2) / (remaining / 2)
                            return centroid.mean + t * (self._max_value - centroid.mean)
                    return centroid.mean
                else:
                    # Middle centroid: interpolate between adjacent centroids
                    prev = self._centroids[i - 1]
                    t = (target_count - left_weight) / centroid.count
                    if t < 0.5:
                        # Closer to previous centroid
                        return prev.mean + (centroid.mean - prev.mean) * (0.5 + t)
                    else:
                        # Closer to this centroid
                        return centroid.mean

            running_count += centroid.count

        # Fallback to last centroid
        return self._centroids[-1].mean

    def cdf(self, value: float) -> float:
        """Estimate the cumulative distribution function at a value.

        Args:
            value: The value to get CDF for.

        Returns:
            Estimated probability that a random sample <= value (0.0 to 1.0).
        """
        # Flush any buffered values
        self._flush()

        if not self._centroids:
            return 0.0

        if self._min_value is not None and value <= self._min_value:
            return 0.0
        if self._max_value is not None and value >= self._max_value:
            return 1.0

        # Count values below this value
        count_below = 0.0

        for i, centroid in enumerate(self._centroids):
            if centroid.mean >= value:
                # Interpolate within this centroid
                if i == 0:
                    # Before first centroid
                    if self._min_value is not None:
                        t = (value - self._min_value) / (centroid.mean - self._min_value)
                        return t * (centroid.count / 2) / self._total_count
                    return 0.0
                else:
                    # Between previous and this centroid
                    prev = self._centroids[i - 1]
                    if prev.mean < value < centroid.mean:
                        # Linear interpolation
                        t = (value - prev.mean) / (centroid.mean - prev.mean)
                        partial = t * centroid.count / 2
                        return (count_below + partial) / self._total_count

                return count_below / self._total_count

            count_below += centroid.count

        return 1.0

    def merge(self, other: "TDigest") -> None:
        """Merge another T-Digest into this one.

        Args:
            other: Another T-Digest to merge.

        Raises:
            TypeError: If other is not a TDigest.
        """
        if not isinstance(other, TDigest):
            raise TypeError(f"Can only merge with TDigest, got {type(other).__name__}")

        # Flush both
        self._flush()
        other._flush()

        # Merge centroids
        self._centroids.extend(other._centroids)

        # Update stats
        self._total_count += other._total_count
        if other._min_value is not None:
            if self._min_value is None or other._min_value < self._min_value:
                self._min_value = other._min_value
        if other._max_value is not None:
            if self._max_value is None or other._max_value > self._max_value:
                self._max_value = other._max_value

        # Re-compress
        self._compress()

    @property
    def memory_bytes(self) -> int:
        """Estimated memory usage in bytes."""
        # Each centroid: mean (8) + count (8) = 16 bytes
        # Plus list overhead
        centroid_bytes = len(self._centroids) * 16
        buffer_bytes = len(self._buffer) * 8
        return centroid_bytes + buffer_bytes + sys.getsizeof(self)

    @property
    def item_count(self) -> int:
        """Total count of items added."""
        return self._total_count

    @property
    def centroid_count(self) -> int:
        """Number of centroids currently stored."""
        self._flush()
        return len(self._centroids)

    @property
    def min(self) -> float | None:
        """Minimum value seen."""
        return self._min_value

    @property
    def max(self) -> float | None:
        """Maximum value seen."""
        return self._max_value

    def clear(self) -> None:
        """Reset the digest to empty state."""
        self._centroids.clear()
        self._buffer.clear()
        self._total_count = 0
        self._min_value = None
        self._max_value = None

    def __repr__(self) -> str:
        self._flush()
        return (
            f"TDigest(compression={self._compression}, "
            f"centroids={len(self._centroids)}, total={self._total_count})"
        )
