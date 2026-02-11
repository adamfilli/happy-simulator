"""Phi Accrual Failure Detector for distributed systems.

Implements the phi accrual failure detection algorithm from Hayashibara et al.
Instead of providing a binary alive/dead decision, it outputs a continuous
suspicion level (phi) that can be compared against a threshold.

The detector maintains a sliding window of heartbeat inter-arrival times
and uses a normal distribution model to compute the probability that the
monitored node has failed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PhiAccrualStats:
    """Statistics snapshot from a PhiAccrualDetector.

    Attributes:
        heartbeats_received: Total heartbeats recorded.
        current_phi: Most recent phi value (or 0 if unavailable).
        mean_interval: Mean inter-arrival time in seconds.
        std_interval: Standard deviation of inter-arrival times.
        is_suspected: Whether phi exceeds the threshold.
    """
    heartbeats_received: int
    current_phi: float
    mean_interval: float
    std_interval: float
    is_suspected: bool


class PhiAccrualDetector:
    """Phi accrual failure detector using a sliding window of heartbeat intervals.

    The phi value represents `-log10(P(no heartbeat | normal behavior))`.
    Higher phi means more suspicion. A phi of 1 means ~10% chance the node
    is still alive; phi of 3 means ~0.1% chance.

    Args:
        threshold: Phi value above which the node is considered suspected.
            Default 8.0 (~1 in 100 million false positive rate).
        max_sample_size: Maximum number of inter-arrival intervals to keep.
        min_std: Minimum standard deviation to prevent division by zero.
        initial_interval: Expected heartbeat interval for bootstrapping.
    """

    def __init__(
        self,
        threshold: float = 8.0,
        max_sample_size: int = 200,
        min_std: float = 0.1,
        initial_interval: float | None = None,
    ) -> None:
        self._threshold = threshold
        self._max_sample_size = max_sample_size
        self._min_std = min_std
        self._intervals: list[float] = []
        self._last_heartbeat: float | None = None
        self._heartbeat_count: int = 0
        if initial_interval is not None and initial_interval > 0:
            self._intervals.append(initial_interval)

    def heartbeat(self, timestamp_s: float) -> None:
        """Record a heartbeat arrival.

        Args:
            timestamp_s: The timestamp (in seconds) of the heartbeat.
        """
        self._heartbeat_count += 1
        if self._last_heartbeat is not None:
            interval = timestamp_s - self._last_heartbeat
            if interval > 0:
                self._intervals.append(interval)
                if len(self._intervals) > self._max_sample_size:
                    self._intervals.pop(0)
        self._last_heartbeat = timestamp_s

    def phi(self, now_s: float) -> float:
        """Compute the current phi value.

        Args:
            now_s: Current time in seconds.

        Returns:
            The phi suspicion level. Returns 0.0 if insufficient data.
        """
        if self._last_heartbeat is None or len(self._intervals) < 1:
            return 0.0

        elapsed = now_s - self._last_heartbeat
        if elapsed < 0:
            return 0.0

        mean = self._mean()
        std = max(self._std(), self._min_std)

        # P(X > elapsed) where X ~ Normal(mean, std)
        # Using the complementary CDF: 1 - Phi((elapsed - mean) / std)
        y = (elapsed - mean) / std
        # Use erfc for numerical stability
        p = 0.5 * math.erfc(y / math.sqrt(2))

        if p <= 0:
            return float('inf')

        return -math.log10(p)

    def is_available(self, now_s: float) -> bool:
        """Check if the monitored node is considered available.

        Args:
            now_s: Current time in seconds.

        Returns:
            True if phi is below the threshold (node is available).
        """
        return self.phi(now_s) < self._threshold

    @property
    def stats(self) -> PhiAccrualStats:
        """Current statistics snapshot.

        Note: current_phi requires a timestamp; this returns 0.0 for phi.
        Use stats_at(now_s) for a phi-inclusive snapshot.
        """
        return PhiAccrualStats(
            heartbeats_received=self._heartbeat_count,
            current_phi=0.0,
            mean_interval=self._mean(),
            std_interval=self._std(),
            is_suspected=False,
        )

    def stats_at(self, now_s: float) -> PhiAccrualStats:
        """Statistics snapshot including current phi value.

        Args:
            now_s: Current time in seconds.
        """
        current_phi = self.phi(now_s)
        return PhiAccrualStats(
            heartbeats_received=self._heartbeat_count,
            current_phi=current_phi,
            mean_interval=self._mean(),
            std_interval=self._std(),
            is_suspected=current_phi >= self._threshold,
        )

    def _mean(self) -> float:
        if not self._intervals:
            return 0.0
        return sum(self._intervals) / len(self._intervals)

    def _std(self) -> float:
        if len(self._intervals) < 2:
            return 0.0
        mean = self._mean()
        variance = sum((x - mean) ** 2 for x in self._intervals) / len(self._intervals)
        return math.sqrt(variance)

    @property
    def threshold(self) -> float:
        """The configured phi threshold."""
        return self._threshold

    @property
    def last_heartbeat(self) -> float | None:
        """Timestamp of the last recorded heartbeat."""
        return self._last_heartbeat

    def __repr__(self) -> str:
        return (
            f"PhiAccrualDetector(threshold={self._threshold}, "
            f"samples={len(self._intervals)}, "
            f"heartbeats={self._heartbeat_count})"
        )
