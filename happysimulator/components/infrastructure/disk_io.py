"""Disk I/O model with device profiles and queue depth effects.

Models the latency characteristics of different storage devices: HDD
(seek-sensitive with rotational latency), SSD (uniform low latency),
and NVMe (high parallelism with minimal queue depth penalty). The queue
depth model captures how concurrent I/O requests affect latency.

Device profiles:
- HDD: Seek time + rotational latency; degrades significantly under
  concurrent access due to head movement.
- SSD: Uniform latency with moderate queue depth scaling.
- NVMe: Very low latency with high parallelism; queue depth has minimal
  impact up to the device's native queue depth.
"""

from __future__ import annotations

import logging
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from happysimulator.core.entity import Entity

if TYPE_CHECKING:
    from collections.abc import Generator

    from happysimulator.core.event import Event

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Disk profiles
# ---------------------------------------------------------------------------


class DiskProfile(ABC):
    """Strategy defining latency characteristics of a storage device."""

    @abstractmethod
    def read_latency_s(self, size_bytes: int, queue_depth: int) -> float:
        """Return read latency in seconds for the given size and queue depth."""
        ...

    @abstractmethod
    def write_latency_s(self, size_bytes: int, queue_depth: int) -> float:
        """Return write latency in seconds for the given size and queue depth."""
        ...


class HDD(DiskProfile):
    """Spinning disk: seek time + rotational latency + transfer.

    Args:
        seek_time_s: Average seek time (default 8ms).
        rotational_latency_s: Average rotational latency (default 4ms for 7200rpm).
        transfer_rate_mbps: Sequential transfer rate in MB/s (default 150).
        queue_depth_penalty: Multiplier per additional queued I/O (default 0.3).
    """

    def __init__(
        self,
        *,
        seek_time_s: float = 0.008,
        rotational_latency_s: float = 0.004,
        transfer_rate_mbps: float = 150.0,
        queue_depth_penalty: float = 0.3,
    ) -> None:
        self.seek_time_s = seek_time_s
        self.rotational_latency_s = rotational_latency_s
        self.transfer_rate_bytes_per_s = transfer_rate_mbps * 1_000_000
        self.queue_depth_penalty = queue_depth_penalty

    def _base_latency(self, size_bytes: int) -> float:
        transfer = size_bytes / self.transfer_rate_bytes_per_s
        # Randomize seek slightly
        seek = self.seek_time_s * (0.5 + random.random())
        return seek + self.rotational_latency_s + transfer

    def read_latency_s(self, size_bytes: int, queue_depth: int) -> float:
        base = self._base_latency(size_bytes)
        # Queue depth increases latency due to head contention
        penalty = 1.0 + self.queue_depth_penalty * max(0, queue_depth - 1)
        return base * penalty

    def write_latency_s(self, size_bytes: int, queue_depth: int) -> float:
        base = self._base_latency(size_bytes)
        penalty = 1.0 + self.queue_depth_penalty * max(0, queue_depth - 1)
        return base * penalty


class SSD(DiskProfile):
    """NAND flash: uniform low latency with moderate queue scaling.

    Args:
        base_read_latency_s: Base read latency (default 25us).
        base_write_latency_s: Base write latency (default 100us).
        transfer_rate_mbps: Transfer rate in MB/s (default 550).
        queue_depth_factor: Log-based queue depth scaling (default 0.15).
    """

    def __init__(
        self,
        *,
        base_read_latency_s: float = 0.000025,
        base_write_latency_s: float = 0.0001,
        transfer_rate_mbps: float = 550.0,
        queue_depth_factor: float = 0.15,
    ) -> None:
        self.base_read_latency_s = base_read_latency_s
        self.base_write_latency_s = base_write_latency_s
        self.transfer_rate_bytes_per_s = transfer_rate_mbps * 1_000_000
        self.queue_depth_factor = queue_depth_factor

    def read_latency_s(self, size_bytes: int, queue_depth: int) -> float:
        transfer = size_bytes / self.transfer_rate_bytes_per_s
        # Logarithmic queue depth impact
        penalty = 1.0 + self.queue_depth_factor * math.log1p(max(0, queue_depth - 1))
        return (self.base_read_latency_s + transfer) * penalty

    def write_latency_s(self, size_bytes: int, queue_depth: int) -> float:
        transfer = size_bytes / self.transfer_rate_bytes_per_s
        penalty = 1.0 + self.queue_depth_factor * math.log1p(max(0, queue_depth - 1))
        return (self.base_write_latency_s + transfer) * penalty


class NVMe(DiskProfile):
    """NVMe SSD: very low latency with high native parallelism.

    Args:
        base_read_latency_s: Base read latency (default 10us).
        base_write_latency_s: Base write latency (default 20us).
        transfer_rate_mbps: Transfer rate in MB/s (default 3500).
        native_queue_depth: Device parallelism before penalties apply (default 32).
        overflow_penalty: Penalty per I/O beyond native queue depth (default 0.05).
    """

    def __init__(
        self,
        *,
        base_read_latency_s: float = 0.00001,
        base_write_latency_s: float = 0.00002,
        transfer_rate_mbps: float = 3500.0,
        native_queue_depth: int = 32,
        overflow_penalty: float = 0.05,
    ) -> None:
        self.base_read_latency_s = base_read_latency_s
        self.base_write_latency_s = base_write_latency_s
        self.transfer_rate_bytes_per_s = transfer_rate_mbps * 1_000_000
        self.native_queue_depth = native_queue_depth
        self.overflow_penalty = overflow_penalty

    def _penalty(self, queue_depth: int) -> float:
        overflow = max(0, queue_depth - self.native_queue_depth)
        return 1.0 + self.overflow_penalty * overflow

    def read_latency_s(self, size_bytes: int, queue_depth: int) -> float:
        transfer = size_bytes / self.transfer_rate_bytes_per_s
        return (self.base_read_latency_s + transfer) * self._penalty(queue_depth)

    def write_latency_s(self, size_bytes: int, queue_depth: int) -> float:
        transfer = size_bytes / self.transfer_rate_bytes_per_s
        return (self.base_write_latency_s + transfer) * self._penalty(queue_depth)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiskIOStats:
    """Frozen snapshot of DiskIO statistics.

    Attributes:
        reads: Total read operations completed.
        writes: Total write operations completed.
        bytes_read: Total bytes read.
        bytes_written: Total bytes written.
        total_read_latency_s: Cumulative read latency.
        total_write_latency_s: Cumulative write latency.
        current_queue_depth: Current number of in-flight I/O operations.
        peak_queue_depth: Maximum concurrent I/O observed.
    """

    reads: int = 0
    writes: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    total_read_latency_s: float = 0.0
    total_write_latency_s: float = 0.0
    current_queue_depth: int = 0
    peak_queue_depth: int = 0

    @property
    def avg_read_latency_s(self) -> float:
        return self.total_read_latency_s / self.reads if self.reads else 0.0

    @property
    def avg_write_latency_s(self) -> float:
        return self.total_write_latency_s / self.writes if self.writes else 0.0


# ---------------------------------------------------------------------------
# DiskIO entity
# ---------------------------------------------------------------------------


class DiskIO(Entity):
    """Disk I/O model with device profiles and queue depth effects.

    Provides ``read()`` and ``write()`` generator methods that yield
    latency based on the configured disk profile and current queue depth.

    Args:
        name: Entity name.
        profile: Disk profile (HDD, SSD, NVMe). Defaults to SSD.

    Example::

        disk = DiskIO("ssd", profile=SSD())
        sim = Simulation(entities=[disk, ...], ...)

        # In another entity's handle_event:
        yield from disk.read(4096)
        yield from disk.write(8192)
    """

    def __init__(self, name: str, *, profile: DiskProfile | None = None) -> None:
        super().__init__(name)
        self._profile = profile or SSD()
        self._queue_depth: int = 0

        # Stats
        self._reads: int = 0
        self._writes: int = 0
        self._bytes_read: int = 0
        self._bytes_written: int = 0
        self._total_read_latency_s: float = 0.0
        self._total_write_latency_s: float = 0.0
        self._peak_queue_depth: int = 0

    @property
    def queue_depth(self) -> int:
        """Current number of in-flight I/O operations."""
        return self._queue_depth

    @property
    def stats(self) -> DiskIOStats:
        """Frozen snapshot of disk I/O statistics."""
        return DiskIOStats(
            reads=self._reads,
            writes=self._writes,
            bytes_read=self._bytes_read,
            bytes_written=self._bytes_written,
            total_read_latency_s=self._total_read_latency_s,
            total_write_latency_s=self._total_write_latency_s,
            current_queue_depth=self._queue_depth,
            peak_queue_depth=self._peak_queue_depth,
        )

    def read(self, size_bytes: int = 4096) -> Generator[float]:
        """Read from disk, yielding I/O latency.

        Args:
            size_bytes: Number of bytes to read.
        """
        self._queue_depth += 1
        if self._queue_depth > self._peak_queue_depth:
            self._peak_queue_depth = self._queue_depth

        latency = self._profile.read_latency_s(size_bytes, self._queue_depth)
        try:
            yield latency
        finally:
            self._queue_depth -= 1
            self._reads += 1
            self._bytes_read += size_bytes
            self._total_read_latency_s += latency

    def write(self, size_bytes: int = 4096) -> Generator[float]:
        """Write to disk, yielding I/O latency.

        Args:
            size_bytes: Number of bytes to write.
        """
        self._queue_depth += 1
        if self._queue_depth > self._peak_queue_depth:
            self._peak_queue_depth = self._queue_depth

        latency = self._profile.write_latency_s(size_bytes, self._queue_depth)
        try:
            yield latency
        finally:
            self._queue_depth -= 1
            self._writes += 1
            self._bytes_written += size_bytes
            self._total_write_latency_s += latency

    def handle_event(self, event: Event) -> None:
        """DiskIO does not process events directly."""

    def __repr__(self) -> str:
        profile_name = type(self._profile).__name__
        return f"DiskIO('{self.name}', profile={profile_name}, qd={self._queue_depth})"
