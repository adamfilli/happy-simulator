"""Unit tests for DiskIO."""

import contextlib
import random

from happysimulator.components.infrastructure.disk_io import (
    HDD,
    SSD,
    DiskIO,
    DiskIOStats,
    NVMe,
)
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant


class TestDiskProfiles:
    def test_hdd_read_latency_positive(self):
        random.seed(42)
        hdd = HDD()
        lat = hdd.read_latency_s(4096, queue_depth=1)
        assert lat > 0

    def test_hdd_write_latency_positive(self):
        random.seed(42)
        hdd = HDD()
        lat = hdd.write_latency_s(4096, queue_depth=1)
        assert lat > 0

    def test_hdd_queue_depth_penalty(self):
        random.seed(42)
        hdd = HDD()
        lat_1 = hdd.read_latency_s(4096, queue_depth=1)
        random.seed(42)
        lat_8 = hdd.read_latency_s(4096, queue_depth=8)
        assert lat_8 > lat_1

    def test_ssd_read_latency_lower_than_hdd(self):
        random.seed(42)
        hdd = HDD()
        ssd = SSD()
        hdd_lat = hdd.read_latency_s(4096, queue_depth=1)
        ssd_lat = ssd.read_latency_s(4096, queue_depth=1)
        assert ssd_lat < hdd_lat

    def test_ssd_queue_depth_scaling(self):
        ssd = SSD()
        lat_1 = ssd.read_latency_s(4096, queue_depth=1)
        lat_32 = ssd.read_latency_s(4096, queue_depth=32)
        assert lat_32 > lat_1

    def test_nvme_lowest_latency(self):
        nvme = NVMe()
        ssd = SSD()
        assert nvme.read_latency_s(4096, 1) < ssd.read_latency_s(4096, 1)

    def test_nvme_within_native_queue_depth(self):
        nvme = NVMe(native_queue_depth=32)
        lat_1 = nvme.read_latency_s(4096, queue_depth=1)
        lat_16 = nvme.read_latency_s(4096, queue_depth=16)
        # Within native queue depth: same latency
        assert lat_16 == lat_1

    def test_nvme_overflow_penalty(self):
        nvme = NVMe(native_queue_depth=32)
        lat_32 = nvme.read_latency_s(4096, queue_depth=32)
        lat_64 = nvme.read_latency_s(4096, queue_depth=64)
        assert lat_64 > lat_32

    def test_larger_io_takes_longer(self):
        ssd = SSD()
        small = ssd.read_latency_s(4096, queue_depth=1)
        large = ssd.read_latency_s(1_000_000, queue_depth=1)
        assert large > small


class TestDiskIO:
    def _make_disk(self, **kwargs) -> tuple[DiskIO, Simulation]:
        disk = DiskIO("test_disk", **kwargs)
        sim = Simulation(
            start_time=Instant.from_seconds(0),
            end_time=Instant.from_seconds(100),
            entities=[disk],
        )
        return disk, sim

    def test_creation_defaults(self):
        disk = DiskIO("disk")
        assert disk.name == "disk"
        assert disk.queue_depth == 0

    def test_creation_with_profile(self):
        disk = DiskIO("disk", profile=HDD())
        assert "HDD" in repr(disk)

    def test_stats_initial(self):
        disk, _sim = self._make_disk()
        stats = disk.stats
        assert isinstance(stats, DiskIOStats)
        assert stats.reads == 0
        assert stats.writes == 0
        assert stats.bytes_read == 0
        assert stats.bytes_written == 0
        assert stats.current_queue_depth == 0
        assert stats.peak_queue_depth == 0

    def test_stats_avg_latency_zero_when_no_ops(self):
        disk, _sim = self._make_disk()
        assert disk.stats.avg_read_latency_s == 0.0
        assert disk.stats.avg_write_latency_s == 0.0

    def test_read_generator(self):
        disk, _sim = self._make_disk()
        gen = disk.read(4096)
        latency = next(gen)
        assert latency > 0
        # Complete the generator
        with contextlib.suppress(StopIteration):
            next(gen)
        assert disk.stats.reads == 1
        assert disk.stats.bytes_read == 4096

    def test_write_generator(self):
        disk, _sim = self._make_disk()
        gen = disk.write(8192)
        latency = next(gen)
        assert latency > 0
        with contextlib.suppress(StopIteration):
            next(gen)
        assert disk.stats.writes == 1
        assert disk.stats.bytes_written == 8192

    def test_peak_queue_depth_tracked(self):
        disk, _sim = self._make_disk()
        # Start two reads concurrently
        g1 = disk.read(4096)
        g2 = disk.read(4096)
        next(g1)  # enter g1
        next(g2)  # enter g2 — queue depth is now 2
        assert disk.queue_depth == 2
        assert disk.stats.peak_queue_depth == 2

    def test_queue_depth_decreases_after_completion(self):
        disk, _sim = self._make_disk()
        gen = disk.read(4096)
        next(gen)
        assert disk.queue_depth == 1
        with contextlib.suppress(StopIteration):
            next(gen)
        assert disk.queue_depth == 0

    def test_handle_event_is_noop(self):
        disk, _sim = self._make_disk()
        from happysimulator.core.event import Event

        event = Event(
            time=Instant.from_seconds(1),
            event_type="Test",
            target=disk,
        )
        result = disk.handle_event(event)
        assert result is None

    def test_repr(self):
        disk, _sim = self._make_disk()
        assert "test_disk" in repr(disk)
        assert "SSD" in repr(disk)  # default profile

    def test_hdd_profile_in_repr(self):
        disk, _sim = self._make_disk(profile=HDD())
        assert "HDD" in repr(disk)

    def test_nvme_profile_in_repr(self):
        disk, _sim = self._make_disk(profile=NVMe())
        assert "NVMe" in repr(disk)
