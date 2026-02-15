"""Disk I/O contention: HDD vs SSD vs NVMe under concurrent load.

This example demonstrates how different disk profiles handle concurrent
I/O requests. The key insight: HDD performance degrades dramatically
under queue depth due to physical seek contention, while NVMe maintains
performance thanks to hardware parallelism.

## Architecture Diagram

```
    Source (constant rate)
        |
        v
    IOWorkloadDriver ──> DiskIO (HDD)  ──> Sink
    IOWorkloadDriver ──> DiskIO (SSD)  ──> Sink
    IOWorkloadDriver ──> DiskIO (NVMe) ──> Sink
```

## Key Metrics

- Average read latency per device profile
- Queue depth effects on latency
- Total IOPS achieved per profile
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from happysimulator import (
    Data,
    Entity,
    Event,
    Instant,
    Probe,
    Simulation,
    SimulationSummary,
    Sink,
    Source,
)
from happysimulator.components.infrastructure import (
    HDD,
    SSD,
    DiskIO,
    DiskIOStats,
    NVMe,
)

if TYPE_CHECKING:
    from collections.abc import Generator

# =============================================================================
# Custom Entity: IOWorkloadDriver
# =============================================================================


class IOWorkloadDriver(Entity):
    """Drives a mixed read/write workload against a DiskIO instance.

    Each incoming event triggers either a read or write operation
    on the configured disk, with configurable read/write ratio.
    """

    def __init__(
        self,
        name: str,
        *,
        disk: DiskIO,
        downstream: Entity | None = None,
        read_fraction: float = 0.7,
        io_size_bytes: int = 4096,
    ) -> None:
        super().__init__(name)
        self._disk = disk
        self._downstream = downstream
        self._read_fraction = read_fraction
        self._io_size = io_size_bytes
        self._ops_completed: int = 0

    @property
    def ops_completed(self) -> int:
        return self._ops_completed

    def handle_event(self, event: Event) -> Generator[float, None, list[Event]]:
        if random.random() < self._read_fraction:
            yield from self._disk.read(self._io_size)
        else:
            yield from self._disk.write(self._io_size)
        self._ops_completed += 1

        if self._downstream:
            return [self.forward(event, self._downstream, event_type="Completed")]
        return []


# =============================================================================
# Simulation
# =============================================================================


@dataclass
class ProfileResult:
    profile_name: str
    disk: DiskIO
    stats: DiskIOStats
    summary: SimulationSummary
    ops_data: Data


@dataclass
class SimulationResult:
    hdd: ProfileResult
    ssd: ProfileResult
    nvme: ProfileResult
    duration_s: float


def _run_profile(
    profile_name: str,
    disk: DiskIO,
    *,
    duration_s: float,
    rate: float,
) -> ProfileResult:
    sink = Sink()
    driver = IOWorkloadDriver(
        f"Driver_{profile_name}",
        disk=disk,
        downstream=sink,
    )

    source = Source.constant(
        rate=rate,
        target=driver,
        event_type="IO",
        stop_after=Instant.from_seconds(duration_s),
    )

    probe, ops_data = Probe.on(driver, "ops_completed", interval=0.5)

    sim = Simulation(
        start_time=Instant.Epoch,
        duration=duration_s + 1.0,
        sources=[source],
        entities=[disk, driver, sink],
        probes=[probe],
    )
    summary = sim.run()

    return ProfileResult(
        profile_name=profile_name,
        disk=disk,
        stats=disk.stats,
        summary=summary,
        ops_data=ops_data,
    )


def run_simulation(
    *,
    duration_s: float = 10.0,
    rate: float = 500.0,
    seed: int | None = 42,
) -> SimulationResult:
    """Compare HDD, SSD, and NVMe under concurrent I/O load."""
    if seed is not None:
        random.seed(seed)

    hdd_result = _run_profile(
        "HDD",
        DiskIO("HDD_Disk", profile=HDD()),
        duration_s=duration_s,
        rate=rate,
    )

    if seed is not None:
        random.seed(seed)

    ssd_result = _run_profile(
        "SSD",
        DiskIO("SSD_Disk", profile=SSD()),
        duration_s=duration_s,
        rate=rate,
    )

    if seed is not None:
        random.seed(seed)

    nvme_result = _run_profile(
        "NVMe",
        DiskIO("NVMe_Disk", profile=NVMe()),
        duration_s=duration_s,
        rate=rate,
    )

    return SimulationResult(
        hdd=hdd_result,
        ssd=ssd_result,
        nvme=nvme_result,
        duration_s=duration_s,
    )


# =============================================================================
# Summary
# =============================================================================


def print_summary(result: SimulationResult) -> None:
    print("\n" + "=" * 72)
    print("DISK I/O CONTENTION: HDD vs SSD vs NVMe")
    print("=" * 72)

    header = f"{'Metric':<30} {'HDD':>12} {'SSD':>12} {'NVMe':>12}"
    print(f"\n{header}")
    print("-" * len(header))

    for _label, _stats in [
        ("HDD", result.hdd.stats),
        ("SSD", result.ssd.stats),
        ("NVMe", result.nvme.stats),
    ]:
        pass  # just for reference

    h, s, n = result.hdd.stats, result.ssd.stats, result.nvme.stats
    print(f"{'Reads':<30} {h.reads:>12,} {s.reads:>12,} {n.reads:>12,}")
    print(f"{'Writes':<30} {h.writes:>12,} {s.writes:>12,} {n.writes:>12,}")
    print(f"{'Bytes read':<30} {h.bytes_read:>12,} {s.bytes_read:>12,} {n.bytes_read:>12,}")
    print(
        f"{'Bytes written':<30} {h.bytes_written:>12,} {s.bytes_written:>12,} {n.bytes_written:>12,}"
    )
    print(
        f"{'Avg read latency (ms)':<30} {h.avg_read_latency_s * 1000:>12.3f} {s.avg_read_latency_s * 1000:>12.3f} {n.avg_read_latency_s * 1000:>12.3f}"
    )
    print(
        f"{'Avg write latency (ms)':<30} {h.avg_write_latency_s * 1000:>12.3f} {s.avg_write_latency_s * 1000:>12.3f} {n.avg_write_latency_s * 1000:>12.3f}"
    )
    print(
        f"{'Peak queue depth':<30} {h.peak_queue_depth:>12} {s.peak_queue_depth:>12} {n.peak_queue_depth:>12}"
    )

    print("\n" + "=" * 72)
    print("INTERPRETATION:")
    print("-" * 72)
    print("\n  HDD latency is dominated by physical seek time (~8ms) that")
    print("  compounds under concurrent access. SSD provides uniform low")
    print("  latency (~0.025ms) with moderate queue depth scaling. NVMe")
    print("  achieves the lowest latency (~0.01ms) with minimal penalty")
    print("  up to its native queue depth (32).")
    print("\n" + "=" * 72)


# =============================================================================
# Visualization
# =============================================================================


def visualize_results(result: SimulationResult, output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping visualization")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Chart 1: Average latency comparison
    ax = axes[0]
    profiles = ["HDD", "SSD", "NVMe"]
    read_lats = [
        result.hdd.stats.avg_read_latency_s * 1000,
        result.ssd.stats.avg_read_latency_s * 1000,
        result.nvme.stats.avg_read_latency_s * 1000,
    ]
    write_lats = [
        result.hdd.stats.avg_write_latency_s * 1000,
        result.ssd.stats.avg_write_latency_s * 1000,
        result.nvme.stats.avg_write_latency_s * 1000,
    ]
    x = range(len(profiles))
    w = 0.35
    ax.bar([i - w / 2 for i in x], read_lats, w, label="Read", color="#4C72B0")
    ax.bar([i + w / 2 for i in x], write_lats, w, label="Write", color="#DD8452")
    ax.set_xticks(list(x))
    ax.set_xticklabels(profiles)
    ax.set_ylabel("Avg Latency (ms)")
    ax.set_title("I/O Latency by Profile")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")

    # Chart 2: IOPS over time
    ax = axes[1]
    for name, pr in [("HDD", result.hdd), ("SSD", result.ssd), ("NVMe", result.nvme)]:
        times = pr.ops_data.times()
        vals = pr.ops_data.raw_values()
        ax.plot(times, vals, linewidth=1.5, label=name, alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative Ops")
    ax.set_title("Operations Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Disk I/O Contention Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "disk_io_contention.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'disk_io_contention.png'}")


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Disk I/O contention comparison")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="output/disk_io_contention")
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    seed = None if args.seed == -1 else args.seed

    print("Running disk I/O contention comparison...")
    result = run_simulation(duration_s=args.duration, seed=seed)
    print_summary(result)

    if not args.no_viz:
        visualize_results(result, Path(args.output))
