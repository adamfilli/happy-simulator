"""LSM compaction strategy comparison: Size-Tiered vs Leveled.

This example demonstrates the impact of compaction strategy on LSM tree
performance metrics. By running the same write-heavy workload through two
LSM trees with different compaction strategies, we observe:

1. SizeTieredCompaction groups similarly-sized SSTables, yielding lower
   write amplification but higher space amplification and read cost.
2. LeveledCompaction maintains sorted runs per level, yielding lower
   read amplification and space amplification at the cost of more
   frequent compactions and higher write amplification.

## Architecture Diagram

```
            Source (constant rate)
                |
                v
    +-----------+-----------+
    |                       |
    v                       v
LSMTree (SizeTiered)   LSMTree (Leveled)
 - memtable_size=64    - memtable_size=64
 - min_sstables=4      - level_0_max=4
    |                       |
    v                       v
  Sink                    Sink
```

## Key Metrics

- Write amplification: bytes written to SSTables / user bytes written
- Read amplification: SSTables checked per read operation
- Space amplification: total stored bytes / logical data bytes
- Compaction count: how many compaction cycles were triggered
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
from happysimulator.components.storage import (
    LeveledCompaction,
    LSMTree,
    LSMTreeStats,
    SizeTieredCompaction,
)

if TYPE_CHECKING:
    from collections.abc import Generator

# =============================================================================
# Custom Entity: StorageWorkloadDriver
# =============================================================================


class StorageWorkloadDriver(Entity):
    """Drives a write-then-read workload against an LSMTree.

    Receives events from a Source and performs put/get operations on the
    configured LSM tree. The workload has three phases:

    Phase 1 (writes): Insert keys sequentially.
    Phase 2 (reads): Read back all previously written keys.
    Phase 3 (mixed): Interleave writes and reads at a 50/50 ratio.

    Yields:
        I/O latencies from the underlying LSM tree operations.
    """

    def __init__(
        self,
        name: str,
        *,
        lsm: LSMTree,
        downstream: Entity | None = None,
        total_keys: int = 2000,
        write_fraction: float = 0.5,
    ) -> None:
        super().__init__(name)
        self._lsm = lsm
        self._downstream = downstream
        self._total_keys = total_keys
        self._write_fraction = write_fraction
        self._next_key: int = 0
        self._phase: str = "write"
        self._read_index: int = 0
        self._ops_completed: int = 0

    @property
    def ops_completed(self) -> int:
        """Total operations driven so far."""
        return self._ops_completed

    def handle_event(self, event: Event) -> Generator[float, None, list[Event]]:
        """Drive one operation per incoming event."""
        if self._phase == "write":
            # Phase 1: sequential writes
            key = f"key_{self._next_key:06d}"
            value = f"value_{self._next_key}"
            self._lsm.put_sync(key, value)
            self._next_key += 1
            self._ops_completed += 1
            if self._next_key >= self._total_keys:
                self._phase = "read"
                self._read_index = 0
        elif self._phase == "read":
            # Phase 2: sequential reads of previously written keys
            key = f"key_{self._read_index:06d}"
            self._lsm.get_sync(key)
            self._read_index += 1
            self._ops_completed += 1
            if self._read_index >= self._next_key:
                self._phase = "mixed"
        else:
            # Phase 3: mixed read/write
            if random.random() < self._write_fraction:
                key = f"key_{self._next_key:06d}"
                value = f"value_{self._next_key}"
                self._lsm.put_sync(key, value)
                self._next_key += 1
            else:
                read_key_idx = random.randint(0, self._next_key - 1)
                key = f"key_{read_key_idx:06d}"
                self._lsm.get_sync(key)
            self._ops_completed += 1

        if self._downstream is not None:
            return [self.forward(event, self._downstream, event_type="Completed")]
        return []


# =============================================================================
# Simulation
# =============================================================================


@dataclass
class CompactionResult:
    """Results from a single compaction strategy run."""

    strategy_name: str
    lsm: LSMTree
    stats: LSMTreeStats
    level_summary: list[dict]
    ops_data: Data
    summary: SimulationSummary


@dataclass
class SimulationResult:
    """Combined results comparing both compaction strategies."""

    size_tiered: CompactionResult
    leveled: CompactionResult
    total_keys: int
    duration_s: float


def _run_single_strategy(
    strategy_name: str,
    lsm: LSMTree,
    *,
    duration_s: float,
    rate: float,
    total_keys: int,
    probe_interval_s: float,
) -> CompactionResult:
    """Run one LSM workload with the given compaction strategy."""
    sink = Sink()
    driver = StorageWorkloadDriver(
        name=f"Driver_{strategy_name}",
        lsm=lsm,
        downstream=sink,
        total_keys=total_keys,
    )

    source = Source.constant(
        rate=rate,
        target=driver,
        event_type="Op",
        stop_after=Instant.from_seconds(duration_s),
    )

    ops_probe, ops_data = Probe.on(driver, "ops_completed", interval=probe_interval_s)

    sim = Simulation(
        start_time=Instant.Epoch,
        duration=duration_s + 1.0,
        sources=[source],
        entities=[lsm, driver, sink],
        probes=[ops_probe],
    )
    summary = sim.run()

    stats = lsm.stats
    level_summary = lsm.level_summary

    return CompactionResult(
        strategy_name=strategy_name,
        lsm=lsm,
        stats=stats,
        level_summary=level_summary,
        ops_data=ops_data,
        summary=summary,
    )


def run_compaction_simulation(
    *,
    duration_s: float = 30.0,
    rate: float = 500.0,
    total_keys: int = 2000,
    memtable_size: int = 64,
    probe_interval_s: float = 0.5,
    seed: int | None = 42,
) -> SimulationResult:
    """Run the compaction comparison simulation.

    Runs the same workload against an LSM tree with SizeTieredCompaction
    and another with LeveledCompaction, collecting amplification metrics.

    Args:
        duration_s: How long to run each simulation.
        rate: Operations per second driven by the source.
        total_keys: Total unique keys written in the write phase.
        memtable_size: Memtable capacity (smaller = more flushes).
        probe_interval_s: Metric sampling interval.
        seed: Random seed for reproducibility.

    Returns:
        SimulationResult with metrics from both strategies.
    """
    # --- Size-Tiered ---
    if seed is not None:
        random.seed(seed)

    lsm_st = LSMTree(
        "LSM_SizeTiered",
        memtable_size=memtable_size,
        compaction_strategy=SizeTieredCompaction(min_sstables=4),
    )
    size_tiered = _run_single_strategy(
        "SizeTiered",
        lsm_st,
        duration_s=duration_s,
        rate=rate,
        total_keys=total_keys,
        probe_interval_s=probe_interval_s,
    )

    # --- Leveled ---
    if seed is not None:
        random.seed(seed)

    lsm_lc = LSMTree(
        "LSM_Leveled",
        memtable_size=memtable_size,
        compaction_strategy=LeveledCompaction(level_0_max=4, base_size_keys=256),
    )
    leveled = _run_single_strategy(
        "Leveled",
        lsm_lc,
        duration_s=duration_s,
        rate=rate,
        total_keys=total_keys,
        probe_interval_s=probe_interval_s,
    )

    return SimulationResult(
        size_tiered=size_tiered,
        leveled=leveled,
        total_keys=total_keys,
        duration_s=duration_s,
    )


# =============================================================================
# Summary
# =============================================================================


def print_summary(result: SimulationResult) -> None:
    """Print comparison of compaction strategies."""
    print("\n" + "=" * 72)
    print("LSM COMPACTION STRATEGY COMPARISON")
    print("=" * 72)

    print(f"\nWorkload: {result.total_keys} keys, {result.duration_s}s duration")
    print()

    st = result.size_tiered.stats
    lv = result.leveled.stats

    header = f"{'Metric':<30} {'SizeTiered':>15} {'Leveled':>15}"
    print(header)
    print("-" * len(header))
    print(f"{'Total writes':<30} {st.writes:>15,} {lv.writes:>15,}")
    print(f"{'Total reads':<30} {st.reads:>15,} {lv.reads:>15,}")
    print(f"{'Memtable flushes':<30} {st.memtable_flushes:>15,} {lv.memtable_flushes:>15,}")
    print(f"{'Compactions':<30} {st.compactions:>15,} {lv.compactions:>15,}")
    print(f"{'Total SSTables':<30} {st.total_sstables:>15,} {lv.total_sstables:>15,}")
    print(f"{'Occupied levels':<30} {st.levels:>15,} {lv.levels:>15,}")
    print(f"{'Bloom filter saves':<30} {st.bloom_filter_saves:>15,} {lv.bloom_filter_saves:>15,}")
    print(
        f"{'Read amplification':<30} {st.read_amplification:>15.2f} {lv.read_amplification:>15.2f}"
    )
    print(
        f"{'Write amplification':<30} {st.write_amplification:>15.2f} {lv.write_amplification:>15.2f}"
    )
    print(
        f"{'Space amplification':<30} {st.space_amplification:>15.2f} {lv.space_amplification:>15.2f}"
    )

    # Level summaries
    for cr in [result.size_tiered, result.leveled]:
        print(f"\n  {cr.strategy_name} Level Summary:")
        if cr.level_summary:
            for level in cr.level_summary:
                print(
                    f"    L{level['level']}: {level['sstables']} SSTables, "
                    f"{level['total_keys']:,} keys, {level['total_bytes']:,} bytes"
                )
        else:
            print("    (empty)")

    print("\n" + "=" * 72)
    print("INTERPRETATION:")
    print("-" * 72)

    if st.write_amplification < lv.write_amplification:
        print(
            f"\n  SizeTiered has LOWER write amplification "
            f"({st.write_amplification:.2f}x vs {lv.write_amplification:.2f}x)."
        )
        print("  This makes it better for write-heavy workloads.")
    else:
        print(
            f"\n  Leveled has LOWER write amplification "
            f"({lv.write_amplification:.2f}x vs {st.write_amplification:.2f}x)."
        )

    if st.read_amplification > lv.read_amplification:
        print(
            f"\n  Leveled has LOWER read amplification "
            f"({lv.read_amplification:.2f}x vs {st.read_amplification:.2f}x)."
        )
        print("  This makes it better for read-heavy workloads.")
    else:
        print(
            f"\n  SizeTiered has LOWER read amplification "
            f"({st.read_amplification:.2f}x vs {lv.read_amplification:.2f}x)."
        )

    if lv.space_amplification < st.space_amplification:
        print(
            f"\n  Leveled has LOWER space amplification "
            f"({lv.space_amplification:.2f}x vs {st.space_amplification:.2f}x)."
        )
        print("  Leveled compaction is more space-efficient on disk.")

    print("\n" + "=" * 72)


# =============================================================================
# Visualization
# =============================================================================


def visualize_results(result: SimulationResult, output_dir: Path) -> None:
    """Generate comparison charts for the two compaction strategies."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping visualization")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    st = result.size_tiered.stats
    lv = result.leveled.stats

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Chart 1: Amplification comparison (grouped bar) ---
    ax = axes[0, 0]
    metrics = ["Read\nAmplification", "Write\nAmplification", "Space\nAmplification"]
    st_values = [st.read_amplification, st.write_amplification, st.space_amplification]
    lv_values = [lv.read_amplification, lv.write_amplification, lv.space_amplification]

    x = range(len(metrics))
    width = 0.35
    bars1 = ax.bar(
        [i - width / 2 for i in x],
        st_values,
        width,
        label="SizeTiered",
        color="#4C72B0",
        edgecolor="black",
        alpha=0.85,
    )
    bars2 = ax.bar(
        [i + width / 2 for i in x],
        lv_values,
        width,
        label="Leveled",
        color="#DD8452",
        edgecolor="black",
        alpha=0.85,
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Amplification Factor")
    ax.set_title("Amplification Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{bar.get_height():.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{bar.get_height():.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # --- Chart 2: Compaction and flush counts ---
    ax = axes[0, 1]
    count_metrics = ["Compactions", "Memtable\nFlushes", "Total\nSSTables", "Bloom\nSaves"]
    st_counts = [st.compactions, st.memtable_flushes, st.total_sstables, st.bloom_filter_saves]
    lv_counts = [lv.compactions, lv.memtable_flushes, lv.total_sstables, lv.bloom_filter_saves]

    x2 = range(len(count_metrics))
    ax.bar(
        [i - width / 2 for i in x2],
        st_counts,
        width,
        label="SizeTiered",
        color="#4C72B0",
        edgecolor="black",
        alpha=0.85,
    )
    ax.bar(
        [i + width / 2 for i in x2],
        lv_counts,
        width,
        label="Leveled",
        color="#DD8452",
        edgecolor="black",
        alpha=0.85,
    )
    ax.set_xticks(list(x2))
    ax.set_xticklabels(count_metrics)
    ax.set_ylabel("Count")
    ax.set_title("Operation Counts")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # --- Chart 3: Operations throughput over time ---
    ax = axes[1, 0]
    st_ops = result.size_tiered.ops_data
    lv_ops = result.leveled.ops_data

    st_times = st_ops.times()
    st_vals = st_ops.raw_values()
    lv_times = lv_ops.times()
    lv_vals = lv_ops.raw_values()

    ax.plot(st_times, st_vals, "b-", linewidth=1.5, label="SizeTiered", alpha=0.8)
    ax.plot(lv_times, lv_vals, "r-", linewidth=1.5, label="Leveled", alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative Operations")
    ax.set_title("Operations Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Chart 4: Level distribution (stacked bar) ---
    ax = axes[1, 1]
    strategies = ["SizeTiered", "Leveled"]
    max_levels = (
        max(
            max((d["level"] for d in result.size_tiered.level_summary), default=0),
            max((d["level"] for d in result.leveled.level_summary), default=0),
        )
        + 1
    )

    level_keys_st = [0] * max_levels
    level_keys_lv = [0] * max_levels
    for d in result.size_tiered.level_summary:
        if d["level"] < max_levels:
            level_keys_st[d["level"]] = d["total_keys"]
    for d in result.leveled.level_summary:
        if d["level"] < max_levels:
            level_keys_lv[d["level"]] = d["total_keys"]

    x3 = range(len(strategies))
    colors = plt.cm.viridis([i / max(max_levels - 1, 1) for i in range(max_levels)])

    for level_idx in range(max_levels):
        vals = [level_keys_st[level_idx], level_keys_lv[level_idx]]
        bottoms = [sum(level_keys_st[:level_idx]), sum(level_keys_lv[:level_idx])]
        ax.bar(
            list(x3),
            vals,
            bottom=bottoms,
            width=0.5,
            label=f"L{level_idx}",
            color=colors[level_idx],
            edgecolor="black",
        )

    ax.set_xticks(list(x3))
    ax.set_xticklabels(strategies)
    ax.set_ylabel("Total Keys")
    ax.set_title("Key Distribution Across Levels")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("LSM Compaction Strategy Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "lsm_compaction_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'lsm_compaction_comparison.png'}")


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LSM compaction strategy comparison simulation")
    parser.add_argument(
        "--duration", type=float, default=30.0, help="Simulation duration in seconds"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (-1 for random)")
    parser.add_argument(
        "--output",
        type=str,
        default="output/lsm_compaction",
        help="Output directory for visualizations",
    )
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization generation")
    args = parser.parse_args()

    seed = None if args.seed == -1 else args.seed

    print("Running LSM compaction strategy comparison...")
    print(f"  Duration: {args.duration}s")
    print(f"  Seed: {seed if seed is not None else 'random'}")

    result = run_compaction_simulation(
        duration_s=args.duration,
        seed=seed,
    )

    print_summary(result)

    if not args.no_viz:
        output_dir = Path(args.output)
        visualize_results(result, output_dir)
        print(f"\nVisualizations saved to: {output_dir.absolute()}")
