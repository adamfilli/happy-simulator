"""B-tree vs LSM tree: storage engine comparison under identical workloads.

This example runs the same three-phase workload (write-only, read-only,
mixed read/write) against both a B-tree and an LSM tree, collecting
page-level I/O metrics to illustrate the fundamental tradeoffs:

- B-tree: in-place updates with O(depth) page reads per lookup and
  O(depth) page reads + O(1) page writes per insert. Better for
  read-heavy and point-lookup workloads.
- LSM tree: append-only writes with buffered memtable and background
  compaction. Better for write-heavy workloads due to sequential I/O.

## Architecture Diagram

```
         Source (constant rate, 3 phases)
                |
          StorageWorkloadDriver
         /                     \\
   BTree("btree")         LSMTree("lsm")
         \\                     /
          Sink (tracks count)
```

## Phases

Phase 1 - Write:  Insert N unique keys sequentially.
Phase 2 - Read:   Read back all N keys (random order).
Phase 3 - Mixed:  50/50 random reads and new writes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

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
    BTree,
    BTreeStats,
    LSMTree,
    LSMTreeStats,
    SizeTieredCompaction,
)

# =============================================================================
# Custom Entity: DualEngineDriver
# =============================================================================


class DualEngineDriver(Entity):
    """Drives an identical workload against a B-tree and an LSM tree.

    Each incoming event triggers one operation on both engines so that
    both see exactly the same key sequence and access pattern.

    Properties exposed for probing:
    - phase_number: Current workload phase (1=write, 2=read, 3=mixed).
    - ops_completed: Total operations driven.
    """

    def __init__(
        self,
        name: str,
        *,
        btree: BTree,
        lsm: LSMTree,
        downstream: Entity | None = None,
        num_keys: int = 2000,
        write_fraction: float = 0.5,
    ) -> None:
        super().__init__(name)
        self._btree = btree
        self._lsm = lsm
        self._downstream = downstream
        self._num_keys = num_keys
        self._write_fraction = write_fraction

        self._next_write_key: int = 0
        self._read_order: list[int] = []
        self._read_index: int = 0
        self._phase: int = 1  # 1=write, 2=read, 3=mixed
        self._ops_completed: int = 0

    @property
    def phase_number(self) -> int:
        """Current workload phase."""
        return self._phase

    @property
    def ops_completed(self) -> int:
        """Total operations completed so far."""
        return self._ops_completed

    def handle_event(self, event: Event) -> list[Event]:
        """Drive one operation on both engines per incoming event."""
        if self._phase == 1:
            self._do_write_phase()
        elif self._phase == 2:
            self._do_read_phase()
        else:
            self._do_mixed_phase()

        self._ops_completed += 1

        if self._downstream is not None:
            return [self.forward(event, self._downstream, event_type="Completed")]
        return []

    def _do_write_phase(self) -> None:
        """Phase 1: sequential writes to both engines."""
        key = f"key_{self._next_write_key:06d}"
        value = f"val_{self._next_write_key}"
        self._btree.put_sync(key, value)
        self._lsm.put_sync(key, value)
        self._next_write_key += 1
        if self._next_write_key >= self._num_keys:
            # Prepare random read order for phase 2
            self._read_order = list(range(self._num_keys))
            random.shuffle(self._read_order)
            self._read_index = 0
            self._phase = 2

    def _do_read_phase(self) -> None:
        """Phase 2: random-order reads from both engines."""
        idx = self._read_order[self._read_index]
        key = f"key_{idx:06d}"
        self._btree.get_sync(key)
        self._lsm.get_sync(key)
        self._read_index += 1
        if self._read_index >= len(self._read_order):
            self._phase = 3

    def _do_mixed_phase(self) -> None:
        """Phase 3: random 50/50 read/write mix."""
        if random.random() < self._write_fraction:
            key = f"key_{self._next_write_key:06d}"
            value = f"val_{self._next_write_key}"
            self._btree.put_sync(key, value)
            self._lsm.put_sync(key, value)
            self._next_write_key += 1
        else:
            max_key = max(self._next_write_key - 1, 0)
            idx = random.randint(0, max_key)
            key = f"key_{idx:06d}"
            self._btree.get_sync(key)
            self._lsm.get_sync(key)


# =============================================================================
# Simulation
# =============================================================================


@dataclass
class SimulationResult:
    """Results from the B-tree vs LSM comparison."""

    btree: BTree
    btree_stats: BTreeStats
    lsm: LSMTree
    lsm_stats: LSMTreeStats
    phase_data: Data
    num_keys: int
    duration_s: float
    summary: SimulationSummary


def run_btree_vs_lsm_simulation(
    *,
    duration_s: float = 30.0,
    rate: float = 500.0,
    num_keys: int = 2000,
    memtable_size: int = 128,
    btree_order: int = 64,
    probe_interval_s: float = 0.5,
    seed: int | None = 42,
) -> SimulationResult:
    """Run the B-tree vs LSM comparison simulation.

    Both engines process the same key operations driven by a single
    DualEngineDriver entity. This ensures a fair comparison since
    both engines see identical workload phases.

    Args:
        duration_s: Simulation duration in seconds.
        rate: Operations per second.
        num_keys: Number of unique keys for phases 1 and 2.
        memtable_size: LSM memtable capacity (entries before flush).
        btree_order: B-tree branching factor.
        probe_interval_s: Metric sampling interval.
        seed: Random seed for reproducibility.

    Returns:
        SimulationResult with stats from both engines.
    """
    if seed is not None:
        random.seed(seed)

    # Create storage engines
    btree = BTree("btree", order=btree_order)
    lsm = LSMTree(
        "lsm",
        memtable_size=memtable_size,
        compaction_strategy=SizeTieredCompaction(min_sstables=4),
    )

    # Create driver and downstream sink
    sink = Sink()
    driver = DualEngineDriver(
        name="Driver",
        btree=btree,
        lsm=lsm,
        downstream=sink,
        num_keys=num_keys,
    )

    # Source drives operations at a constant rate
    source = Source.constant(
        rate=rate,
        target=driver,
        event_type="Op",
        stop_after=Instant.from_seconds(duration_s),
    )

    # Probe tracks which phase we are in

    phase_probe, phase_data = Probe.on(driver, "phase_number", interval=probe_interval_s)

    sim = Simulation(
        start_time=Instant.Epoch,
        duration=duration_s + 1.0,
        sources=[source],
        entities=[btree, lsm, driver, sink],
        probes=[phase_probe],
    )
    summary = sim.run()

    return SimulationResult(
        btree=btree,
        btree_stats=btree.stats,
        lsm=lsm,
        lsm_stats=lsm.stats,
        phase_data=phase_data,
        num_keys=num_keys,
        duration_s=duration_s,
        summary=summary,
    )


# =============================================================================
# Summary
# =============================================================================


def print_summary(result: SimulationResult) -> None:
    """Print a comparison table of B-tree vs LSM statistics."""
    print("\n" + "=" * 72)
    print("B-TREE vs LSM TREE COMPARISON")
    print("=" * 72)

    print(
        f"\nWorkload: {result.num_keys} keys, {result.duration_s}s, phases: write -> read -> mixed"
    )

    bt = result.btree_stats
    ls = result.lsm_stats

    header = f"{'Metric':<30} {'B-Tree':>15} {'LSM Tree':>15}"
    print(f"\n{header}")
    print("-" * len(header))

    # Writes
    print(f"{'Total writes':<30} {bt.writes:>15,} {ls.writes:>15,}")
    print(f"{'Total reads':<30} {bt.reads:>15,} {ls.reads:>15,}")

    # Page I/O
    print(f"{'Page reads':<30} {bt.page_reads:>15,} {'(N/A)':>15}")
    print(f"{'Page writes':<30} {bt.page_writes:>15,} {'(N/A)':>15}")
    print(f"{'Node splits':<30} {bt.node_splits:>15,} {'(N/A)':>15}")
    print(f"{'Tree depth':<30} {bt.tree_depth:>15} {'(N/A)':>15}")
    print(f"{'Total keys':<30} {bt.total_keys:>15,} {'(N/A)':>15}")

    # LSM-specific
    print(f"{'Memtable flushes':<30} {'(N/A)':>15} {ls.memtable_flushes:>15,}")
    print(f"{'Compactions':<30} {'(N/A)':>15} {ls.compactions:>15,}")
    print(f"{'Total SSTables':<30} {'(N/A)':>15} {ls.total_sstables:>15,}")
    print(f"{'Bloom filter saves':<30} {'(N/A)':>15} {ls.bloom_filter_saves:>15,}")
    print(f"{'Read amplification':<30} {'(N/A)':>15} {ls.read_amplification:>15.2f}")
    print(f"{'Write amplification':<30} {'(N/A)':>15} {ls.write_amplification:>15.2f}")
    print(f"{'Space amplification':<30} {'(N/A)':>15} {ls.space_amplification:>15.2f}")

    # Derived: page I/O per operation for B-tree
    total_bt_ops = bt.reads + bt.writes
    if total_bt_ops > 0:
        print("\nB-Tree Efficiency:")
        print(f"  Page reads / operation:  {bt.page_reads / total_bt_ops:.2f}")
        print(f"  Page writes / operation: {bt.page_writes / total_bt_ops:.2f}")
        print(f"  Page reads / read:       {bt.page_reads / bt.reads:.2f}" if bt.reads > 0 else "")
        if bt.writes > 0:
            print(f"  Page writes / write:     {bt.page_writes / bt.writes:.2f}")

    # LSM level summary
    print("\nLSM Level Summary:")
    for level in result.lsm.level_summary:
        print(f"  L{level['level']}: {level['sstables']} SSTables, {level['total_keys']:,} keys")

    print("\n" + "=" * 72)
    print("INTERPRETATION:")
    print("-" * 72)
    print("\n  B-Tree: every read/write traverses the tree (depth page reads).")
    print("  Writes also incur page writes, plus extra writes on node splits.")
    print(f"  Current tree depth = {bt.tree_depth}, so each point lookup costs")
    print(f"  {bt.tree_depth} page reads.")
    print()
    print("  LSM Tree: writes go to an in-memory memtable (fast), then flush")
    print("  to SSTables on disk. Reads may check multiple SSTables (read amp).")
    print(f"  Read amplification = {ls.read_amplification:.2f} SSTables/read.")
    print(f"  Write amplification = {ls.write_amplification:.2f}x (compaction overhead).")
    print("\n" + "=" * 72)


# =============================================================================
# Visualization
# =============================================================================


def visualize_results(result: SimulationResult, output_dir: Path) -> None:
    """Generate grouped bar charts comparing B-tree and LSM page I/O."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping visualization")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    bt = result.btree_stats
    ls = result.lsm_stats

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Chart 1: Page reads comparison ---
    ax = axes[0, 0]
    # For LSM, use sstables_checked as a proxy for page reads
    categories = ["Page Reads\n(B-Tree)", "SSTables Checked\n(LSM)"]
    values = [bt.page_reads, ls.reads * ls.read_amplification if ls.reads > 0 else 0]
    colors = ["#4C72B0", "#DD8452"]
    bars = ax.bar(categories, values, color=colors, edgecolor="black", alpha=0.85, width=0.5)
    for bar, val in zip(bars, values, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            f"{val:,.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylabel("Count")
    ax.set_title("Read I/O Cost")
    ax.grid(True, alpha=0.3, axis="y")

    # --- Chart 2: Page writes comparison ---
    ax = axes[0, 1]
    categories = ["Page Writes\n(B-Tree)", "SSTable Bytes Written\n/ 64 (LSM)"]
    # Normalize LSM writes to approximate page-equivalent
    lsm_page_equiv = ls.writes * ls.write_amplification if ls.writes > 0 else 0
    values = [bt.page_writes, lsm_page_equiv]
    bars = ax.bar(categories, values, color=colors, edgecolor="black", alpha=0.85, width=0.5)
    for bar, val in zip(bars, values, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            f"{val:,.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylabel("Count")
    ax.set_title("Write I/O Cost")
    ax.grid(True, alpha=0.3, axis="y")

    # --- Chart 3: Grouped bar of all comparable metrics ---
    ax = axes[1, 0]
    metric_names = ["Reads", "Writes", "Node Splits\n/ Compactions"]
    bt_vals = [bt.reads, bt.writes, bt.node_splits]
    ls_vals = [ls.reads, ls.writes, ls.compactions]

    x = range(len(metric_names))
    width = 0.35
    ax.bar(
        [i - width / 2 for i in x],
        bt_vals,
        width,
        label="B-Tree",
        color="#4C72B0",
        edgecolor="black",
        alpha=0.85,
    )
    ax.bar(
        [i + width / 2 for i in x],
        ls_vals,
        width,
        label="LSM Tree",
        color="#DD8452",
        edgecolor="black",
        alpha=0.85,
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(metric_names)
    ax.set_ylabel("Count")
    ax.set_title("Operation Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # --- Chart 4: Workload phase timeline ---
    ax = axes[1, 1]
    phase_times = result.phase_data.times()
    phase_vals = result.phase_data.raw_values()

    phase_colors = {1: "#55A868", 2: "#4C72B0", 3: "#C44E52"}

    if phase_times and phase_vals:
        # Draw colored spans for each phase
        prev_t = phase_times[0]
        prev_p = int(phase_vals[0])
        for t, p in zip(phase_times[1:], phase_vals[1:], strict=False):
            p = int(p)
            ax.axvspan(prev_t, t, alpha=0.3, color=phase_colors.get(prev_p, "gray"))
            if p != prev_p:
                ax.axvline(x=t, color="black", linestyle="--", alpha=0.5)
            prev_t = t
            prev_p = p
        # Fill last span to end
        ax.axvspan(prev_t, result.duration_s, alpha=0.3, color=phase_colors.get(prev_p, "gray"))

    # Add legend patches
    from matplotlib.patches import Patch

    legend_patches = [
        Patch(facecolor=phase_colors[1], alpha=0.5, label="Phase 1: Write"),
        Patch(facecolor=phase_colors[2], alpha=0.5, label="Phase 2: Read"),
        Patch(facecolor=phase_colors[3], alpha=0.5, label="Phase 3: Mixed"),
    ]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=9)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Phase")
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["Write", "Read", "Mixed"])
    ax.set_title("Workload Phase Timeline")
    ax.set_ylim(0.5, 3.5)
    ax.grid(True, alpha=0.3, axis="x")

    fig.suptitle("B-Tree vs LSM Tree Storage Engine Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "btree_vs_lsm_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'btree_vs_lsm_comparison.png'}")


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="B-tree vs LSM tree storage engine comparison")
    parser.add_argument(
        "--duration", type=float, default=30.0, help="Simulation duration in seconds"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (-1 for random)")
    parser.add_argument(
        "--output",
        type=str,
        default="output/btree_vs_lsm",
        help="Output directory for visualizations",
    )
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization generation")
    args = parser.parse_args()

    seed = None if args.seed == -1 else args.seed

    print("Running B-tree vs LSM tree comparison...")
    print(f"  Duration: {args.duration}s")
    print(f"  Seed: {seed if seed is not None else 'random'}")

    result = run_btree_vs_lsm_simulation(
        duration_s=args.duration,
        seed=seed,
    )

    print_summary(result)

    if not args.no_viz:
        output_dir = Path(args.output)
        visualize_results(result, output_dir)
        print(f"\nVisualizations saved to: {output_dir.absolute()}")
