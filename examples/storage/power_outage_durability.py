"""Power outage durability demonstration.

Shows that async WAL sync policies can lose data on power failure while
sync-every-write preserves all acknowledged writes.

Two database nodes receive identical streams of writes:

- **durable_db**: SyncEveryWrite — every WAL append is fsynced before ack.
  Maximum durability, higher write latency.
- **fast_db**: SyncOnBatch(50) — WAL is fsynced every 50 writes. Lower
  latency, but up to 49 writes can be lost on crash.

After processing writes for a few seconds, both nodes suffer a simulated
power outage (crash). On recovery, the durable node has all its data while
the fast node is missing recent writes that hadn't been synced.

## Architecture Diagram

```
    Source (constant 500 writes/s)
        |
        v
    FanoutWriter ──────── yield from lsm.put(key, value)
        |                     │                  │
        ├── durable_db        │                  │
        │   (SyncEveryWrite)  │                  │
        │   WAL ──► Memtable ──► SSTable         │
        │   ^ every write synced                 │
        │                                        │
        └── fast_db                              │
            (SyncOnBatch(50))                    │
            WAL ──► Memtable ──► SSTable
            ^ synced every 50 writes
                ╲
                 ╲ gap: up to 49 writes
                   only in volatile memory!

    ── Power Outage at t=3s ──

    durable_db: SSTable data + all WAL entries survived
    fast_db:    SSTable data + WAL entries up to last batch only
```

## Key Insight

The window of data loss equals the gap between the last sync and the crash.
With SyncEveryWrite there is no gap. With SyncOnBatch(N) the gap can be up
to N-1 writes. SyncPeriodic(interval) can lose up to `interval` seconds of
writes.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Generator

from happysimulator import (
    Entity,
    Event,
    Instant,
    Simulation,
    Source,
)
from happysimulator.components.storage import (
    LSMTree,
    WriteAheadLog,
    SyncEveryWrite,
    SyncOnBatch,
)
from happysimulator.core.control.breakpoints import TimeBreakpoint


# =============================================================================
# Writer Entity
# =============================================================================


class FanoutWriter(Entity):
    """Receives write events and fans out to multiple LSMTree instances.

    Each incoming event triggers a put() on every registered database.
    Tracks which keys have been acknowledged by all databases.
    """

    def __init__(
        self,
        name: str,
        *,
        databases: list[LSMTree],
    ) -> None:
        super().__init__(name)
        self.databases = databases
        self.keys_written: list[str] = []
        self._counter: int = 0

    def set_clock(self, clock) -> None:
        super().set_clock(clock)
        for db in self.databases:
            db.set_clock(clock)

    def handle_event(self, event: Event) -> Generator[float, None, None]:
        key = f"key_{self._counter:05d}"
        value = f"value_{self._counter}"
        self._counter += 1

        for db in self.databases:
            yield from db.put(key, value)

        self.keys_written.append(key)


# =============================================================================
# Simulation
# =============================================================================


def run_power_outage_demo(
    *,
    write_rate: float = 500.0,
    crash_time_s: float = 3.0,
    batch_size: int = 50,
    memtable_size: int = 100,
    seed: int = 42,
) -> dict:
    """Run the power outage durability demonstration.

    Args:
        write_rate: Writes per second.
        crash_time_s: Simulation time at which the power outage occurs.
        batch_size: Batch size for the async sync policy.
        memtable_size: Entries per memtable before flush to SSTable.
        seed: Random seed for reproducibility.

    Returns:
        Dict with results for both nodes.
    """
    random.seed(seed)

    # --- Build two databases with different sync policies ---

    wal_durable = WriteAheadLog(
        "wal_durable",
        sync_policy=SyncEveryWrite(),
        write_latency=0.0001,
        sync_latency=0.001,
    )
    lsm_durable = LSMTree(
        "durable_db",
        memtable_size=memtable_size,
        wal=wal_durable,
    )

    wal_fast = WriteAheadLog(
        "wal_fast",
        sync_policy=SyncOnBatch(batch_size=batch_size),
        write_latency=0.0001,
        sync_latency=0.001,
    )
    lsm_fast = LSMTree(
        "fast_db",
        memtable_size=memtable_size,
        wal=wal_fast,
    )

    writer = FanoutWriter(
        "Writer",
        databases=[lsm_durable, lsm_fast],
    )

    source = Source.constant(
        rate=write_rate,
        target=writer,
        event_type="Write",
    )

    # --- Run until crash time, then pause ---

    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=Instant.from_seconds(crash_time_s + 10.0),
        sources=[source],
        entities=[writer, wal_durable, wal_fast, lsm_durable, lsm_fast],
    )

    # Breakpoint pauses the simulation at crash time, leaving
    # unflushed data in memtables and unsynced entries in WALs.
    sim.control.add_breakpoint(TimeBreakpoint(time=Instant.from_seconds(crash_time_s)))
    sim.run()

    total_written = len(writer.keys_written)

    # --- Verify pre-crash state: both DBs have the same data ---

    pre_crash_durable = sum(
        1 for k in writer.keys_written if lsm_durable.get_sync(k) is not None
    )
    pre_crash_fast = sum(
        1 for k in writer.keys_written if lsm_fast.get_sync(k) is not None
    )

    # --- POWER OUTAGE: crash both nodes ---

    crash_durable = lsm_durable.crash()
    crash_fast = lsm_fast.crash()

    # --- RECOVERY: bring both nodes back ---

    recovery_durable = lsm_durable.recover_from_crash()
    recovery_fast = lsm_fast.recover_from_crash()

    # --- Check what survived ---

    surviving_durable = sum(
        1 for k in writer.keys_written if lsm_durable.get_sync(k) is not None
    )
    surviving_fast = sum(
        1 for k in writer.keys_written if lsm_fast.get_sync(k) is not None
    )

    lost_durable = total_written - surviving_durable
    lost_fast = total_written - surviving_fast

    return {
        "total_written": total_written,
        "pre_crash": {
            "durable": pre_crash_durable,
            "fast": pre_crash_fast,
        },
        "crash_info": {
            "durable": crash_durable,
            "fast": crash_fast,
        },
        "recovery_info": {
            "durable": recovery_durable,
            "fast": recovery_fast,
        },
        "post_recovery": {
            "durable_surviving": surviving_durable,
            "fast_surviving": surviving_fast,
            "durable_lost": lost_durable,
            "fast_lost": lost_fast,
        },
        "wal_stats": {
            "durable_syncs": wal_durable.stats.syncs,
            "fast_syncs": wal_fast.stats.syncs,
        },
    }


# =============================================================================
# Summary
# =============================================================================


def print_summary(results: dict) -> None:
    """Print a formatted comparison of the two nodes."""
    total = results["total_written"]
    pre = results["pre_crash"]
    crash = results["crash_info"]
    recovery = results["recovery_info"]
    post = results["post_recovery"]
    wal = results["wal_stats"]

    print("\n" + "=" * 70)
    print("POWER OUTAGE DURABILITY DEMONSTRATION")
    print("=" * 70)

    print(f"\nTotal writes acknowledged before crash: {total}")

    print(f"\n--- Pre-crash verification ---")
    print(f"  Durable DB (SyncEveryWrite): {pre['durable']}/{total} keys present")
    print(f"  Fast DB    (SyncOnBatch):     {pre['fast']}/{total} keys present")

    print(f"\n--- Power outage! ---")
    cd = crash["durable"]
    cf = crash["fast"]
    print(f"  Durable DB lost: {cd['memtable_entries_lost']} memtable + {cd['wal_entries_lost']} WAL entries")
    print(f"  Fast DB    lost: {cf['memtable_entries_lost']} memtable + {cf['wal_entries_lost']} WAL entries")

    print(f"\n--- Recovery ---")
    rd = recovery["durable"]
    rf = recovery["fast"]
    print(f"  Durable DB: replayed {rd['wal_entries_replayed']} WAL entries, {rd['sstable_keys']} SSTable keys")
    print(f"  Fast DB:    replayed {rf['wal_entries_replayed']} WAL entries, {rf['sstable_keys']} SSTable keys")

    print(f"\n--- Post-recovery data ---")
    print(f"  {'':30s} {'Durable':>10s} {'Fast':>10s}")
    print(f"  {'-'*30} {'-'*10} {'-'*10}")
    print(f"  {'Keys surviving':30s} {post['durable_surviving']:>10d} {post['fast_surviving']:>10d}")
    print(f"  {'Keys LOST':30s} {post['durable_lost']:>10d} {post['fast_lost']:>10d}")
    print(f"  {'WAL syncs performed':30s} {wal['durable_syncs']:>10d} {wal['fast_syncs']:>10d}")

    if post["fast_lost"] > post["durable_lost"]:
        extra = post["fast_lost"] - post["durable_lost"]
        print(f"\n  ** The fast node lost {extra} MORE writes than the durable node.")
        print(f"     These were acknowledged but not yet fsynced to disk.")
        if post["durable_lost"] > 0:
            print(f"     (The durable node's {post['durable_lost']} losses are writes")
            print(f"      that were mid-fsync at the moment of the crash.)")
        print(f"     Durability cost: {wal['durable_syncs']} syncs vs {wal['fast_syncs']} syncs.")
    elif post["durable_lost"] == 0 and post["fast_lost"] == 0:
        print(f"\n  Both nodes recovered all data (crash aligned with sync boundary).")

    print("\n" + "=" * 70)


# =============================================================================
# Visualization
# =============================================================================


def visualize_results(results: dict, output_dir: Path) -> None:
    """Generate a comparison chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping visualization")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    post = results["post_recovery"]
    total = results["total_written"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Chart 1: Data survival comparison
    ax = axes[0]
    labels = ["Durable\n(SyncEveryWrite)", "Fast\n(SyncOnBatch)"]
    surviving = [post["durable_surviving"], post["fast_surviving"]]
    lost = [post["durable_lost"], post["fast_lost"]]
    colors_surviving = ["#2ecc71", "#3498db"]
    colors_lost = ["#e74c3c", "#e74c3c"]

    bars_s = ax.bar(labels, surviving, color=colors_surviving, label="Surviving", edgecolor="black", alpha=0.85)
    bars_l = ax.bar(labels, lost, bottom=surviving, color=colors_lost, label="Lost", edgecolor="black", alpha=0.85)

    for bar_s, bar_l, s, l in zip(bars_s, bars_l, surviving, lost):
        ax.text(bar_s.get_x() + bar_s.get_width() / 2, bar_s.get_height() / 2,
                f"{s}", ha="center", va="center", fontsize=11, fontweight="bold", color="white")
        if l > 0:
            ax.text(bar_l.get_x() + bar_l.get_width() / 2, bar_s.get_height() + bar_l.get_height() / 2,
                    f"{l} lost", ha="center", va="center", fontsize=11, fontweight="bold", color="white")

    ax.set_ylabel("Keys")
    ax.set_title("Data After Power Outage + Recovery")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Chart 2: WAL sync count comparison
    ax = axes[1]
    wal = results["wal_stats"]
    sync_counts = [wal["durable_syncs"], wal["fast_syncs"]]
    bars = ax.bar(labels, sync_counts, color=["#e74c3c", "#2ecc71"], edgecolor="black", alpha=0.85)
    for bar, val in zip(bars, sync_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of fsyncs")
    ax.set_title("Durability Cost (fsync count)")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Power Outage Durability: SyncEveryWrite vs SyncOnBatch", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "power_outage_durability.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'power_outage_durability.png'}")


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Demonstrate data loss from power outage with async WAL sync"
    )
    parser.add_argument("--rate", type=float, default=500.0, help="Write rate (writes/s)")
    parser.add_argument("--crash-time", type=float, default=3.0, help="When the power outage happens (s)")
    parser.add_argument("--batch-size", type=int, default=50, help="SyncOnBatch batch size")
    parser.add_argument("--memtable-size", type=int, default=100, help="Memtable flush threshold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="output/power_outage", help="Output directory")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    args = parser.parse_args()

    print("Running power outage durability demo...")
    print(f"  Write rate:     {args.rate} writes/s")
    print(f"  Crash time:     {args.crash_time}s")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Memtable size:  {args.memtable_size}")

    results = run_power_outage_demo(
        write_rate=args.rate,
        crash_time_s=args.crash_time,
        batch_size=args.batch_size,
        memtable_size=args.memtable_size,
        seed=args.seed,
    )

    print_summary(results)

    if not args.no_viz:
        visualize_results(results, Path(args.output))
