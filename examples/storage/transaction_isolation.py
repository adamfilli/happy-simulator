"""Transaction isolation level comparison: conflict behavior under contention.

This example demonstrates how different isolation levels affect transaction
success rates when concurrent transactions access overlapping key sets.

For each isolation level (READ_COMMITTED, SNAPSHOT_ISOLATION, SERIALIZABLE),
we run pairs of concurrent transactions that read and write the same keys,
then measure commits, aborts, and conflict rates.

## Isolation Level Semantics

- READ_COMMITTED: No conflict detection. All transactions commit. Reads see
  the latest committed value. Susceptible to write skew and lost updates.
- SNAPSHOT_ISOLATION: Write-write conflict detection. Two transactions that
  write the same key conflict; the second to commit is aborted. Prevents
  lost updates but allows write skew on disjoint key sets.
- SERIALIZABLE: Full conflict detection. Write-write AND read-write conflicts
  detected. If transaction A writes a key that transaction B read (or vice
  versa), one of them aborts. Strongest guarantee, highest abort rate.

## Architecture Diagram

```
    TransactionWorkloadDriver
           |
           v
    TransactionManager (isolation=X)
           |
           v
    LSMTree (backing store)
```

Each round:
1. Transaction A begins, reads key_0..key_K, writes key_0..key_K.
2. Transaction A commits (records in commit log).
3. Transaction B begins (same snapshot), reads key_0..key_K, writes key_0..key_K.
4. Transaction B attempts commit -> may conflict with A's commit log entry.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

from happysimulator import (
    Entity,
    Event,
    Instant,
    Simulation,
    SimulationSummary,
    Source,
)
from happysimulator.components.storage import (
    LSMTree,
    TransactionManager,
    StorageTransaction,
    IsolationLevel,
    SizeTieredCompaction,
)


# =============================================================================
# Custom Entity: TransactionWorkloadDriver
# =============================================================================


class TransactionWorkloadDriver(Entity):
    """Drives pairs of concurrent transactions to provoke conflicts.

    Each incoming event triggers one round of two transactions that
    read and write the same set of hot keys. The first transaction
    always commits; the second may conflict depending on isolation level.

    Attributes:
        commits: Number of successful commits across all rounds.
        aborts: Number of aborted transactions across all rounds.
        conflicts: Number of conflicts detected.
        rounds: Number of transaction pair rounds completed.
    """

    def __init__(
        self,
        name: str,
        *,
        tx_manager: TransactionManager,
        num_hot_keys: int = 5,
    ) -> None:
        super().__init__(name)
        self._tx_manager = tx_manager
        self._num_hot_keys = num_hot_keys
        self.commits: int = 0
        self.aborts: int = 0
        self.conflicts: int = 0
        self.rounds: int = 0

    def handle_event(self, event: Event) -> list[Event]:
        """Run one round of two conflicting transactions."""
        self.rounds += 1

        # Transaction A: read all hot keys, then write new values
        tx_a = self._tx_manager.begin_sync()
        for i in range(self._num_hot_keys):
            key = f"hot_key_{i}"
            tx_a._read_set.add(key)
            tx_a._write_set[key] = f"value_a_{self.rounds}_{i}"
        self._tx_manager._total_reads += self._num_hot_keys
        self._tx_manager._total_writes += self._num_hot_keys

        # Commit A (should always succeed since no prior commits to conflict with)
        has_conflict_a = self._tx_manager._check_conflict(tx_a)
        if has_conflict_a:
            tx_a.abort()
            self.aborts += 1
            self.conflicts += 1
        else:
            # Apply writes manually (sync path)
            for key, value in tx_a._write_set.items():
                self._tx_manager._store.put_sync(key, value)
            self._tx_manager._version += 1
            from happysimulator.components.storage.transaction_manager import _CommitLogEntry
            entry = _CommitLogEntry(
                tx_id=tx_a._tx_id,
                version=self._tx_manager._version,
                keys_written=frozenset(tx_a._write_set.keys()),
                keys_read=frozenset(tx_a._read_set),
            )
            self._tx_manager._commit_log.append(entry)
            tx_a._committed = True
            self._tx_manager._total_committed += 1
            self._tx_manager._active_txns.pop(tx_a._tx_id, None)
            self.commits += 1

        # Transaction B: begins at the same snapshot version as A started,
        # reads and writes the same keys
        tx_b = self._tx_manager.begin_sync()
        # Reset snapshot to match A's start (simulate concurrent start)
        tx_b._snapshot_version = tx_a._snapshot_version
        for i in range(self._num_hot_keys):
            key = f"hot_key_{i}"
            tx_b._read_set.add(key)
            tx_b._write_set[key] = f"value_b_{self.rounds}_{i}"
        self._tx_manager._total_reads += self._num_hot_keys
        self._tx_manager._total_writes += self._num_hot_keys

        # Commit B (may conflict with A's committed writes)
        has_conflict_b = self._tx_manager._check_conflict(tx_b)
        if has_conflict_b:
            tx_b._aborted = True
            self._tx_manager._total_aborted += 1
            self._tx_manager._active_txns.pop(tx_b._tx_id, None)
            self.aborts += 1
            self.conflicts += 1
        else:
            for key, value in tx_b._write_set.items():
                self._tx_manager._store.put_sync(key, value)
            self._tx_manager._version += 1
            entry = _CommitLogEntry(
                tx_id=tx_b._tx_id,
                version=self._tx_manager._version,
                keys_written=frozenset(tx_b._write_set.keys()),
                keys_read=frozenset(tx_b._read_set),
            )
            self._tx_manager._commit_log.append(entry)
            tx_b._committed = True
            self._tx_manager._total_committed += 1
            self._tx_manager._active_txns.pop(tx_b._tx_id, None)
            self.commits += 1

        return []


# =============================================================================
# Simulation
# =============================================================================


@dataclass
class IsolationResult:
    """Results from running transactions at a single isolation level."""

    isolation: IsolationLevel
    driver: TransactionWorkloadDriver
    tx_manager: TransactionManager
    summary: SimulationSummary


@dataclass
class SimulationResult:
    """Combined results across all isolation levels."""

    results: list[IsolationResult]
    num_rounds: int
    num_hot_keys: int
    duration_s: float


def _run_single_isolation(
    isolation: IsolationLevel,
    *,
    duration_s: float,
    rate: float,
    num_hot_keys: int,
) -> IsolationResult:
    """Run the transaction workload at a single isolation level."""
    lsm = LSMTree(
        f"store_{isolation.value}",
        memtable_size=1000,
        compaction_strategy=SizeTieredCompaction(min_sstables=4),
    )

    # Pre-populate hot keys so reads find values
    for i in range(num_hot_keys):
        lsm.put_sync(f"hot_key_{i}", f"initial_{i}")

    tx_manager = TransactionManager(
        f"txm_{isolation.value}",
        store=lsm,
        isolation=isolation,
    )

    driver = TransactionWorkloadDriver(
        f"driver_{isolation.value}",
        tx_manager=tx_manager,
        num_hot_keys=num_hot_keys,
    )

    source = Source.constant(
        rate=rate,
        target=driver,
        event_type="TxRound",
        stop_after=Instant.from_seconds(duration_s),
    )

    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=Instant.from_seconds(duration_s + 1.0),
        sources=[source],
        entities=[lsm, tx_manager, driver],
    )
    summary = sim.run()

    return IsolationResult(
        isolation=isolation,
        driver=driver,
        tx_manager=tx_manager,
        summary=summary,
    )


def run_isolation_simulation(
    *,
    duration_s: float = 10.0,
    rate: float = 100.0,
    num_hot_keys: int = 5,
    seed: int | None = 42,
) -> SimulationResult:
    """Run the transaction isolation comparison simulation.

    Executes the same contention workload under each isolation level,
    collecting commit/abort/conflict counts.

    Args:
        duration_s: Simulation duration in seconds.
        rate: Transaction-pair rounds per second.
        num_hot_keys: Number of hot keys that each transaction touches.
        seed: Random seed for reproducibility.

    Returns:
        SimulationResult with results for each isolation level.
    """
    if seed is not None:
        random.seed(seed)

    isolation_levels = [
        IsolationLevel.READ_COMMITTED,
        IsolationLevel.SNAPSHOT_ISOLATION,
        IsolationLevel.SERIALIZABLE,
    ]

    results: list[IsolationResult] = []
    for level in isolation_levels:
        if seed is not None:
            random.seed(seed)
        result = _run_single_isolation(
            level,
            duration_s=duration_s,
            rate=rate,
            num_hot_keys=num_hot_keys,
        )
        results.append(result)

    # Use the first driver's round count as canonical
    num_rounds = results[0].driver.rounds if results else 0

    return SimulationResult(
        results=results,
        num_rounds=num_rounds,
        num_hot_keys=num_hot_keys,
        duration_s=duration_s,
    )


# =============================================================================
# Summary
# =============================================================================


def print_summary(result: SimulationResult) -> None:
    """Print comparison table of isolation level outcomes."""
    print("\n" + "=" * 72)
    print("TRANSACTION ISOLATION LEVEL COMPARISON")
    print("=" * 72)

    print(f"\nWorkload: {result.num_rounds} rounds of 2 concurrent transactions")
    print(f"  Each transaction reads and writes {result.num_hot_keys} hot keys")
    print(f"  Duration: {result.duration_s}s")

    # Table header
    header = (
        f"{'Isolation Level':<25} "
        f"{'Rounds':>8} "
        f"{'Commits':>8} "
        f"{'Aborts':>8} "
        f"{'Conflicts':>10} "
        f"{'Commit %':>10} "
        f"{'Abort %':>10}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    for ir in result.results:
        d = ir.driver
        total_tx = d.commits + d.aborts
        commit_pct = (d.commits / total_tx * 100) if total_tx > 0 else 0.0
        abort_pct = (d.aborts / total_tx * 100) if total_tx > 0 else 0.0

        print(
            f"{ir.isolation.value:<25} "
            f"{d.rounds:>8} "
            f"{d.commits:>8} "
            f"{d.aborts:>8} "
            f"{d.conflicts:>10} "
            f"{commit_pct:>9.1f}% "
            f"{abort_pct:>9.1f}%"
        )

    # Transaction manager stats
    print(f"\n{'TransactionManager Stats':}")
    tm_header = (
        f"{'Isolation Level':<25} "
        f"{'Started':>8} "
        f"{'Committed':>10} "
        f"{'Aborted':>8} "
        f"{'Detected':>10}"
    )
    print(f"{tm_header}")
    print("-" * len(tm_header))

    for ir in result.results:
        ts = ir.tx_manager.stats
        print(
            f"{ir.isolation.value:<25} "
            f"{ts.transactions_started:>8} "
            f"{ts.transactions_committed:>10} "
            f"{ts.transactions_aborted:>8} "
            f"{ts.conflicts_detected:>10}"
        )

    print("\n" + "=" * 72)
    print("INTERPRETATION:")
    print("-" * 72)

    rc = next((r for r in result.results
               if r.isolation == IsolationLevel.READ_COMMITTED), None)
    si = next((r for r in result.results
               if r.isolation == IsolationLevel.SNAPSHOT_ISOLATION), None)
    sr = next((r for r in result.results
               if r.isolation == IsolationLevel.SERIALIZABLE), None)

    if rc:
        print(f"\n  READ_COMMITTED:")
        print(f"    All {rc.driver.commits} transactions committed (0 conflicts).")
        print(f"    No conflict detection means lost updates are possible.")

    if si:
        print(f"\n  SNAPSHOT_ISOLATION:")
        print(f"    {si.driver.commits} commits, {si.driver.aborts} aborts.")
        print(f"    Write-write conflicts detected: when two transactions")
        print(f"    write the same key, the second to commit is aborted.")

    if sr:
        print(f"\n  SERIALIZABLE:")
        print(f"    {sr.driver.commits} commits, {sr.driver.aborts} aborts.")
        print(f"    Read-write + write-write conflicts detected.")
        print(f"    Highest abort rate, but strongest consistency guarantee.")

    if si and sr:
        si_abort_pct = si.driver.aborts / max(si.driver.commits + si.driver.aborts, 1) * 100
        sr_abort_pct = sr.driver.aborts / max(sr.driver.commits + sr.driver.aborts, 1) * 100
        print(f"\n  Abort rate comparison:")
        print(f"    SNAPSHOT_ISOLATION: {si_abort_pct:.1f}%")
        print(f"    SERIALIZABLE:      {sr_abort_pct:.1f}%")
        if sr_abort_pct > si_abort_pct:
            print(f"    Serializable has {sr_abort_pct - si_abort_pct:.1f}pp higher abort rate")
            print(f"    due to additional read-write conflict detection.")

    print("\n" + "=" * 72)


# =============================================================================
# Visualization
# =============================================================================


def visualize_results(result: SimulationResult, output_dir: Path) -> None:
    """Generate stacked bar chart of commits vs aborts per isolation level."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping visualization")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    levels = [r.isolation.value.replace("_", "\n") for r in result.results]
    commits = [r.driver.commits for r in result.results]
    aborts = [r.driver.aborts for r in result.results]

    # --- Chart 1: Stacked bar chart of commits vs aborts ---
    ax = axes[0]
    x = range(len(levels))
    bars_commits = ax.bar(x, commits, label="Commits", color="#55A868",
                          edgecolor="black", alpha=0.85)
    bars_aborts = ax.bar(x, aborts, bottom=commits, label="Aborts",
                         color="#C44E52", edgecolor="black", alpha=0.85)

    # Add value labels
    for i, (c, a) in enumerate(zip(commits, aborts)):
        # Commit label
        if c > 0:
            ax.text(i, c / 2, str(c), ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white")
        # Abort label
        if a > 0:
            ax.text(i, c + a / 2, str(a), ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white")

    ax.set_xticks(list(x))
    ax.set_xticklabels(levels, fontsize=9)
    ax.set_ylabel("Transaction Count")
    ax.set_title("Commits vs Aborts")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    # --- Chart 2: Commit percentage ---
    ax = axes[1]
    commit_pcts = []
    abort_pcts = []
    for r in result.results:
        total = r.driver.commits + r.driver.aborts
        commit_pcts.append(r.driver.commits / total * 100 if total > 0 else 0)
        abort_pcts.append(r.driver.aborts / total * 100 if total > 0 else 0)

    bars_c = ax.bar(x, commit_pcts, label="Commit %", color="#55A868",
                    edgecolor="black", alpha=0.85)
    bars_a = ax.bar(x, abort_pcts, bottom=commit_pcts, label="Abort %",
                    color="#C44E52", edgecolor="black", alpha=0.85)

    for i, (cp, ap) in enumerate(zip(commit_pcts, abort_pcts)):
        if cp > 5:
            ax.text(i, cp / 2, f"{cp:.0f}%", ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white")
        if ap > 5:
            ax.text(i, cp + ap / 2, f"{ap:.0f}%", ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white")

    ax.set_xticks(list(x))
    ax.set_xticklabels(levels, fontsize=9)
    ax.set_ylabel("Percentage")
    ax.set_title("Commit / Abort Rate")
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    # --- Chart 3: Conflicts detected ---
    ax = axes[2]
    conflicts = [r.driver.conflicts for r in result.results]
    colors = ["#4C72B0", "#DD8452", "#C44E52"]
    bars = ax.bar(x, conflicts, color=colors, edgecolor="black", alpha=0.85)

    for bar, val in zip(bars, conflicts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels(levels, fontsize=9)
    ax.set_ylabel("Conflicts Detected")
    ax.set_title("Conflict Detection by Isolation Level")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Transaction Isolation Level Comparison "
        f"({result.num_rounds} rounds, {result.num_hot_keys} hot keys)",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "transaction_isolation_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'transaction_isolation_comparison.png'}")


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Transaction isolation level comparison simulation"
    )
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Simulation duration in seconds")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (-1 for random)")
    parser.add_argument("--output", type=str, default="output/transaction_isolation",
                        help="Output directory for visualizations")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip visualization generation")
    args = parser.parse_args()

    seed = None if args.seed == -1 else args.seed

    print("Running transaction isolation comparison...")
    print(f"  Duration: {args.duration}s")
    print(f"  Seed: {seed if seed is not None else 'random'}")

    result = run_isolation_simulation(
        duration_s=args.duration,
        seed=seed,
    )

    print_summary(result)

    if not args.no_viz:
        output_dir = Path(args.output)
        visualize_results(result, output_dir)
        print(f"\nVisualizations saved to: {output_dir.absolute()}")
