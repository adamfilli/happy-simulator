"""SSTable bloom filter analysis: page reads saved for missing keys.

This example demonstrates how bloom filters in SSTables avoid unnecessary
disk I/O. When looking up a key that does not exist in an SSTable, the
bloom filter can reject the query with zero page reads. Without the
filter, every lookup would need at least 2 page reads (index + data).

## How It Works

```
+-----------------------------------------------------------------------+
|                       SSTABLE POINT LOOKUP                             |
+-----------------------------------------------------------------------+

    Query: get("key_12345")

    Step 1: Bloom Filter Check
    +--------------------------+
    |  Bloom Filter            |
    |  "key_12345" present?    |
    |                          |
    |  NO  -> return None      |   0 page reads (saved 2!)
    |  YES -> continue lookup  |
    +--------------------------+
             |
             v (bloom says "maybe yes")
    Step 2: Sparse Index Lookup     1 page read
    Step 3: Data Page Read          1 page read
                                    --------
                                    2 page reads total
```

## Tradeoff

Lower false positive rates require more memory for the bloom filter
but save more page reads on missing-key lookups. This example creates
SSTables with different ``bloom_fp_rate`` settings and measures:

- Actual vs configured false positive rate
- Page reads saved for 10,000 missing-key lookups
- Memory overhead of the bloom filter at each rate

This is a pure data structure demo -- no simulation is needed because
SSTable is not an Entity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from happysimulator.components.storage import SSTable


# =============================================================================
# Analysis
# =============================================================================


@dataclass(frozen=True)
class BloomFilterResult:
    """Results from testing one SSTable's bloom filter."""

    configured_fp_rate: float
    actual_fp_rate: float
    bloom_size_bits: int
    num_entries: int
    existing_keys_tested: int
    missing_keys_tested: int
    false_positives: int
    true_negatives: int
    page_reads_with_bloom: int
    page_reads_without_bloom: int
    page_reads_saved: int


@dataclass
class ComparisonResult:
    """Combined results across all tested FP rates."""

    results: list[BloomFilterResult]
    num_entries: int
    num_existing_tested: int
    num_missing_tested: int


def run_bloom_filter_analysis(
    *,
    num_entries: int = 10_000,
    num_test_keys: int = 10_000,
    fp_rates: tuple[float, ...] = (0.001, 0.01, 0.1),
    seed: int | None = 42,
) -> ComparisonResult:
    """Analyze bloom filter effectiveness at various false positive rates.

    Creates SSTables with ``num_entries`` key-value pairs and tests lookups
    against ``num_test_keys`` existing keys and ``num_test_keys`` keys that
    are guaranteed to be absent.

    Args:
        num_entries: Number of entries to insert into each SSTable.
        num_test_keys: Number of existing and missing keys to test.
        fp_rates: Tuple of target bloom filter false positive rates.
        seed: Unused (deterministic), kept for CLI interface consistency.

    Returns:
        ComparisonResult with per-rate metrics.
    """
    # Build the data set: keys are "key_NNNNN" with zero-padded indices
    data = [(f"key_{i:05d}", f"value_{i}") for i in range(num_entries)]

    # Existing keys to test (sample from the data set)
    existing_keys = [f"key_{i:05d}" for i in range(num_test_keys)]

    # Missing keys: use a range that does not overlap with [0, num_entries)
    missing_keys = [f"miss_{i:05d}" for i in range(num_test_keys)]

    results: list[BloomFilterResult] = []
    for fp_rate in fp_rates:
        sst = SSTable(data, bloom_fp_rate=fp_rate)

        # Test existing keys -- bloom filter must return True for all
        existing_tested = 0
        for key in existing_keys:
            sst.contains(key)
            existing_tested += 1

        # Test missing keys -- count false positives
        false_positives = 0
        true_negatives = 0
        page_reads_with_bloom = 0
        for key in missing_keys:
            reads = sst.page_reads_for_get(key)
            page_reads_with_bloom += reads
            if sst.contains(key):
                false_positives += 1
            else:
                true_negatives += 1

        # Without bloom filter, every lookup requires 2 page reads
        page_reads_without_bloom = num_test_keys * 2

        # Page reads for existing keys (always 2 each, bloom always passes)
        page_reads_existing = num_test_keys * 2
        page_reads_with_bloom += page_reads_existing
        page_reads_without_bloom += page_reads_existing

        actual_fp_rate = false_positives / num_test_keys if num_test_keys > 0 else 0.0

        results.append(BloomFilterResult(
            configured_fp_rate=fp_rate,
            actual_fp_rate=actual_fp_rate,
            bloom_size_bits=sst.bloom_filter.size_bits,
            num_entries=num_entries,
            existing_keys_tested=existing_tested,
            missing_keys_tested=num_test_keys,
            false_positives=false_positives,
            true_negatives=true_negatives,
            page_reads_with_bloom=page_reads_with_bloom,
            page_reads_without_bloom=page_reads_without_bloom,
            page_reads_saved=page_reads_without_bloom - page_reads_with_bloom,
        ))

    return ComparisonResult(
        results=results,
        num_entries=num_entries,
        num_existing_tested=num_test_keys,
        num_missing_tested=num_test_keys,
    )


# =============================================================================
# Summary
# =============================================================================


def print_summary(comparison: ComparisonResult) -> None:
    """Print a comparison table of bloom filter effectiveness."""
    print("\n" + "=" * 90)
    print("SSTABLE BLOOM FILTER ANALYSIS")
    print("=" * 90)

    print(f"\nConfiguration:")
    print(f"  SSTable entries:     {comparison.num_entries:,}")
    print(f"  Existing keys tested: {comparison.num_existing_tested:,}")
    print(f"  Missing keys tested:  {comparison.num_missing_tested:,}")

    header = (
        f"{'FP Rate (cfg)':>14} "
        f"{'FP Rate (actual)':>16} "
        f"{'False Pos':>10} "
        f"{'True Neg':>10} "
        f"{'Bloom (KB)':>12} "
        f"{'Reads w/ BF':>12} "
        f"{'Reads w/o BF':>13} "
        f"{'Saved':>8}"
    )
    print(f"\n{header}")
    print("-" * 90)

    for r in comparison.results:
        bloom_kb = r.bloom_size_bits / 8 / 1024
        print(
            f"{r.configured_fp_rate:>14.3f} "
            f"{r.actual_fp_rate:>16.4f} "
            f"{r.false_positives:>10,} "
            f"{r.true_negatives:>10,} "
            f"{bloom_kb:>12.1f} "
            f"{r.page_reads_with_bloom:>12,} "
            f"{r.page_reads_without_bloom:>13,} "
            f"{r.page_reads_saved:>8,}"
        )

    print()

    # Interpretation
    if len(comparison.results) >= 2:
        best = min(comparison.results, key=lambda r: r.actual_fp_rate)
        worst = max(comparison.results, key=lambda r: r.actual_fp_rate)
        print("Observations:")
        print(f"  - Tightest filter (fp={best.configured_fp_rate}) saved {best.page_reads_saved:,} page reads")
        print(f"    but used {best.bloom_size_bits / 8 / 1024:.1f} KB of memory")
        print(f"  - Loosest filter (fp={worst.configured_fp_rate}) saved {worst.page_reads_saved:,} page reads")
        print(f"    using only {worst.bloom_size_bits / 8 / 1024:.1f} KB of memory")
        if best.bloom_size_bits > 0 and worst.bloom_size_bits > 0:
            memory_ratio = best.bloom_size_bits / worst.bloom_size_bits
            reads_diff = best.page_reads_saved - worst.page_reads_saved
            print(f"  - The tightest filter uses {memory_ratio:.1f}x more memory for {reads_diff:,} additional page reads saved")

    print("\n" + "=" * 90)


# =============================================================================
# Visualization
# =============================================================================


def visualize_results(comparison: ComparisonResult, output_dir: Path) -> None:
    """Generate bar charts of bloom filter effectiveness."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping visualization")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    labels = [f"fp={r.configured_fp_rate}" for r in comparison.results]
    configured_rates = [r.configured_fp_rate for r in comparison.results]
    actual_rates = [r.actual_fp_rate for r in comparison.results]
    page_reads_saved = [r.page_reads_saved for r in comparison.results]
    bloom_kb = [r.bloom_size_bits / 8 / 1024 for r in comparison.results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Chart 1: Configured vs Actual FP rate
    ax = axes[0]
    x = range(len(labels))
    width = 0.35
    bars1 = ax.bar([i - width / 2 for i in x], configured_rates, width,
                   label="Configured", color="#3498db", edgecolor="black", alpha=0.85)
    bars2 = ax.bar([i + width / 2 for i in x], actual_rates, width,
                   label="Actual", color="#e74c3c", edgecolor="black", alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("False Positive Rate")
    ax.set_title("Configured vs Actual FP Rate")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Chart 2: Page reads saved
    ax = axes[1]
    colors = ["#2ecc71", "#27ae60", "#1e8449"]
    bars = ax.bar(labels, page_reads_saved, color=colors[:len(labels)],
                  edgecolor="black", alpha=0.85)
    for bar, val in zip(bars, page_reads_saved):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{val:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Page Reads Saved")
    ax.set_title("Page Reads Saved (missing key lookups)")
    ax.grid(True, alpha=0.3, axis="y")

    # Chart 3: Memory overhead
    ax = axes[2]
    bars = ax.bar(labels, bloom_kb, color=["#9b59b6", "#8e44ad", "#6c3483"][:len(labels)],
                  edgecolor="black", alpha=0.85)
    for bar, val in zip(bars, bloom_kb):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Bloom Filter Size (KB)")
    ax.set_title("Memory Overhead")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"SSTable Bloom Filter Analysis ({comparison.num_entries:,} entries, "
        f"{comparison.num_missing_tested:,} missing-key lookups)",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "sstable_bloom_filter.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'sstable_bloom_filter.png'}")


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SSTable bloom filter analysis: page reads saved for missing keys"
    )
    parser.add_argument("--entries", type=int, default=10_000, help="Number of SSTable entries")
    parser.add_argument("--test-keys", type=int, default=10_000, help="Number of test keys (existing and missing)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (-1 for random)")
    parser.add_argument("--output", type=str, default="output/sstable_bloom_filter", help="Output directory")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization generation")
    args = parser.parse_args()

    seed = None if args.seed == -1 else args.seed

    print("Running SSTable bloom filter analysis...")
    print(f"  Entries: {args.entries:,}")
    print(f"  Test keys: {args.test_keys:,} existing + {args.test_keys:,} missing")
    print(f"  FP rates: 0.001, 0.01, 0.1")

    comparison = run_bloom_filter_analysis(
        num_entries=args.entries,
        num_test_keys=args.test_keys,
        fp_rates=(0.001, 0.01, 0.1),
        seed=seed,
    )

    print_summary(comparison)

    if not args.no_viz:
        output_dir = Path(args.output)
        visualize_results(comparison, output_dir)
        print(f"\nVisualizations saved to: {output_dir.absolute()}")
