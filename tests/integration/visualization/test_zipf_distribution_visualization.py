"""Integration tests for Zipf distribution in load generation.

These tests verify that events generated through the simulation pipeline
correctly follow the expected Zipf distribution, with visualization output.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from happysimulator import (
    Entity,
    Event,
    Instant,
    Simulation,
    Source,
    ConstantArrivalTimeProvider,
    ConstantRateProfile,
    ZipfDistribution,
    UniformDistribution,
    DistributedFieldProvider,
)


# =============================================================================
# Statistics Collector Sink
# =============================================================================


class StatisticsCollectorSink(Entity):
    """Sink entity that accumulates statistics about received events.

    Collects field values from event context to verify distributions.
    This acts as the "end of the pipeline" that receives all generated
    events and records their characteristics.

    Attributes:
        field_counts: Dict mapping field names to Counter of observed values.
        events_received: Total count of events received.
        event_times: List of timestamps when events were received.
    """

    def __init__(self, name: str, tracked_fields: list[str]):
        """Initialize the statistics collector.

        Args:
            name: Entity name.
            tracked_fields: List of context field names to track.
        """
        super().__init__(name)
        self._tracked_fields = tracked_fields
        self.field_counts: dict[str, Counter] = {f: Counter() for f in tracked_fields}
        self.events_received: int = 0
        self.event_times: list[Instant] = []

    def handle_event(self, event: Event) -> list[Event]:
        """Record statistics from received event."""
        self.events_received += 1
        self.event_times.append(event.time)

        for field_name in self._tracked_fields:
            value = event.context.get(field_name)
            if value is not None:
                self.field_counts[field_name][value] += 1

        return []

    def get_frequency_distribution(self, field: str) -> list[tuple[Any, int]]:
        """Return (value, count) pairs sorted by count descending."""
        return self.field_counts[field].most_common()

    def get_top_n_percentage(self, field: str, n: int) -> float:
        """Return percentage of events in top n values."""
        counter = self.field_counts[field]
        if not counter:
            return 0.0
        top_n_count = sum(count for _, count in counter.most_common(n))
        return top_n_count / self.events_received * 100


# =============================================================================
# Test Classes
# =============================================================================


class TestZipfDistributionVisualization:
    """Integration tests for Zipf distribution with visualization."""

    def test_zipf_source_generates_expected_distribution(self, test_output_dir: Path):
        """Verify source with Zipf distribution produces expected skew.

        This test:
        1. Creates a Source that generates events with customer_id from Zipf distribution
        2. Runs the simulation to generate many events
        3. Collects statistics in a sink entity
        4. Verifies the distribution matches expected Zipf characteristics
        5. Generates visualizations comparing observed vs expected frequencies
        """
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        # Configuration
        num_customers = 100
        zipf_s = 1.0  # Classic Zipf
        num_events = 10000
        rate_per_second = 1000.0  # Generate quickly
        seed = 42

        # Create the statistics collector sink
        sink = StatisticsCollectorSink("sink", tracked_fields=["customer_id"])

        # Create Zipf distribution for customer IDs
        customer_dist = ZipfDistribution(range(num_customers), s=zipf_s, seed=seed)

        # Create event provider using the distribution
        provider = DistributedFieldProvider(
            target=sink,
            event_type="Request",
            field_distributions={"customer_id": customer_dist},
            stop_after=Instant.from_seconds(num_events / rate_per_second),
        )

        # Create source with constant arrival rate
        source = Source(
            name="ZipfSource",
            event_provider=provider,
            arrival_time_provider=ConstantArrivalTimeProvider(
                ConstantRateProfile(rate=rate_per_second),
                start_time=Instant.Epoch,
            ),
        )

        # Run simulation
        sim = Simulation(
            start_time=Instant.Epoch,
            duration=num_events / rate_per_second + 1.0,
            sources=[source],
            entities=[sink],
        )
        sim.run()

        # === ASSERTIONS ===

        # 1. Verify we generated the expected number of events
        assert sink.events_received >= num_events * 0.95  # Allow 5% tolerance

        # 2. Verify Zipf characteristic: top 10% of customers get majority of traffic
        top_10_pct = sink.get_top_n_percentage("customer_id", num_customers // 10)
        assert top_10_pct > 40, f"Top 10% should get >40% of traffic, got {top_10_pct:.1f}%"

        # 3. Verify rank-frequency relationship
        freq_dist = sink.get_frequency_distribution("customer_id")
        if len(freq_dist) >= 2:
            rank1_count = freq_dist[0][1]
            rank2_count = freq_dist[1][1]
            ratio = rank1_count / rank2_count
            # With s=1, rank 1 should appear ~2x as often as rank 2
            assert 1.5 < ratio < 3.0, f"Rank 1/2 ratio should be ~2, got {ratio:.2f}"

        # === VISUALIZATION ===

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Observed frequency by customer rank (log-log scale)
        ax = axes[0, 0]
        observed_counts = [count for _, count in freq_dist]
        ranks = range(1, len(observed_counts) + 1)
        ax.loglog(ranks, observed_counts, 'bo-', alpha=0.6, label='Observed', markersize=4)

        # Expected Zipf frequencies
        expected_probs = [1.0 / (k ** zipf_s) for k in ranks]
        expected_total = sum(expected_probs)
        expected_counts = [p / expected_total * sink.events_received for p in expected_probs]
        ax.loglog(ranks, expected_counts, 'r--', alpha=0.8, label=f'Expected (s={zipf_s})', linewidth=2)

        ax.set_xlabel("Rank (log scale)")
        ax.set_ylabel("Frequency (log scale)")
        ax.set_title("Zipf Distribution: Rank vs Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')

        # Plot 2: Cumulative distribution
        ax = axes[0, 1]
        cumulative = np.cumsum(observed_counts) / sum(observed_counts) * 100
        ax.plot(ranks, cumulative, 'b-', linewidth=2, label='Observed')

        expected_cumulative = np.cumsum(expected_counts) / sum(expected_counts) * 100
        ax.plot(ranks, expected_cumulative, 'r--', linewidth=2, label='Expected')

        ax.axhline(y=80, color='gray', linestyle=':', alpha=0.7)
        ax.axvline(x=num_customers * 0.2, color='gray', linestyle=':', alpha=0.7)
        ax.set_xlabel("Customer Rank")
        ax.set_ylabel("Cumulative % of Requests")
        ax.set_title("Cumulative Distribution (80/20 Rule)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Top 20 customers bar chart
        ax = axes[1, 0]
        top_20 = freq_dist[:20]
        customer_ids = [str(cid) for cid, _ in top_20]
        counts = [count for _, count in top_20]
        bars = ax.bar(range(len(top_20)), counts, color='steelblue', alpha=0.8)
        ax.set_xticks(range(len(top_20)))
        ax.set_xticklabels(customer_ids, rotation=45, ha='right')
        ax.set_xlabel("Customer ID")
        ax.set_ylabel("Request Count")
        ax.set_title("Top 20 Customers by Request Volume")
        ax.grid(True, alpha=0.3, axis='y')

        # Add percentage labels on bars
        total = sink.events_received
        for i, (bar, count) in enumerate(zip(bars, counts)):
            pct = count / total * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)

        # Plot 4: Distribution comparison (histogram of frequencies)
        ax = axes[1, 1]
        ax.hist(observed_counts, bins=30, alpha=0.6, color='steelblue', label='Observed', edgecolor='black')
        ax.set_xlabel("Request Count per Customer")
        ax.set_ylabel("Number of Customers")
        ax.set_title("Distribution of Request Counts")
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        fig.suptitle(f"Zipf Distribution Verification (s={zipf_s}, n={num_customers}, events={sink.events_received})",
                     fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig.savefig(test_output_dir / "zipf_distribution_verification.png", dpi=150)
        plt.close(fig)

        # Write summary statistics to file
        with open(test_output_dir / "zipf_statistics.txt", "w") as f:
            f.write(f"Zipf Distribution Test Summary\n")
            f.write(f"==============================\n\n")
            f.write(f"Configuration:\n")
            f.write(f"  Number of customers: {num_customers}\n")
            f.write(f"  Zipf exponent (s): {zipf_s}\n")
            f.write(f"  Events generated: {sink.events_received}\n\n")
            f.write(f"Results:\n")
            f.write(f"  Top 10% customers: {top_10_pct:.1f}% of traffic\n")
            f.write(f"  Top 20% customers: {sink.get_top_n_percentage('customer_id', 20):.1f}% of traffic\n")
            f.write(f"  Rank 1 customer: {freq_dist[0][1]} requests ({freq_dist[0][1]/sink.events_received*100:.1f}%)\n")
            f.write(f"  Rank 1/2 ratio: {freq_dist[0][1]/freq_dist[1][1]:.2f} (expected ~2.0)\n\n")
            f.write(f"Top 10 customers:\n")
            for i, (cid, count) in enumerate(freq_dist[:10], 1):
                f.write(f"  {i}. Customer {cid}: {count} requests ({count/sink.events_received*100:.2f}%)\n")

    def test_compare_zipf_parameters(self, test_output_dir: Path):
        """Compare different Zipf s parameters side by side.

        Generates visualization showing how different s values affect
        the distribution skew.
        """
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        num_customers = 100
        num_events = 5000
        s_values = [0.0, 0.5, 1.0, 1.5, 2.0]
        results = {}

        for s in s_values:
            sink = StatisticsCollectorSink(f"sink_s{s}", tracked_fields=["customer_id"])
            customer_dist = ZipfDistribution(range(num_customers), s=s, seed=42)

            provider = DistributedFieldProvider(
                target=sink,
                event_type="Request",
                field_distributions={"customer_id": customer_dist},
                stop_after=Instant.from_seconds(num_events / 1000.0),
            )

            source = Source(
                name=f"Source_s{s}",
                event_provider=provider,
                arrival_time_provider=ConstantArrivalTimeProvider(
                    ConstantRateProfile(rate=1000.0),
                    start_time=Instant.Epoch,
                ),
            )

            sim = Simulation(
                start_time=Instant.Epoch,
                duration=num_events / 1000.0 + 1.0,
                sources=[source],
                entities=[sink],
            )
            sim.run()

            results[s] = {
                "sink": sink,
                "top_10_pct": sink.get_top_n_percentage("customer_id", 10),
                "top_20_pct": sink.get_top_n_percentage("customer_id", 20),
                "freq_dist": sink.get_frequency_distribution("customer_id"),
            }

        # === ASSERTIONS ===
        # Higher s should produce more skew
        assert results[1.0]["top_10_pct"] > results[0.5]["top_10_pct"]
        assert results[1.5]["top_10_pct"] > results[1.0]["top_10_pct"]

        # === VISUALIZATION ===
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Rank-frequency for all s values
        ax = axes[0, 0]
        colors = plt.cm.viridis([i / len(s_values) for i in range(len(s_values))])
        for (s, data), color in zip(results.items(), colors):
            freq_dist = data["freq_dist"]
            counts = [count for _, count in freq_dist]
            ranks = range(1, len(counts) + 1)
            label = f"s={s}" if s > 0 else "s=0 (uniform)"
            ax.loglog(ranks, counts, 'o-', alpha=0.7, label=label, color=color, markersize=3)

        ax.set_xlabel("Rank (log scale)")
        ax.set_ylabel("Frequency (log scale)")
        ax.set_title("Effect of Zipf Exponent on Rank-Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')

        # Plot 2: Top 10% traffic share
        ax = axes[0, 1]
        s_labels = [f"s={s}" for s in s_values]
        top_10_values = [results[s]["top_10_pct"] for s in s_values]
        bars = ax.bar(s_labels, top_10_values, color='steelblue', alpha=0.8)
        ax.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='Uniform (10%)')
        ax.set_xlabel("Zipf Exponent")
        ax.set_ylabel("% of Traffic to Top 10 Customers")
        ax.set_title("Traffic Concentration vs Zipf Exponent")
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, top_10_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.1f}%', ha='center', va='bottom')

        # Plot 3: Cumulative distributions
        ax = axes[1, 0]
        for (s, data), color in zip(results.items(), colors):
            freq_dist = data["freq_dist"]
            counts = [count for _, count in freq_dist]
            cumulative = [sum(counts[:i+1])/sum(counts)*100 for i in range(len(counts))]
            label = f"s={s}" if s > 0 else "s=0 (uniform)"
            ax.plot(range(1, len(cumulative)+1), cumulative, '-', alpha=0.8, label=label, color=color, linewidth=2)

        ax.axhline(y=80, color='gray', linestyle=':', alpha=0.7)
        ax.set_xlabel("Number of Top Customers")
        ax.set_ylabel("Cumulative % of Traffic")
        ax.set_title("Cumulative Traffic Share")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Summary table as text
        ax = axes[1, 1]
        ax.axis('off')
        table_data = [
            ["s value", "Top 10%", "Top 20%", "Rank 1 %"],
        ]
        for s in s_values:
            data = results[s]
            rank1_pct = data["freq_dist"][0][1] / data["sink"].events_received * 100
            table_data.append([
                f"{s:.1f}",
                f"{data['top_10_pct']:.1f}%",
                f"{data['top_20_pct']:.1f}%",
                f"{rank1_pct:.1f}%",
            ])

        table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                         colWidths=[0.2, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.8)
        ax.set_title("Summary Statistics", pad=20)

        fig.suptitle(f"Comparing Zipf Exponent Values (n={num_customers} customers)",
                     fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig.savefig(test_output_dir / "zipf_parameter_comparison.png", dpi=150)
        plt.close(fig)

    def test_zipf_vs_uniform_hotspot_behavior(self, test_output_dir: Path):
        """Compare how Zipf vs Uniform distributions create hotspots.

        Simulates cache/database access patterns and shows how Zipf
        naturally creates "hot keys" vs uniform distribution.
        """
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        num_keys = 1000
        num_accesses = 20000
        seed = 42

        # Run with multiple distributions
        distributions = {
            "Uniform": UniformDistribution(range(num_keys), seed=seed),
            "Zipf (s=1.0)": ZipfDistribution(range(num_keys), s=1.0, seed=seed),
            "Zipf (s=1.5)": ZipfDistribution(range(num_keys), s=1.5, seed=seed),
        }

        results = {}
        for name, dist in distributions.items():
            sink = StatisticsCollectorSink(name, tracked_fields=["key_id"])

            provider = DistributedFieldProvider(
                target=sink,
                event_type="Access",
                field_distributions={"key_id": dist},
                stop_after=Instant.from_seconds(num_accesses / 1000.0),
            )

            source = Source(
                name=f"Source_{name}",
                event_provider=provider,
                arrival_time_provider=ConstantArrivalTimeProvider(
                    ConstantRateProfile(rate=1000.0),
                    start_time=Instant.Epoch,
                ),
            )

            sim = Simulation(
                start_time=Instant.Epoch,
                duration=num_accesses / 1000.0 + 1.0,
                sources=[source],
                entities=[sink],
            )
            sim.run()

            freq_dist = sink.get_frequency_distribution("key_id")
            counts = [count for _, count in freq_dist]
            avg_count = sink.events_received / num_keys

            results[name] = {
                "counts": counts,
                "hot_keys": sum(1 for c in counts if c > avg_count * 5),  # >5x average
                "cold_keys": sum(1 for c in counts if c < avg_count * 0.2),  # <0.2x average
                "max_count": max(counts) if counts else 0,
                "top_1_pct": sink.get_top_n_percentage("key_id", num_keys // 100),
                "events": sink.events_received,
            }

        # === ASSERTIONS ===
        # Zipf should have more hot keys than uniform
        assert results["Zipf (s=1.0)"]["hot_keys"] > results["Uniform"]["hot_keys"]
        assert results["Zipf (s=1.5)"]["top_1_pct"] > results["Zipf (s=1.0)"]["top_1_pct"]

        # === VISUALIZATION ===
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        avg_count = num_accesses / num_keys  # Expected average

        for ax, (name, data) in zip(axes, results.items()):
            counts = sorted(data["counts"], reverse=True)
            ax.bar(range(len(counts)), counts, width=1.0, alpha=0.7)
            ax.axhline(y=avg_count, color='red', linestyle='--',
                       label=f'Average ({avg_count:.0f})')
            ax.set_xlabel("Key Rank")
            ax.set_ylabel("Access Count")
            ax.set_title(f"{name}\nHot keys: {data['hot_keys']}, Cold keys: {data['cold_keys']}")
            ax.legend()
            ax.set_xlim(0, 200)  # Show first 200 keys

        fig.suptitle(f"Hot Key Distribution: Zipf vs Uniform ({num_keys} keys, {num_accesses} accesses)",
                     fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig.savefig(test_output_dir / "zipf_hotspot_comparison.png", dpi=150)
        plt.close(fig)


class TestMultiFieldDistribution:
    """Tests for multiple distributed fields."""

    def test_multi_field_independence(self, test_output_dir: Path):
        """Verify multiple fields are sampled independently."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sink = StatisticsCollectorSink("sink", tracked_fields=["customer_id", "region"])

        provider = DistributedFieldProvider(
            target=sink,
            event_type="Request",
            field_distributions={
                "customer_id": ZipfDistribution(range(100), s=1.0, seed=42),
                "region": UniformDistribution(["us-east", "us-west", "eu-west"], seed=123),
            },
            stop_after=Instant.from_seconds(10.0),
        )

        source = Source(
            name="MultiFieldSource",
            event_provider=provider,
            arrival_time_provider=ConstantArrivalTimeProvider(
                ConstantRateProfile(rate=1000.0),
                start_time=Instant.Epoch,
            ),
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=11.0,
            sources=[source],
            entities=[sink],
        )
        sim.run()

        # === ASSERTIONS ===
        # Customer IDs should follow Zipf
        customer_dist = sink.get_frequency_distribution("customer_id")
        top_10_customer = sink.get_top_n_percentage("customer_id", 10)
        assert top_10_customer > 40  # Zipf characteristic

        # Regions should be approximately uniform
        region_counts = sink.field_counts["region"]
        total_regions = sum(region_counts.values())
        for region in ["us-east", "us-west", "eu-west"]:
            pct = region_counts[region] / total_regions * 100
            assert 28 < pct < 38  # Each region ~33% (with tolerance)

        # === VISUALIZATION ===
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Customer ID distribution
        ax = axes[0]
        customer_counts = [count for _, count in customer_dist[:30]]
        ax.bar(range(len(customer_counts)), customer_counts, color='steelblue', alpha=0.8)
        ax.set_xlabel("Customer Rank")
        ax.set_ylabel("Request Count")
        ax.set_title(f"Customer Distribution (Zipf s=1.0)\nTop 10%: {top_10_customer:.1f}% of traffic")
        ax.grid(True, alpha=0.3, axis='y')

        # Region distribution
        ax = axes[1]
        regions = list(region_counts.keys())
        counts = [region_counts[r] for r in regions]
        bars = ax.bar(regions, counts, color='forestgreen', alpha=0.8)
        ax.set_xlabel("Region")
        ax.set_ylabel("Request Count")
        ax.set_title("Region Distribution (Uniform)")
        ax.grid(True, alpha=0.3, axis='y')

        for bar, count in zip(bars, counts):
            pct = count / total_regions * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{pct:.1f}%', ha='center', va='bottom')

        fig.suptitle(f"Multi-Field Distribution Test ({sink.events_received} events)",
                     fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig.savefig(test_output_dir / "multi_field_distribution.png", dpi=150)
        plt.close(fig)
