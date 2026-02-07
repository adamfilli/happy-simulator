"""Integration tests for load balancing examples.

These tests verify that the load balancing examples run correctly and
produce expected results, demonstrating the benefits of consistent hashing.
"""

import random
import sys
from pathlib import Path

import pytest

# Add examples to path for imports
examples_path = Path(__file__).parent.parent.parent / "examples" / "load-balancing"
sys.path.insert(0, str(examples_path))


class TestConsistentHashingBasics:
    """Tests for Scenario A: Consistent Hash vs Round Robin."""

    def test_consistent_hash_beats_round_robin(self):
        """Consistent hashing achieves higher hit rate than round robin."""
        from consistent_hashing_basics import run_comparison, BasicConfig

        config = BasicConfig(
            arrival_rate=200.0,
            num_customers=500,
            duration_s=20.0,
            num_servers=5,
            cache_capacity=50,
            cache_ttl_s=30.0,
            seed=42,
        )

        result = run_comparison(config)

        ch_rate = result.consistent_hash.final_metrics.aggregate_hit_rate
        rr_rate = result.round_robin.final_metrics.aggregate_hit_rate

        # Consistent hash should significantly outperform round robin
        assert ch_rate > rr_rate * 2, (
            f"Expected consistent hash hit rate ({ch_rate:.2%}) to be at least "
            f"2x round robin ({rr_rate:.2%})"
        )

        # Consistent hash should achieve reasonable hit rate
        # (depends on cache capacity vs unique customers)
        assert ch_rate > 0.3, (
            f"Expected consistent hash hit rate > 30%, got {ch_rate:.2%}"
        )

        # Round robin has low hit rate since customers are spread across servers
        # With N servers and sequential round robin, hit rate ≈ cache_capacity / num_customers
        # (since each server sees only 1/N of each customer's requests)
        assert rr_rate < 0.3, (
            f"Expected round robin hit rate < 30%, got {rr_rate:.2%}"
        )

    def test_consistent_hash_reduces_datastore_reads(self):
        """Consistent hashing reduces load on backing datastore."""
        from consistent_hashing_basics import run_comparison, BasicConfig

        config = BasicConfig(
            arrival_rate=200.0,
            num_customers=500,
            duration_s=15.0,
            num_servers=5,
            cache_capacity=50,
            cache_ttl_s=30.0,
            seed=123,
        )

        result = run_comparison(config)

        ch_reads = result.consistent_hash.datastore_reads
        rr_reads = result.round_robin.datastore_reads

        # Consistent hash should have significantly fewer datastore reads
        assert ch_reads < rr_reads * 0.7, (
            f"Expected consistent hash datastore reads ({ch_reads}) to be <70% "
            f"of round robin ({rr_reads})"
        )


class TestFleetChangeComparison:
    """Tests for Scenario B: Fleet Change Impact."""

    def test_consistent_hash_minimal_key_shift(self):
        """Consistent hashing shifts ~1/N keys on fleet change."""
        from fleet_change_comparison import run_comparison, FleetChangeConfig

        config = FleetChangeConfig(
            arrival_rate=200.0,
            num_customers=1000,
            duration_s=40.0,
            fleet_change_time_s=20.0,
            initial_servers=5,
            cache_capacity=100,
            cache_ttl_s=60.0,
            seed=42,
        )

        result = run_comparison(config)

        ch_shifted = result.consistent_hash.keys_shifted_pct
        mod_shifted = result.modulo_hash.keys_shifted_pct

        # Consistent hash should shift roughly 1/(N+1) keys
        expected_ch = 1 / (config.initial_servers + 1)
        assert 0.5 * expected_ch < ch_shifted < 2.0 * expected_ch, (
            f"Expected consistent hash to shift ~{expected_ch:.1%} keys, "
            f"got {ch_shifted:.1%}"
        )

        # Modulo hash should shift significantly more keys
        assert mod_shifted > ch_shifted * 2, (
            f"Expected modulo hash ({mod_shifted:.1%}) to shift more keys "
            f"than consistent hash ({ch_shifted:.1%})"
        )

    def test_modulo_hash_catastrophic_shift(self):
        """Modulo hashing causes massive key redistribution."""
        from fleet_change_comparison import run_comparison, FleetChangeConfig

        config = FleetChangeConfig(
            arrival_rate=200.0,
            num_customers=1000,
            duration_s=40.0,
            fleet_change_time_s=20.0,
            initial_servers=5,
            cache_capacity=100,
            cache_ttl_s=60.0,
            seed=42,
        )

        result = run_comparison(config)

        mod_shifted = result.modulo_hash.keys_shifted_pct

        # Modulo hash typically shifts 60-90% of keys when N changes
        assert mod_shifted > 0.5, (
            f"Expected modulo hash to shift >50% of keys, got {mod_shifted:.1%}"
        )


class TestVNodesAnalysis:
    """Tests for Scenario C: Virtual Nodes Analysis."""

    def test_more_vnodes_better_uniformity(self):
        """More virtual nodes leads to more uniform distribution."""
        from vnodes_analysis import analyze_distribution

        num_servers = 5
        keys = list(range(5000))

        # Low vnode count
        low_stats = analyze_distribution(1, num_servers, keys)

        # High vnode count
        high_stats = analyze_distribution(100, num_servers, keys)

        # Higher vnodes should have lower coefficient of variation
        assert high_stats.cov < low_stats.cov * 0.5, (
            f"Expected 100 vnodes (CoV={high_stats.cov:.3f}) to have <50% "
            f"the variation of 1 vnode (CoV={low_stats.cov:.3f})"
        )

        # Higher vnodes should have lower max/min ratio
        assert high_stats.max_min_ratio < low_stats.max_min_ratio, (
            f"Expected 100 vnodes (ratio={high_stats.max_min_ratio:.2f}) "
            f"to be more balanced than 1 vnode (ratio={low_stats.max_min_ratio:.2f})"
        )

    def test_100_vnodes_good_uniformity(self):
        """100 virtual nodes achieves good uniformity (CoV < 0.15)."""
        from vnodes_analysis import analyze_distribution

        num_servers = 5
        keys = list(range(10000))

        stats = analyze_distribution(100, num_servers, keys)

        assert stats.cov < 0.15, (
            f"Expected 100 vnodes to achieve CoV < 0.15, got {stats.cov:.3f}"
        )

        assert stats.max_min_ratio < 1.3, (
            f"Expected 100 vnodes to achieve max/min ratio < 1.3, "
            f"got {stats.max_min_ratio:.2f}"
        )


class TestZipfEffect:
    """Tests for Scenario D: Zipf Distribution Effect."""

    def test_zipf_causes_load_imbalance(self):
        """Zipf distribution causes higher load imbalance than uniform."""
        from zipf_effect import run_comparison, ZipfConfig

        config = ZipfConfig(
            arrival_rate=200.0,
            num_customers=500,
            duration_s=15.0,
            num_servers=5,
            cache_capacity=100,
            cache_ttl_s=60.0,
            zipf_s=1.5,
            seed=42,
        )

        result = run_comparison(config)

        # Zipf should have higher load imbalance than uniform
        assert result.zipf.load_imbalance > result.uniform.load_imbalance, (
            f"Expected Zipf load imbalance ({result.zipf.load_imbalance:.2f}) "
            f"to exceed uniform ({result.uniform.load_imbalance:.2f})"
        )

    def test_zipf_creates_hot_server(self):
        """Zipf distribution creates a 'hot' server with disproportionate load."""
        from zipf_effect import run_comparison, ZipfConfig

        config = ZipfConfig(
            arrival_rate=200.0,
            num_customers=500,
            duration_s=15.0,
            num_servers=5,
            cache_capacity=100,
            cache_ttl_s=60.0,
            zipf_s=1.5,
            seed=42,
        )

        result = run_comparison(config)

        expected_pct = 1 / config.num_servers

        # With Zipf, top server should handle more than fair share
        assert result.zipf.top_server_pct > expected_pct * 1.3, (
            f"Expected Zipf top server ({result.zipf.top_server_pct:.1%}) "
            f"to exceed fair share ({expected_pct:.1%}) by >30%"
        )

        # Uniform should be close to fair share
        assert 0.8 * expected_pct < result.uniform.top_server_pct < 1.3 * expected_pct, (
            f"Expected uniform top server ({result.uniform.top_server_pct:.1%}) "
            f"to be near fair share ({expected_pct:.1%})"
        )


class TestCachingServer:
    """Tests for the CachingServer component."""

    def test_caching_server_tracks_hits_misses(self):
        """CachingServer correctly tracks cache hits and misses."""
        from common import CachingServer

        from happysimulator import Instant, Event, Simulation
        from happysimulator.components.datastore.kv_store import KVStore

        datastore = KVStore(name="TestDatastore", read_latency=0.001)

        server = CachingServer(
            name="TestServer",
            server_id=0,
            datastore=datastore,
            cache_capacity=10,
            cache_ttl_s=60.0,
        )

        # Create events for the same customer (should hit cache after first)
        events = []
        for i in range(5):
            events.append(
                Event(
                    time=Instant.from_seconds(i * 0.1),
                    event_type="Request",
                    target=server,
                    context={"customer_id": 1, "metadata": {"customer_id": 1}},
                )
            )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            entities=[server],
        )
        for event in events:
            sim.schedule(event)

        sim.run()

        # First request is a miss, subsequent are hits
        assert server.stats.cache_misses >= 1
        assert server.stats.cache_hits >= 1
        assert server.stats.requests_processed == 5
        assert server.hit_rate > 0.5  # Most requests should hit

    def test_caching_server_different_customers(self):
        """CachingServer has misses for different customers."""
        from common import CachingServer

        from happysimulator import Instant, Event, Simulation
        from happysimulator.components.datastore.kv_store import KVStore

        datastore = KVStore(name="TestDatastore", read_latency=0.001)

        server = CachingServer(
            name="TestServer",
            server_id=0,
            datastore=datastore,
            cache_capacity=10,
            cache_ttl_s=60.0,
        )

        # Create events for different customers (all misses)
        events = []
        for i in range(5):
            events.append(
                Event(
                    time=Instant.from_seconds(i * 0.1),
                    event_type="Request",
                    target=server,
                    context={"customer_id": i, "metadata": {"customer_id": i}},
                )
            )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            entities=[server],
        )
        for event in events:
            sim.schedule(event)

        sim.run()

        # All different customers = all misses
        assert server.stats.cache_misses == 5
        assert server.stats.cache_hits == 0
        assert server.hit_rate == 0.0


class TestCustomerRequestProvider:
    """Tests for the CustomerRequestProvider component."""

    def test_provider_creates_events_with_metadata(self):
        """CustomerRequestProvider creates events with correct metadata structure."""
        from common import CustomerRequestProvider

        from happysimulator import Instant, Entity, Event
        from happysimulator.distributions.uniform import UniformDistribution

        class DummyTarget(Entity):
            def __init__(self):
                super().__init__("Dummy")
                self.received: list[Event] = []

            def handle_event(self, event: Event) -> list[Event]:
                self.received.append(event)
                return []

        target = DummyTarget()
        dist = UniformDistribution([1, 2, 3])

        provider = CustomerRequestProvider(
            target=target,
            customer_distribution=dist,
        )

        events = provider.get_events(Instant.from_seconds(1.0))

        assert len(events) == 1
        event = events[0]

        # Check event structure
        assert event.target == target
        assert event.event_type == "Request"

        # Check context has customer_id in both locations
        assert "customer_id" in event.context
        assert "metadata" in event.context
        assert "customer_id" in event.context["metadata"]

        # Values should match
        assert event.context["customer_id"] == event.context["metadata"]["customer_id"]

    def test_provider_respects_stop_after(self):
        """CustomerRequestProvider stops generating after stop_after time."""
        from common import CustomerRequestProvider

        from happysimulator import Instant, Entity
        from happysimulator.distributions.uniform import UniformDistribution

        class DummyTarget(Entity):
            def __init__(self):
                super().__init__("Dummy")

            def handle_event(self, event):
                return []

        target = DummyTarget()
        dist = UniformDistribution([1, 2, 3])

        provider = CustomerRequestProvider(
            target=target,
            customer_distribution=dist,
            stop_after=Instant.from_seconds(5.0),
        )

        # Before stop_after
        events_before = provider.get_events(Instant.from_seconds(4.0))
        assert len(events_before) == 1

        # After stop_after
        events_after = provider.get_events(Instant.from_seconds(6.0))
        assert len(events_after) == 0
