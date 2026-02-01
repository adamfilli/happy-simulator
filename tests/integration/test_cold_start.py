"""
Integration tests for the cold start cache simulation example.

These tests verify:
1. Cache warmup behavior - hit rate improves over time
2. Cache reset behavior - hit rate drops after invalidation
3. Datastore load spikes after reset
4. Visualization file generation

Run with: pytest tests/integration/test_cold_start.py -v
"""

import pytest
from pathlib import Path

# Import the example module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "examples"))

from cold_start import (
    ColdStartConfig,
    CachedServer,
    LatencyTrackingSink,
    run_cold_start_simulation,
    visualize_results,
    compute_datastore_read_rate,
)

from happysimulator import (
    ConstantArrivalTimeProvider,
    ConstantRateProfile,
    Data,
    DistributedFieldProvider,
    Entity,
    Event,
    FIFOQueue,
    Instant,
    Probe,
    Simulation,
    Source,
    ZipfDistribution,
)


class TestCacheWarmup:
    """Tests verifying cache warmup behavior."""

    def test_cache_warms_up_over_time(self):
        """Hit rate should improve during the warmup phase."""
        config = ColdStartConfig(
            arrival_rate=100.0,
            num_customers=500,
            distribution_type="zipf",
            zipf_s=1.0,
            cache_capacity=50,
            cold_start_time_s=None,  # No reset
            duration_s=30.0,
            probe_interval_s=1.0,
            seed=42,
            use_poisson=False,  # Deterministic for testing
        )

        result = run_cold_start_simulation(config)

        # Extract hit rate data
        hr_values = [v for (t, v) in result.hit_rate_data.values]

        # There should be multiple samples
        assert len(hr_values) >= 10, f"Expected at least 10 samples, got {len(hr_values)}"

        # Hit rate should generally improve over time
        early_hr = sum(hr_values[:5]) / 5 if len(hr_values) >= 5 else hr_values[0]
        late_hr = sum(hr_values[-5:]) / 5 if len(hr_values) >= 5 else hr_values[-1]

        # The late hit rate should be higher than early hit rate
        # With Zipf distribution and small cache, we expect significant improvement
        assert late_hr > early_hr, (
            f"Expected hit rate to improve over time. "
            f"Early: {early_hr:.2%}, Late: {late_hr:.2%}"
        )

    def test_cache_fills_to_capacity(self):
        """Cache should fill up to its configured capacity."""
        config = ColdStartConfig(
            arrival_rate=100.0,
            num_customers=500,
            distribution_type="zipf",
            zipf_s=1.0,
            cache_capacity=50,
            cold_start_time_s=None,
            duration_s=20.0,
            probe_interval_s=1.0,
            seed=42,
            use_poisson=False,
        )

        result = run_cold_start_simulation(config)

        # Cache should reach capacity
        final_cache_size = result.server.cache_size
        assert final_cache_size == config.cache_capacity, (
            f"Expected cache to fill to capacity {config.cache_capacity}, "
            f"got {final_cache_size}"
        )


class TestCacheReset:
    """Tests verifying cache reset (cold start) behavior."""

    def test_cache_reset_drops_hit_rate(self):
        """Hit rate should drop after cache reset."""
        config = ColdStartConfig(
            arrival_rate=100.0,
            num_customers=500,
            distribution_type="zipf",
            zipf_s=1.0,
            cache_capacity=50,
            cold_start_time_s=15.0,  # Reset at 15s
            duration_s=30.0,
            probe_interval_s=1.0,
            seed=42,
            use_poisson=False,
        )

        result = run_cold_start_simulation(config)

        # Extract hit rate data
        hr_data = result.hit_rate_data.values

        # Find hit rate before and after reset
        before_reset = [v for (t, v) in hr_data if 10.0 <= t < 15.0]
        after_reset = [v for (t, v) in hr_data if 15.0 <= t < 20.0]

        assert len(before_reset) > 0, "No samples before reset"
        assert len(after_reset) > 0, "No samples after reset"

        avg_before = sum(before_reset) / len(before_reset)
        # First sample after reset should show the drop
        first_after = after_reset[0] if after_reset else 0

        # Hit rate should drop after reset
        # Note: windowed hit rate starts from 0 hits/0 misses, so early values may be 0
        assert first_after < avg_before or first_after < 0.3, (
            f"Expected hit rate to drop after reset. "
            f"Before: {avg_before:.2%}, First after: {first_after:.2%}"
        )

    def test_cache_size_drops_to_zero_on_reset(self):
        """Cache size should drop to zero when reset."""
        config = ColdStartConfig(
            arrival_rate=100.0,
            num_customers=500,
            distribution_type="zipf",
            zipf_s=1.0,
            cache_capacity=50,
            cold_start_time_s=15.0,
            duration_s=30.0,
            probe_interval_s=0.5,  # Finer granularity to catch the drop
            seed=42,
            use_poisson=False,
        )

        result = run_cold_start_simulation(config)

        # Extract cache size data
        cs_data = result.cache_size_data.values

        # Find cache size right before and right after reset
        before_reset = [v for (t, v) in cs_data if 14.0 <= t < 15.0]
        after_reset = [v for (t, v) in cs_data if 15.0 <= t < 16.0]

        assert len(before_reset) > 0, "No samples before reset"
        assert len(after_reset) > 0, "No samples after reset"

        # Cache should have been full before reset
        assert before_reset[-1] > 0, f"Cache was empty before reset: {before_reset[-1]}"

        # First sample after reset should show empty or nearly empty cache
        # (some requests may have already added entries)
        first_after = after_reset[0]
        assert first_after < before_reset[-1], (
            f"Cache size should have dropped after reset. "
            f"Before: {before_reset[-1]}, After: {first_after}"
        )


class TestDatastoreLoad:
    """Tests verifying datastore load patterns."""

    def test_datastore_load_spikes_after_reset(self):
        """Datastore read rate should spike after cache reset."""
        config = ColdStartConfig(
            arrival_rate=100.0,
            num_customers=500,
            distribution_type="zipf",
            zipf_s=1.0,
            cache_capacity=50,
            cold_start_time_s=15.0,
            duration_s=30.0,
            probe_interval_s=1.0,
            seed=42,
            use_poisson=False,
        )

        result = run_cold_start_simulation(config)

        # Compute datastore read rate
        dr_times, dr_rates = compute_datastore_read_rate(result.datastore_reads_data)

        # Find rates before and after reset
        before_reset = [r for (t, r) in zip(dr_times, dr_rates) if 10.0 <= t < 15.0]
        after_reset = [r for (t, r) in zip(dr_times, dr_rates) if 15.0 <= t < 20.0]

        if before_reset and after_reset:
            avg_before = sum(before_reset) / len(before_reset)
            max_after = max(after_reset)

            # Datastore load should spike after reset
            assert max_after > avg_before, (
                f"Expected datastore load to spike after reset. "
                f"Before avg: {avg_before:.1f}/s, After max: {max_after:.1f}/s"
            )

    def test_datastore_reads_increase_after_reset(self):
        """Total datastore reads should increase faster after reset."""
        config = ColdStartConfig(
            arrival_rate=100.0,
            num_customers=500,
            distribution_type="zipf",
            zipf_s=1.0,
            cache_capacity=50,
            cold_start_time_s=15.0,
            duration_s=30.0,
            probe_interval_s=1.0,
            seed=42,
            use_poisson=False,
        )

        result = run_cold_start_simulation(config)

        # Get datastore reads at key points
        dr_data = result.datastore_reads_data.values

        # Find reads before and after reset
        reads_at_10s = [v for (t, v) in dr_data if 9.5 <= t < 10.5]
        reads_at_15s = [v for (t, v) in dr_data if 14.5 <= t < 15.5]
        reads_at_20s = [v for (t, v) in dr_data if 19.5 <= t < 20.5]

        if reads_at_10s and reads_at_15s and reads_at_20s:
            # Calculate read rate in each period
            reads_10 = reads_at_10s[0]
            reads_15 = reads_at_15s[0]
            reads_20 = reads_at_20s[0]

            rate_before = (reads_15 - reads_10) / 5.0
            rate_after = (reads_20 - reads_15) / 5.0

            # Rate after reset should be higher (more cache misses)
            assert rate_after > rate_before, (
                f"Expected more datastore reads after reset. "
                f"Rate before: {rate_before:.1f}/s, Rate after: {rate_after:.1f}/s"
            )


class TestVisualization:
    """Tests verifying visualization generation."""

    def test_generates_visualization_files(self, test_output_dir: Path):
        """Running the example should generate PNG files."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")

        config = ColdStartConfig(
            arrival_rate=50.0,
            num_customers=200,
            distribution_type="zipf",
            zipf_s=1.0,
            cache_capacity=20,
            cold_start_time_s=10.0,
            duration_s=20.0,
            probe_interval_s=0.5,
            seed=42,
            use_poisson=False,
        )

        result = run_cold_start_simulation(config)
        visualize_results(result, test_output_dir)

        # Check that visualization files were created
        overview_path = test_output_dir / "cold_start_overview.png"

        assert overview_path.exists(), f"Overview plot not created: {overview_path}"

        # Verify files have content (not empty)
        assert overview_path.stat().st_size > 1000, "Overview plot file too small"


class TestDistributions:
    """Tests verifying distribution behavior."""

    def test_zipf_distribution_improves_hit_rate(self):
        """Zipf distribution should result in higher hit rate than uniform."""
        base_config = {
            "arrival_rate": 100.0,
            "num_customers": 500,
            "cache_capacity": 50,
            "cold_start_time_s": None,
            "duration_s": 20.0,
            "probe_interval_s": 1.0,
            "seed": 42,
            "use_poisson": False,
        }

        # Run with Zipf distribution
        zipf_config = ColdStartConfig(**base_config, distribution_type="zipf", zipf_s=1.0)
        zipf_result = run_cold_start_simulation(zipf_config)

        # Run with uniform distribution
        uniform_config = ColdStartConfig(**base_config, distribution_type="uniform", zipf_s=1.0)
        uniform_result = run_cold_start_simulation(uniform_config)

        # Compare final hit rates
        zipf_hr = zipf_result.server.hit_rate
        uniform_hr = uniform_result.server.hit_rate

        # Zipf should have higher hit rate due to skewed access pattern
        assert zipf_hr > uniform_hr, (
            f"Expected Zipf to have higher hit rate than uniform. "
            f"Zipf: {zipf_hr:.2%}, Uniform: {uniform_hr:.2%}"
        )


class TestArchitecture:
    """Tests verifying the architecture with separate datastore entity."""

    def test_datastore_is_separate_entity(self):
        """Datastore should be a separate entity from the server."""
        config = ColdStartConfig(
            arrival_rate=50.0,
            num_customers=100,
            cache_capacity=20,
            cold_start_time_s=None,
            duration_s=10.0,
            seed=42,
            use_poisson=False,
        )

        result = run_cold_start_simulation(config)

        # Datastore should be accessible separately
        assert result.datastore is not None, "Datastore should be in result"
        assert result.datastore.name == "Datastore", "Datastore should have its own name"

        # Datastore stats should be accessible
        assert result.datastore.stats.reads > 0, "Datastore should have recorded reads"

        # Server's datastore_reads should match external datastore stats
        assert result.server.datastore_reads == result.datastore.stats.reads, (
            "Server's datastore_reads should match external datastore stats"
        )


class TestLatencyTracking:
    """Tests verifying latency tracking."""

    def test_latency_is_recorded(self):
        """Sink should record latency for completed requests."""
        config = ColdStartConfig(
            arrival_rate=50.0,
            num_customers=100,
            cache_capacity=20,
            cold_start_time_s=None,
            duration_s=10.0,
            seed=42,
            use_poisson=False,
        )

        result = run_cold_start_simulation(config)

        # Should have recorded latencies
        assert result.sink.events_received > 0, "No events received by sink"
        assert len(result.sink.latencies_s) > 0, "No latencies recorded"
        assert len(result.sink.latencies_s) == result.sink.events_received

    def test_latency_values_are_reasonable(self):
        """Recorded latencies should be within expected bounds."""
        config = ColdStartConfig(
            arrival_rate=50.0,
            num_customers=100,
            cache_capacity=20,
            cache_read_latency_s=0.0001,
            ingress_latency_s=0.005,
            db_network_latency_s=0.002,
            datastore_read_latency_s=0.001,
            cold_start_time_s=None,
            duration_s=10.0,
            seed=42,
            use_poisson=False,
        )

        result = run_cold_start_simulation(config)

        min_latency = min(result.sink.latencies_s)
        max_latency = max(result.sink.latencies_s)

        # Minimum latency should be at least ingress + cache hit + processing
        min_expected = config.ingress_latency_s + config.cache_read_latency_s + 0.001
        assert min_latency >= min_expected * 0.9, (
            f"Min latency {min_latency*1000:.2f}ms is less than expected {min_expected*1000:.2f}ms"
        )

        # Maximum latency should be reasonable (not infinite)
        # Total datastore latency = db network RTT + datastore processing
        total_datastore_latency = config.db_network_latency_s + config.datastore_read_latency_s
        max_expected = config.ingress_latency_s + total_datastore_latency + 0.002
        # Allow some variance due to queuing
        assert max_latency < max_expected * 5, (
            f"Max latency {max_latency*1000:.2f}ms is unexpectedly high"
        )
