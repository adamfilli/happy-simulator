"""Integration tests for GC-induced metastable collapse.

These tests verify that:
1. GC pauses exceeding client timeout cause metastable collapse with retries
2. Without retries, the system recovers between GC events

Run:
    pytest tests/integration/test_gc_caused_collapse.py -v

Output:
    test_output/test_gc_caused_collapse/<test_name>/
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add examples directory to path for imports
examples_dir = Path(__file__).parent.parent.parent / "examples"
sys.path.insert(0, str(examples_dir))

from gc_caused_collapse import (
    ComparisonResult,
    GCServer,
    RetryingClientWithStats,
    get_final_queue_depth,
    run_comparison,
    run_gc_collapse_simulation,
    visualize_results,
)


class TestGCCausedCollapse:
    """Tests for GC-induced metastable collapse behavior."""

    def test_gc_server_pauses_during_gc(self):
        """Verify that GCServer correctly calculates GC pause timing."""
        from happysimulator import Entity, Event, FIFOQueue, Instant, Simulation

        # Create a minimal GCServer
        server = GCServer(
            name="TestServer",
            service_time_s=0.01,  # Fast for testing
            gc_interval_s=1.0,
            gc_duration_s=0.1,
            gc_start_time_s=0.5,
            downstream=None,
        )

        # Manually set clock to test _get_gc_pause_remaining
        class MockClock:
            def __init__(self, time_s: float):
                self._time = Instant.from_seconds(time_s)

            @property
            def now(self) -> Instant:
                return self._time

        # Before first GC - no pause
        server.set_clock(MockClock(0.3))
        assert server._get_gc_pause_remaining() == 0.0

        # During first GC (at 0.5s) - should have pause remaining
        server.set_clock(MockClock(0.52))
        remaining = server._get_gc_pause_remaining()
        assert remaining > 0.0
        assert remaining <= 0.1  # GC duration

        # After first GC - no pause
        server.set_clock(MockClock(0.65))
        assert server._get_gc_pause_remaining() == 0.0

        # During second GC (at 1.5s)
        server.set_clock(MockClock(1.52))
        remaining = server._get_gc_pause_remaining()
        assert remaining > 0.0
        assert remaining <= 0.1

    def test_retrying_client_tracks_attempts(self):
        """Verify that RetryingClientWithStats correctly tracks retry attempts."""
        from happysimulator import Entity, Event, Instant

        class DummyServer(Entity):
            def __init__(self):
                super().__init__("DummyServer")

            def handle_event(self, event):
                return []

        server = DummyServer()
        client = RetryingClientWithStats(
            name="TestClient",
            server=server,
            timeout_s=0.1,
            max_retries=3,
            retry_enabled=True,
        )

        # Simulate clock
        class MockClock:
            def __init__(self):
                self._time = Instant.Epoch

            @property
            def now(self) -> Instant:
                return self._time

        client.set_clock(MockClock())

        # Send a new request
        event = Event(
            time=Instant.Epoch,
            event_type="NewRequest",
            target=client,
            context={"request_id": 1},
        )

        result_events = client.handle_event(event)

        assert client.stats_requests_received == 1
        assert client.stats_attempts_sent == 1
        assert len(result_events) == 2  # Request to server + timeout
        assert result_events[0].event_type == "Request"
        assert result_events[1].event_type == "Timeout"

    def test_collapse_occurs_with_retries(self, test_output_dir):
        """
        Test that GC pauses cause retry amplification when retries are enabled.

        With retries:
        - GC pause causes some in-flight requests to timeout
        - Retries create load amplification (>1x)
        - Timeouts accumulate during GC windows
        """
        result = run_gc_collapse_simulation(
            duration_s=60.0,  # Shorter for faster test
            drain_s=5.0,
            arrival_rate=7.0,
            service_time_s=0.1,
            timeout_s=0.2,
            max_retries=3,
            retry_delay_s=0.05,
            gc_interval_s=10.0,
            gc_duration_s=0.5,
            gc_start_time_s=10.0,
            retry_enabled=True,
            seed=42,
        )

        client = result.client

        # Verify retry behavior
        # 1. Some retry amplification (GC causes some timeouts that get retried)
        amplification = client.stats_attempts_sent / max(
            1, client.stats_requests_received
        )
        assert amplification > 1.0, (
            f"Expected retry amplification > 1.0, got {amplification:.2f}"
        )

        # 2. Timeouts occur (due to GC pauses)
        assert client.stats_timeouts > 20, (
            f"Expected > 20 timeouts from GC pauses, got {client.stats_timeouts}"
        )

        # 3. Retries were attempted
        assert client.stats_retries > 0, (
            f"Expected some retries, got {client.stats_retries}"
        )

    def test_recovery_without_retries(self, test_output_dir):
        """
        Test that system recovers between GC events when retries are disabled.

        Without retries:
        - GC pause causes requests to timeout (fail permanently)
        - No retry storm occurs
        - Queue drains normally between GC events
        - System maintains steady state
        """
        result = run_gc_collapse_simulation(
            duration_s=60.0,
            drain_s=5.0,
            arrival_rate=7.0,
            service_time_s=0.1,
            timeout_s=0.2,
            max_retries=3,
            retry_delay_s=0.05,
            gc_interval_s=10.0,
            gc_duration_s=0.5,
            gc_start_time_s=10.0,
            retry_enabled=False,
            seed=42,
        )

        client = result.client
        final_queue = get_final_queue_depth(result.queue_depth_data)

        # Verify recovery indicators
        # 1. No retry amplification
        amplification = client.stats_attempts_sent / max(
            1, client.stats_requests_received
        )
        assert amplification == 1.0, (
            f"Expected no retry amplification (1.0), got {amplification:.2f}"
        )

        # 2. No retries
        assert client.stats_retries == 0, (
            f"Expected 0 retries, got {client.stats_retries}"
        )

        # 3. Queue should be empty or near-empty at the end
        assert final_queue < 10, (
            f"Expected final queue depth < 10, got {final_queue}"
        )

        # 4. Reasonable success rate (some requests timeout during GC)
        # With 500ms GC every 10s, ~5% of time is in GC, but requests queued
        # during GC may also timeout, so ~80%+ success rate is expected
        success_rate = client.stats_completions / max(1, client.stats_requests_received)
        assert success_rate > 0.75, (
            f"Expected success rate > 75%, got {success_rate*100:.1f}%"
        )

    def test_comparison_shows_divergent_behavior(self, test_output_dir):
        """
        Test that comparing with/without retries shows clear divergence.

        This test runs both scenarios and verifies that:
        - With retries: retry amplification > 1 (retries occur)
        - Without retries: no amplification
        """
        result = run_comparison(
            duration_s=60.0,
            drain_s=5.0,
            arrival_rate=7.0,
            service_time_s=0.1,
            timeout_s=0.2,
            max_retries=3,
            gc_interval_s=10.0,
            gc_duration_s=0.5,
            gc_start_time_s=10.0,
            seed=42,
        )

        with_r = result.with_retries
        without_r = result.without_retries

        # Calculate metrics
        amp_with = with_r.client.stats_attempts_sent / max(
            1, with_r.client.stats_requests_received
        )
        amp_without = without_r.client.stats_attempts_sent / max(
            1, without_r.client.stats_requests_received
        )

        # Verify divergent behavior
        # With retries should have amplification > 1
        assert amp_with > amp_without, (
            f"With retries should have higher amplification: {amp_with:.2f} vs {amp_without:.2f}"
        )

        # Without retries should have no amplification
        assert amp_without == 1.0, (
            f"Without retries should have no amplification, got {amp_without:.2f}"
        )

        # With retries should show some amplification
        assert amp_with > 1.0, (
            f"With retries should show amplification > 1.0, got {amp_with:.2f}"
        )

        # With retries should have retries
        assert with_r.client.stats_retries > 0, (
            "With retries should have some retries"
        )

        # Without retries should have no retries
        assert without_r.client.stats_retries == 0, (
            "Without retries should have 0 retries"
        )

    def test_visualization_output(self, test_output_dir):
        """
        Test that visualization functions produce output files.
        """
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")

        result = run_comparison(
            duration_s=30.0,  # Short duration for fast test
            drain_s=3.0,
            seed=42,
        )

        visualize_results(result, test_output_dir)

        # Check that visualization files were created
        overview_path = test_output_dir / "gc_collapse_overview.png"
        analysis_path = test_output_dir / "gc_collapse_analysis.png"

        assert overview_path.exists(), f"Overview plot not created at {overview_path}"
        assert analysis_path.exists(), f"Analysis plot not created at {analysis_path}"

        # Check file sizes are reasonable (not empty)
        assert overview_path.stat().st_size > 10000, "Overview plot seems too small"
        assert analysis_path.stat().st_size > 10000, "Analysis plot seems too small"

    def test_gc_events_are_recorded(self):
        """Test that GC events are properly recorded for visualization."""
        result = run_gc_collapse_simulation(
            duration_s=30.0,
            gc_interval_s=10.0,
            gc_duration_s=0.5,
            gc_start_time_s=10.0,
            retry_enabled=True,
            seed=42,
        )

        gc_events = result.server.gc_events

        # Should have GC events recorded (at least 2 in 30s starting at 10s)
        assert len(gc_events) >= 2, f"Expected >= 2 GC events, got {len(gc_events)}"

        # Verify GC timing
        for gc_start, gc_end in gc_events:
            # Duration should be approximately gc_duration_s
            duration = gc_end - gc_start
            assert abs(duration - 0.5) < 0.01, (
                f"GC duration should be ~0.5s, got {duration}"
            )

    def test_deterministic_with_seed(self):
        """Test that results are deterministic with the same seed."""
        result1 = run_gc_collapse_simulation(
            duration_s=20.0,
            seed=12345,
            retry_enabled=True,
        )

        result2 = run_gc_collapse_simulation(
            duration_s=20.0,
            seed=12345,
            retry_enabled=True,
        )

        # Results should be identical
        assert result1.client.stats_completions == result2.client.stats_completions
        assert result1.client.stats_timeouts == result2.client.stats_timeouts
        assert result1.client.stats_attempts_sent == result2.client.stats_attempts_sent


class TestGCCollapseEdgeCases:
    """Edge case tests for GC collapse behavior."""

    def test_zero_retries_same_as_disabled(self):
        """Test that max_retries=0 behaves same as retry_enabled=False."""
        result_disabled = run_gc_collapse_simulation(
            duration_s=20.0,
            max_retries=3,
            retry_enabled=False,
            seed=42,
        )

        result_zero = run_gc_collapse_simulation(
            duration_s=20.0,
            max_retries=0,
            retry_enabled=True,
            seed=42,
        )

        # Both should have no retries
        assert result_disabled.client.stats_retries == 0
        assert result_zero.client.stats_retries == 0

    def test_no_gc_no_collapse(self):
        """Test that without GC, system remains stable even with retries."""
        result = run_gc_collapse_simulation(
            duration_s=30.0,
            arrival_rate=7.0,
            gc_start_time_s=1000.0,  # GC starts after simulation ends
            retry_enabled=True,
            seed=42,
        )

        client = result.client
        final_queue = get_final_queue_depth(result.queue_depth_data)

        # Without GC, should have very few timeouts and stable queue
        # Some timeouts may occur due to random service time variation
        assert client.stats_timeouts < 50, (
            f"Expected few timeouts without GC, got {client.stats_timeouts}"
        )

        # Queue should be near-empty
        assert final_queue < 20, (
            f"Expected low queue depth without GC, got {final_queue}"
        )

    def test_short_gc_no_collapse(self):
        """Test that GC shorter than timeout doesn't cause collapse."""
        result = run_gc_collapse_simulation(
            duration_s=30.0,
            timeout_s=0.5,  # 500ms timeout
            gc_duration_s=0.1,  # 100ms GC (shorter than timeout)
            retry_enabled=True,
            seed=42,
        )

        client = result.client

        # With GC shorter than timeout, should have minimal timeouts
        # Some may still occur due to service time + queue wait
        timeout_rate = client.stats_timeouts / max(1, client.stats_attempts_sent)
        assert timeout_rate < 0.3, (
            f"Expected low timeout rate with short GC, got {timeout_rate*100:.1f}%"
        )
