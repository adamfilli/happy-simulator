"""Integration tests for the metric collection pipeline example.

Tests verify:
- All N tasks flow through the pipeline correctly
- Batch completion tracking works
- Rate limiters can be swapped in
- Deterministic behavior with constant latencies
"""

import sys
from pathlib import Path

import pytest

# Add examples to path for import
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "examples"))

from metric_collection_pipeline import (
    BatchJobProvider,
    BatchTracker,
    MetricExecutor,
    MetricSink,
    run_simulation,
)

from happysimulator import (
    ConstantArrivalTimeProvider,
    ConstantLatency,
    ConstantRateProfile,
    Entity,
    Event,
    FIFOQueue,
    Instant,
    Simulation,
    Source,
)
from happysimulator.components.rate_limiter import NullRateLimiter


class TestBatchTracker:
    """Tests for BatchTracker class."""

    def test_start_and_complete_batch(self):
        """Batch tracker records start and completion times."""
        tracker = BatchTracker()
        start_time = Instant.from_seconds(10.0)

        tracker.start_batch(batch_id=0, start_time=start_time, num_tasks=3)

        batch = tracker.get_batch(0)
        assert batch is not None
        assert batch.batch_id == 0
        assert batch.start_time == start_time
        assert batch.tasks_total == 3
        assert batch.tasks_completed == 0
        assert batch.end_time is None

    def test_task_completion_tracking(self):
        """Batch tracker counts task completions correctly."""
        tracker = BatchTracker()
        tracker.start_batch(batch_id=0, start_time=Instant.Epoch, num_tasks=3)

        # Complete first two tasks
        assert not tracker.task_completed(0, Instant.from_seconds(1.0))
        assert not tracker.task_completed(0, Instant.from_seconds(2.0))

        batch = tracker.get_batch(0)
        assert batch.tasks_completed == 2
        assert batch.end_time is None

        # Complete final task
        assert tracker.task_completed(0, Instant.from_seconds(3.0))
        assert batch.tasks_completed == 3
        assert batch.end_time == Instant.from_seconds(3.0)

    def test_batch_durations(self):
        """Batch durations are calculated correctly."""
        tracker = BatchTracker()

        tracker.start_batch(batch_id=0, start_time=Instant.from_seconds(0.0), num_tasks=1)
        tracker.task_completed(0, Instant.from_seconds(5.0))

        tracker.start_batch(batch_id=1, start_time=Instant.from_seconds(10.0), num_tasks=1)
        tracker.task_completed(1, Instant.from_seconds(12.5))

        durations = tracker.batch_durations_seconds()
        assert len(durations) == 2
        assert durations[0] == (0, 5.0)
        assert durations[1] == (1, 2.5)


class TestBatchJobProvider:
    """Tests for BatchJobProvider class."""

    def test_generates_correct_number_of_tasks(self):
        """Provider generates exactly N tasks per batch."""
        tracker = BatchTracker()
        target = _DummyTarget()
        provider = BatchJobProvider(
            target=target,
            tasks_per_batch=5,
            batch_tracker=tracker,
        )

        events = provider.get_events(Instant.from_seconds(0.0))
        assert len(events) == 5

        for i, event in enumerate(events):
            assert event.context["batch_id"] == 0
            assert event.context["task_id"] == i
            assert event.context["created_at"] == Instant.from_seconds(0.0)

    def test_increments_batch_id(self):
        """Provider increments batch ID for each call."""
        tracker = BatchTracker()
        target = _DummyTarget()
        provider = BatchJobProvider(
            target=target,
            tasks_per_batch=2,
            batch_tracker=tracker,
        )

        events1 = provider.get_events(Instant.from_seconds(0.0))
        events2 = provider.get_events(Instant.from_seconds(60.0))

        assert events1[0].context["batch_id"] == 0
        assert events2[0].context["batch_id"] == 1

    def test_respects_stop_after(self):
        """Provider stops generating events after stop_after time."""
        tracker = BatchTracker()
        target = _DummyTarget()
        provider = BatchJobProvider(
            target=target,
            tasks_per_batch=3,
            batch_tracker=tracker,
            stop_after=Instant.from_seconds(100.0),
        )

        events1 = provider.get_events(Instant.from_seconds(50.0))
        assert len(events1) == 3

        events2 = provider.get_events(Instant.from_seconds(150.0))
        assert len(events2) == 0


class TestMetricExecutor:
    """Tests for MetricExecutor class."""

    def test_has_capacity_respects_concurrency(self):
        """Executor respects concurrency limit."""
        sink = _DummyTarget()
        executor = MetricExecutor(
            name="TestExecutor",
            downstream=sink,
            concurrency=2,
            latency=ConstantLatency(0.1),
        )

        # Initially has capacity
        assert executor.has_capacity()
        assert executor.in_flight == 0

    def test_processes_tasks_with_latency(self):
        """Executor applies latency and forwards downstream."""
        tracker = BatchTracker()
        sink = MetricSink("Sink", tracker)
        executor = MetricExecutor(
            name="Executor",
            downstream=sink,
            concurrency=10,
            latency=ConstantLatency(0.1),
        )

        # Create a simple simulation to test
        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            entities=[executor, sink],
        )
        executor.set_clock(sim._clock)
        sink.set_clock(sim._clock)

        # Schedule a task
        tracker.start_batch(0, Instant.Epoch, 1)
        event = Event(
            time=Instant.Epoch,
            event_type="Task",
            target=executor,
            context={"batch_id": 0, "task_id": 0, "created_at": Instant.Epoch},
        )
        sim._event_heap.push(event)
        sim.run()

        # Verify task was processed
        assert executor.stats_processed == 1
        assert sink.tasks_received == 1


class TestNullRateLimiter:
    """Tests for NullRateLimiter class."""

    def test_forwards_all_events(self):
        """NullRateLimiter forwards all events to downstream."""
        downstream = _DummyTarget()
        limiter = NullRateLimiter("Limiter", downstream)

        event = Event(
            time=Instant.from_seconds(1.0),
            event_type="Test",
            target=limiter,
            context={"key": "value"},
        )

        result = limiter.handle_event(event)

        assert len(result) == 1
        assert result[0].target == downstream
        assert result[0].time == event.time
        assert result[0].event_type == event.event_type
        assert result[0].context == event.context


class TestFullPipeline:
    """Integration tests for the complete pipeline."""

    def test_deterministic_single_batch(self):
        """Single batch with deterministic latency completes correctly."""
        result = run_simulation(
            duration_s=60.0,
            batch_interval_s=60.0,
            tasks_per_batch=10,
            executor1_workers=10,
            executor1_latency_s=0.05,
            executor1_dist="constant",
            executor2_workers=10,
            executor2_latency_s=0.1,
            executor2_dist="constant",
            seed=42,
        )

        # One batch should complete
        assert result.total_tasks_generated == 10
        assert result.sink.tasks_received == 10
        assert result.executor1.stats_processed == 10
        assert result.executor2.stats_processed == 10

        # Batch should complete
        durations = result.batch_tracker.batch_durations_seconds()
        assert len(durations) == 1
        batch_id, duration = durations[0]
        assert batch_id == 0
        # With constant latency: 0.05s + 0.1s = 0.15s per task
        # With 10 workers and 10 tasks, should complete quickly
        assert duration < 1.0

    def test_multiple_batches(self):
        """Multiple batches complete within deadline."""
        result = run_simulation(
            duration_s=180.0,  # 3 batches at 60s intervals
            batch_interval_s=60.0,
            tasks_per_batch=50,
            executor1_workers=10,
            executor1_latency_s=0.05,
            executor1_dist="constant",
            executor2_workers=10,
            executor2_latency_s=0.1,
            executor2_dist="constant",
            seed=42,
        )

        # 3 batches should complete
        assert result.total_tasks_generated == 150
        assert result.sink.tasks_received == 150

        durations = result.batch_tracker.batch_durations_seconds()
        assert len(durations) == 3

        # All batches should complete within deadline (60s)
        for batch_id, duration in durations:
            assert duration < 60.0, f"Batch {batch_id} took {duration}s (over deadline)"

    def test_tasks_flow_in_order(self):
        """Tasks from same batch complete with consistent latencies."""
        result = run_simulation(
            duration_s=60.0,
            batch_interval_s=60.0,
            tasks_per_batch=5,
            executor1_workers=5,
            executor1_latency_s=0.1,
            executor1_dist="constant",
            executor2_workers=5,
            executor2_latency_s=0.2,
            executor2_dist="constant",
            seed=42,
        )

        assert result.sink.tasks_received == 5

        # With constant distribution, latencies should be deterministic
        # All 5 tasks enter Executor1 together, exit at same time (0.1s)
        # Then enter Executor2 queue - processed one at a time due to queue
        # Expected latencies are predictable based on queue order
        latencies = sorted(result.sink.latencies_s)
        min_latency = min(latencies)
        max_latency = max(latencies)

        # Minimum should be ~0.3s (0.1 + 0.2)
        assert 0.29 <= min_latency <= 0.31, f"Min latency {min_latency} unexpected"
        # With queue effects, max should be bounded
        assert max_latency < 1.0, f"Max latency {max_latency} unexpectedly high"

    def test_exponential_latency_varies(self):
        """Exponential distribution produces varying latencies."""
        result = run_simulation(
            duration_s=60.0,
            batch_interval_s=60.0,
            tasks_per_batch=100,
            executor1_workers=10,
            executor1_latency_s=0.1,
            executor1_dist="exponential",
            executor2_workers=10,
            executor2_latency_s=0.1,
            executor2_dist="exponential",
            seed=42,
        )

        latencies = result.sink.latencies_s
        assert len(latencies) == 100

        # Exponential should produce variation
        min_lat = min(latencies)
        max_lat = max(latencies)
        assert max_lat > min_lat * 2, "Exponential should have significant variation"

    def test_with_rate_limiting(self):
        """Pipeline works with rate limiting enabled."""
        result = run_simulation(
            duration_s=120.0,
            batch_interval_s=60.0,
            tasks_per_batch=20,
            executor1_workers=10,
            executor1_latency_s=0.05,
            executor1_dist="constant",
            executor2_workers=10,
            executor2_latency_s=0.1,
            executor2_dist="constant",
            rate_limit_stage1=50,  # 50 req/s
            rate_limit_stage2=50,
            seed=42,
        )

        # All tasks should still complete
        assert result.sink.tasks_received == 40  # 2 batches x 20 tasks

    def test_queue_builds_when_saturated(self):
        """Queue depth increases when workers are saturated."""
        result = run_simulation(
            duration_s=60.0,
            batch_interval_s=60.0,
            tasks_per_batch=100,  # Many tasks
            executor1_workers=2,   # Few workers
            executor1_latency_s=0.5,  # Long latency
            executor1_dist="constant",
            executor2_workers=10,
            executor2_latency_s=0.01,
            executor2_dist="constant",
            probe_interval_s=0.1,
            seed=42,
        )

        # Check queue depth was recorded
        assert len(result.queue1_depth_data.values) > 0

        # Queue should have built up at some point
        max_depth = max(v for _, v in result.queue1_depth_data.values)
        assert max_depth > 10, f"Expected queue buildup, max depth was {max_depth}"


class TestVisualization:
    """Tests for visualization (saves outputs for manual inspection)."""

    def test_generates_visualizations(self, test_output_dir):
        """Run simulation and generate visualizations."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")

        from metric_collection_pipeline import visualize_results

        result = run_simulation(
            duration_s=180.0,
            batch_interval_s=60.0,
            tasks_per_batch=50,
            executor1_workers=10,
            executor1_latency_s=0.05,
            executor1_dist="exponential",
            executor2_workers=10,
            executor2_latency_s=0.1,
            executor2_dist="exponential",
            seed=42,
        )

        visualize_results(result, test_output_dir)

        # Verify files were created
        assert (test_output_dir / "batch_completion_times.png").exists()
        assert (test_output_dir / "queue_depths.png").exists()
        assert (test_output_dir / "worker_utilization.png").exists()
        assert (test_output_dir / "latency_distribution.png").exists()
        assert (test_output_dir / "throughput.png").exists()


# =============================================================================
# Helper Classes
# =============================================================================


class _DummyTarget(Entity):
    """Dummy entity for testing that just accepts events."""

    def __init__(self):
        super().__init__("DummyTarget")
        self.received: list[Event] = []

    def handle_event(self, event: Event) -> list[Event]:
        self.received.append(event)
        return []
