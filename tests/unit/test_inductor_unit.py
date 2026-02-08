"""Unit tests for the Inductor (digital inductor / EWMA burst suppression)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from happysimulator.components.common import Sink
from happysimulator.components.rate_limiter.inductor import Inductor, InductorStats
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant
from happysimulator.load.source import Source


class TestInductorFirstEvent:
    """The very first arrival should always be forwarded immediately."""

    def test_first_event_forwarded(self):
        sink = Sink()
        inductor = Inductor("ind", downstream=sink, time_constant=2.0)

        source = Source.constant(rate=1.0, target=inductor, name="src")

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.5),
            sources=[source],
            entities=[inductor, sink],
        )
        sim.run()

        assert inductor.stats.forwarded >= 1
        assert inductor.stats.dropped == 0


class TestInductorSteadyState:
    """A constant-rate stream should pass through after warmup."""

    def test_constant_rate_passes_through(self):
        sink = Sink()
        inductor = Inductor("ind", downstream=sink, time_constant=1.0)

        source = Source.constant(rate=10.0, target=inductor, name="src")

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(10.0),
            sources=[source],
            entities=[inductor, sink],
        )
        sim.run()

        # At steady state all events should eventually be forwarded
        assert inductor.stats.received > 0
        assert inductor.stats.forwarded > 0
        assert inductor.stats.dropped == 0
        # Most events should be forwarded (allowing some early queuing during warmup)
        assert inductor.stats.forwarded >= inductor.stats.received * 0.8


class TestInductorBurstQueuing:
    """Bursts should be queued, not dropped (assuming queue capacity)."""

    def test_burst_events_get_queued(self):
        sink = Sink()
        inductor = Inductor(
            "ind", downstream=sink, time_constant=2.0, queue_capacity=10_000,
        )

        # High rate to create burst behaviour
        source = Source.constant(rate=200.0, target=inductor, name="src")

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[source],
            entities=[inductor, sink],
        )
        sim.run()

        assert inductor.stats.queued > 0
        assert inductor.stats.dropped == 0


class TestInductorRateConvergence:
    """The EWMA rate estimate should converge to the actual arrival rate."""

    def test_rate_estimate_converges(self):
        sink = Sink()
        target_rate = 20.0
        inductor = Inductor("ind", downstream=sink, time_constant=1.0)

        source = Source.constant(rate=target_rate, target=inductor, name="src")

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(10.0),
            sources=[source],
            entities=[inductor, sink],
        )
        sim.run()

        assert len(inductor.rate_history) > 0
        # Last few rate estimates should be close to target_rate
        final_rates = [r for _, r in inductor.rate_history[-10:]]
        avg_final = sum(final_rates) / len(final_rates)
        assert abs(avg_final - target_rate) < target_rate * 0.15


class TestInductorTimeConstant:
    """Higher time constant should mean slower adaptation."""

    def test_higher_tau_slower_adaptation(self):
        sink_fast = Sink()
        sink_slow = Sink()

        inductor_fast = Inductor("fast", downstream=sink_fast, time_constant=0.5)
        inductor_slow = Inductor("slow", downstream=sink_slow, time_constant=5.0)

        source_fast = Source.constant(rate=50.0, target=inductor_fast, name="src_fast")
        source_slow = Source.constant(rate=50.0, target=inductor_slow, name="src_slow")

        sim_fast = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(3.0),
            sources=[source_fast],
            entities=[inductor_fast, sink_fast],
        )
        sim_fast.run()

        sim_slow = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(3.0),
            sources=[source_slow],
            entities=[inductor_slow, sink_slow],
        )
        sim_slow.run()

        # Both should receive the same number of events
        assert inductor_fast.stats.received > 0
        assert inductor_slow.stats.received > 0
        # The fast-adapting inductor should forward more events early
        # (it adapts to the rate quicker and stops queuing sooner)
        assert inductor_fast.stats.forwarded >= inductor_slow.stats.forwarded


class TestInductorQueueOverflow:
    """Queue overflow should cause drops (bounded queue)."""

    def test_queue_overflow_drops(self):
        sink = Sink()
        inductor = Inductor(
            "ind", downstream=sink, time_constant=10.0, queue_capacity=5,
        )

        # Low rate first so EWMA settles to a slow interval,
        # then a sudden burst overflows the tiny queue.
        from happysimulator.load.profile import Profile

        @dataclass(frozen=True)
        class _BurstProfile(Profile):
            def get_rate(self, time: Instant) -> float:
                return 2.0 if time.to_seconds() < 3.0 else 500.0

        profile = _BurstProfile()
        source = Source.with_profile(
            profile, target=inductor, poisson=False, name="src",
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(5.0),
            sources=[source],
            entities=[inductor, sink],
        )
        sim.run()

        assert inductor.stats.dropped > 0


class TestInductorInvariants:
    """Conservation law: received == forwarded + in_queue + dropped."""

    def test_received_equals_forwarded_plus_queued_plus_dropped(self):
        sink = Sink()
        inductor = Inductor(
            "ind", downstream=sink, time_constant=2.0, queue_capacity=50,
        )

        source = Source.constant(rate=100.0, target=inductor, name="src")

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(3.0),
            sources=[source],
            entities=[inductor, sink],
        )
        sim.run()

        total = inductor.stats.forwarded + inductor.queue_depth + inductor.stats.dropped
        assert inductor.stats.received == total

    def test_forwarded_matches_sink(self):
        sink = Sink()
        inductor = Inductor("ind", downstream=sink, time_constant=1.0)

        source = Source.constant(rate=10.0, target=inductor, name="src")

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(5.0),
            sources=[source],
            entities=[inductor, sink],
        )
        sim.run()

        assert abs(inductor.stats.forwarded - sink.events_received) <= 1


class TestInductorProperties:
    """Test observable properties."""

    def test_estimated_rate_zero_initially(self):
        sink = Sink()
        inductor = Inductor("ind", downstream=sink, time_constant=1.0)
        assert inductor.estimated_rate == 0.0

    def test_queue_depth_zero_initially(self):
        sink = Sink()
        inductor = Inductor("ind", downstream=sink, time_constant=1.0)
        assert inductor.queue_depth == 0

    def test_time_constant_property(self):
        sink = Sink()
        inductor = Inductor("ind", downstream=sink, time_constant=3.5)
        assert inductor.time_constant == 3.5

    def test_stats_dataclass_defaults(self):
        stats = InductorStats()
        assert stats.received == 0
        assert stats.forwarded == 0
        assert stats.queued == 0
        assert stats.dropped == 0
