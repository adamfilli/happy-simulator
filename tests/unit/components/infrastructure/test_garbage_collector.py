"""Unit tests for GarbageCollector."""

import pytest
import random

from happysimulator.components.infrastructure.garbage_collector import (
    GarbageCollector,
    GCStats,
    StopTheWorld,
    ConcurrentGC,
    GenerationalGC,
)
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant


class TestGCStrategies:
    def test_stop_the_world_pause(self):
        random.seed(42)
        strategy = StopTheWorld()
        pause = strategy.pause_duration_s(0.5)
        assert pause > 0
        assert strategy.name == "StopTheWorld"

    def test_stop_the_world_interval(self):
        strategy = StopTheWorld(interval_s=5.0)
        assert strategy.collection_interval_s() == 5.0

    def test_stop_the_world_pressure_increases_pause(self):
        random.seed(42)
        strategy = StopTheWorld()
        low = strategy.pause_duration_s(0.1)
        random.seed(42)
        high = strategy.pause_duration_s(0.9)
        assert high > low

    def test_concurrent_gc_pause(self):
        random.seed(42)
        strategy = ConcurrentGC()
        pause = strategy.pause_duration_s(0.5)
        assert pause > 0
        assert strategy.name == "ConcurrentGC"

    def test_concurrent_gc_shorter_than_stw(self):
        random.seed(42)
        stw = StopTheWorld()
        stw_pause = stw.pause_duration_s(0.5)
        random.seed(42)
        concurrent = ConcurrentGC()
        concurrent_pause = concurrent.pause_duration_s(0.5)
        assert concurrent_pause < stw_pause

    def test_generational_gc_minor(self):
        random.seed(42)
        strategy = GenerationalGC()
        pause = strategy.pause_duration_s(0.3)  # below major threshold
        assert pause > 0
        assert strategy.name == "GenerationalGC"

    def test_generational_gc_major_on_high_pressure(self):
        random.seed(42)
        strategy = GenerationalGC(major_threshold=0.75)
        minor_pause = strategy.pause_duration_s(0.3)
        random.seed(42)
        major_pause = strategy.pause_duration_s(0.8)
        assert major_pause > minor_pause

    def test_generational_gc_interval(self):
        strategy = GenerationalGC(minor_interval_s=0.5)
        assert strategy.collection_interval_s() == 0.5


class TestGarbageCollector:
    def _make_gc(self, **kwargs) -> tuple[GarbageCollector, Simulation]:
        gc = GarbageCollector("test_gc", **kwargs)
        sim = Simulation(
            start_time=Instant.from_seconds(0),
            end_time=Instant.from_seconds(100),
            entities=[gc],
        )
        return gc, sim

    def test_creation_defaults(self):
        gc = GarbageCollector("gc")
        assert gc.name == "gc"
        assert gc.collection_count == 0

    def test_stats_initial(self):
        gc, sim = self._make_gc()
        stats = gc.stats
        assert isinstance(stats, GCStats)
        assert stats.collections == 0
        assert stats.total_pause_s == 0.0
        assert stats.max_pause_s == 0.0
        assert stats.min_pause_s == 0.0
        assert stats.avg_pause_s == 0.0

    def test_pause_generator(self):
        random.seed(42)
        gc, sim = self._make_gc(strategy=StopTheWorld(), heap_pressure=0.5)
        gen = gc.pause()
        latency = next(gen)
        assert latency > 0
        try:
            next(gen)
        except StopIteration as e:
            assert e.value > 0  # returns pause duration

        assert gc.stats.collections == 1
        assert gc.stats.total_pause_s > 0

    def test_fixed_heap_pressure(self):
        random.seed(42)
        gc, sim = self._make_gc(heap_pressure=0.9)
        gc._do_collect()
        # The heap pressure function should return 0.9
        assert gc._heap_pressure() == 0.9

    def test_stats_track_min_max(self):
        random.seed(42)
        gc, sim = self._make_gc(strategy=StopTheWorld(), heap_pressure=0.5)
        gc._do_collect()
        gc._do_collect()
        gc._do_collect()

        stats = gc.stats
        assert stats.collections == 3
        assert stats.max_pause_s >= stats.min_pause_s
        assert stats.min_pause_s > 0

    def test_generational_tracks_minor_major(self):
        random.seed(42)
        gc, sim = self._make_gc(
            strategy=GenerationalGC(major_threshold=0.75),
            heap_pressure=0.5,
        )
        gc._do_collect()  # minor (0.5 < 0.75)
        assert gc.stats.minor_collections == 1
        assert gc.stats.major_collections == 0

    def test_generational_major_above_threshold(self):
        random.seed(42)
        gc, sim = self._make_gc(
            strategy=GenerationalGC(major_threshold=0.75),
            heap_pressure=0.8,
        )
        gc._do_collect()
        assert gc.stats.major_collections == 1

    def test_handle_event_gc_collect(self):
        random.seed(42)
        gc, sim = self._make_gc(strategy=ConcurrentGC(), heap_pressure=0.5)

        from happysimulator.core.event import Event
        event = Event(
            time=Instant.from_seconds(1),
            event_type="_gc_collect",
            target=gc,
        )
        gen = gc.handle_event(event)
        # Generator should yield pause
        latency = next(gen)
        assert latency > 0
        # Should return next scheduled event
        try:
            next(gen)
        except StopIteration as e:
            events = e.value
            assert len(events) == 1
            assert events[0].event_type == "_gc_collect"

    def test_repr(self):
        gc, sim = self._make_gc()
        assert "test_gc" in repr(gc)
        assert "GenerationalGC" in repr(gc)  # default strategy

    def test_prime_creates_event(self):
        gc, sim = self._make_gc()
        event = gc.prime()
        assert event.event_type == "_gc_collect"
        assert event.target is gc
