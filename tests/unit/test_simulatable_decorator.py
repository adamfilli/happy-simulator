"""Tests for the @simulatable decorator."""

import pytest

from happysimulator.core.clock import Clock
from happysimulator.core.decorators import simulatable
from happysimulator.core.event import Event
from happysimulator.core.protocols import Simulatable
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant


class TestSimulatableDecorator:
    """Test that @simulatable enables duck-typed simulation participation."""

    def test_decorator_adds_set_clock(self):
        """Decorated class gets set_clock method."""

        @simulatable
        class Counter:
            def __init__(self, name: str):
                self.name = name

            def handle_event(self, event):
                return None

        c = Counter("test")
        assert hasattr(c, "set_clock")
        assert callable(c.set_clock)

    def test_decorator_adds_now_property(self):
        """Decorated class gets now property."""

        @simulatable
        class Counter:
            def __init__(self, name: str):
                self.name = name

            def handle_event(self, event):
                return None

        c = Counter("test")
        assert hasattr(type(c), "now")

    def test_now_raises_before_clock_injection(self):
        """Accessing now before simulation raises RuntimeError."""

        @simulatable
        class Counter:
            def __init__(self, name: str):
                self.name = name

            def handle_event(self, event):
                return None

        c = Counter("test")
        with pytest.raises(RuntimeError, match="not attached to a simulation"):
            _ = c.now

    def test_now_works_after_clock_injection(self):
        """Accessing now after set_clock works."""

        @simulatable
        class Counter:
            def __init__(self, name: str):
                self.name = name

            def handle_event(self, event):
                return None

        c = Counter("test")
        clock = Clock(Instant.from_seconds(5.0))
        c.set_clock(clock)

        assert c.now == Instant.from_seconds(5.0)

    def test_satisfies_simulatable_protocol(self):
        """Decorated class is recognized as Simulatable."""

        @simulatable
        class Counter:
            def __init__(self, name: str):
                self.name = name

            def handle_event(self, event):
                return None

        c = Counter("test")
        assert isinstance(c, Simulatable)

    def test_has_capacity_default(self):
        """Decorated class gets default has_capacity returning True."""

        @simulatable
        class Counter:
            def __init__(self, name: str):
                self.name = name

            def handle_event(self, event):
                return None

        c = Counter("test")
        assert c.has_capacity() is True

    def test_has_capacity_can_be_overridden(self):
        """User-defined has_capacity is preserved."""

        @simulatable
        class LimitedServer:
            def __init__(self, name: str):
                self.name = name
                self.busy = False

            def has_capacity(self) -> bool:
                return not self.busy

            def handle_event(self, event):
                return None

        s = LimitedServer("server")
        assert s.has_capacity() is True
        s.busy = True
        assert s.has_capacity() is False


class TestSimulatableInSimulation:
    """Test decorated classes work in actual simulations."""

    def test_simple_counter_simulation(self):
        """A decorated counter can count events in a simulation."""

        @simulatable
        class Counter:
            def __init__(self, name: str):
                self.name = name
                self.count = 0

            def handle_event(self, event):
                self.count += 1
                return

        counter = Counter("my-counter")

        # Schedule 3 events targeting the counter
        events = [
            Event(time=Instant.from_seconds(0.1), event_type="tick", target=counter),
            Event(time=Instant.from_seconds(0.2), event_type="tick", target=counter),
            Event(time=Instant.from_seconds(0.3), event_type="tick", target=counter),
        ]

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            entities=[counter],
        )
        for e in events:
            sim.schedule(e)

        sim.run()

        assert counter.count == 3

    def test_decorated_class_can_access_now(self):
        """Decorated class can use self.now during event handling."""

        @simulatable
        class TimeRecorder:
            def __init__(self, name: str):
                self.name = name
                self.recorded_times: list[Instant] = []

            def handle_event(self, event):
                self.recorded_times.append(self.now)
                return

        recorder = TimeRecorder("recorder")

        events = [
            Event(time=Instant.from_seconds(0.5), event_type="ping", target=recorder),
            Event(time=Instant.from_seconds(1.5), event_type="ping", target=recorder),
        ]

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            entities=[recorder],
        )
        for e in events:
            sim.schedule(e)

        sim.run()

        assert recorder.recorded_times == [
            Instant.from_seconds(0.5),
            Instant.from_seconds(1.5),
        ]

    def test_decorated_class_can_return_events(self):
        """Decorated class can schedule follow-up events."""

        @simulatable
        class PingPong:
            def __init__(self, name: str):
                self.name = name
                self.hits = 0

            def handle_event(self, event):
                self.hits += 1
                # Stop after 3 hits
                if self.hits >= 3:
                    return None
                # Schedule another hit 0.1s later
                return Event(
                    time=self.now + 0.1,
                    event_type="bounce",
                    target=self,
                )

        ball = PingPong("ball")

        # Start with one event
        start = Event(time=Instant.from_seconds(0.1), event_type="serve", target=ball)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            entities=[ball],
        )
        sim.schedule(start)
        sim.run()

        assert ball.hits == 3

    def test_decorated_class_with_generator(self):
        """Decorated class can use generator-style multi-step processing."""

        @simulatable
        class SlowProcessor:
            def __init__(self, name: str):
                self.name = name
                self.started_at: Instant | None = None
                self.finished_at: Instant | None = None

            def handle_event(self, event):
                self.started_at = self.now
                yield 0.5  # Wait 500ms
                self.finished_at = self.now

        processor = SlowProcessor("slow")

        event = Event(
            time=Instant.from_seconds(1.0),
            event_type="process",
            target=processor,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(5.0),
            entities=[processor],
        )
        sim.schedule(event)
        sim.run()

        assert processor.started_at == Instant.from_seconds(1.0)
        assert processor.finished_at == Instant.from_seconds(1.5)
