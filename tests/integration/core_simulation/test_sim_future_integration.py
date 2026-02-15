"""Integration tests for SimFuture with full simulation execution.

Tests cover: basic resolve, pre-resolved futures, request-response
patterns, timeout races (any_of), quorum waits (all_of), and chaining.
"""

from collections.abc import Generator

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.sim_future import SimFuture, all_of, any_of
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant

# ---------------------------------------------------------------------------
# Helper entities
# ---------------------------------------------------------------------------


class FutureResolver(Entity):
    """Entity that resolves a future from event context after a delay."""

    def __init__(self, name: str = "Resolver"):
        super().__init__(name)
        self.handled = 0

    def handle_event(self, event: Event) -> Generator:
        self.handled += 1
        delay = event.context.get("delay", 0.1)
        yield delay
        future: SimFuture = event.context["future"]
        value = event.context.get("value", "resolved")
        future.resolve(value)
        return []


class RequestResponseClient(Entity):
    """Client that sends a request with a future and waits for response."""

    def __init__(self, name: str, server: Entity):
        super().__init__(name)
        self.server = server
        self.response_value = None
        self.completed = False

    def handle_event(self, event: Event) -> Generator:
        future = SimFuture()
        yield (
            0.0,
            [
                Event(
                    time=self.now,
                    event_type="Request",
                    target=self.server,
                    context={"future": future, "delay": 0.5, "value": {"status": "ok"}},
                )
            ],
        )
        self.response_value = yield future
        self.completed = True
        return []


def _make_sim(*entities):
    """Create a Simulation with clock injection for the given entities."""
    return Simulation(
        duration=60,
        entities=list(entities),
    )


def _trigger(sim, target, event_type="Go", time_s=0.0, **extra_context):
    """Schedule a trigger event into the simulation."""
    sim.schedule(
        Event(
            time=Instant.from_seconds(time_s),
            event_type=event_type,
            target=target,
            context=extra_context,
        )
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSimFutureBasicResolve:
    """Test basic resolve flow through the simulation."""

    def test_resolve_resumes_generator(self):
        resolver = FutureResolver()
        client = RequestResponseClient("Client", resolver)

        sim = _make_sim(client, resolver)
        _trigger(sim, client)
        sim.run()

        assert client.completed
        assert client.response_value == {"status": "ok"}

    def test_resolve_value_is_none_by_default(self):
        """resolve() with no args sends None into the generator."""

        class NoneResolver(Entity):
            def __init__(self):
                super().__init__("NoneResolver")

            def handle_event(self, event: Event) -> Generator:
                future: SimFuture = event.context["future"]
                yield 0.1
                future.resolve()
                return []

        class NoneClient(Entity):
            def __init__(self, resolver: Entity):
                super().__init__("NoneClient")
                self.resolver = resolver
                self.result = "NOT_SET"
                self.completed = False

            def handle_event(self, event: Event) -> Generator:
                future = SimFuture()
                yield (
                    0.0,
                    [
                        Event(
                            time=self.now,
                            event_type="Req",
                            target=self.resolver,
                            context={"future": future},
                        )
                    ],
                )
                self.result = yield future
                self.completed = True
                return []

        resolver = NoneResolver()
        client = NoneClient(resolver)
        sim = _make_sim(client, resolver)
        _trigger(sim, client)
        sim.run()

        assert client.completed
        assert client.result is None


class TestPreResolvedFuture:
    """Test yielding a future that was already resolved."""

    def test_pre_resolved_resumes_immediately(self):
        class PreResolved(Entity):
            def __init__(self):
                super().__init__("PreResolved")
                self.result = None
                self.completed = False

            def handle_event(self, event: Event) -> Generator:
                future = SimFuture()
                future.resolve(99)
                self.result = yield future
                self.completed = True
                return []

        client = PreResolved()
        sim = _make_sim(client)
        _trigger(sim, client)
        sim.run()

        assert client.completed
        assert client.result == 99


class TestAnyOfIntegration:
    """Test any_of in a full simulation (timeout race patterns)."""

    def test_response_wins_race(self):
        """Server responds in 0.1s, timeout at 1.0s — response wins."""
        resolver = FutureResolver()

        class RaceClient(Entity):
            def __init__(self):
                super().__init__("RaceClient")
                self.result_index = None
                self.result_value = None
                self.completed = False

            def handle_event(self, event: Event) -> Generator:
                response_future = SimFuture()
                timeout_future = SimFuture()
                yield (
                    0.0,
                    [
                        Event(
                            time=self.now,
                            event_type="Request",
                            target=resolver,
                            context={"future": response_future, "delay": 0.1, "value": "response"},
                        ),
                        Event.once(
                            time=Instant.from_seconds(self.now.to_seconds() + 1.0),
                            event_type="Timeout",
                            fn=lambda e: timeout_future.resolve("timeout"),
                        ),
                    ],
                )
                idx, value = yield any_of(response_future, timeout_future)
                self.result_index = idx
                self.result_value = value
                self.completed = True
                return []

        client = RaceClient()
        sim = _make_sim(client, resolver)
        _trigger(sim, client)
        sim.run()

        assert client.completed
        assert client.result_index == 0
        assert client.result_value == "response"

    def test_timeout_wins_race(self):
        """Server responds in 2.0s, timeout at 0.5s — timeout wins."""
        resolver = FutureResolver()

        class RaceClient(Entity):
            def __init__(self):
                super().__init__("RaceClient")
                self.result_index = None
                self.result_value = None
                self.completed = False

            def handle_event(self, event: Event) -> Generator:
                response_future = SimFuture()
                timeout_future = SimFuture()
                yield (
                    0.0,
                    [
                        Event(
                            time=self.now,
                            event_type="Request",
                            target=resolver,
                            context={"future": response_future, "delay": 2.0, "value": "response"},
                        ),
                        Event.once(
                            time=Instant.from_seconds(self.now.to_seconds() + 0.5),
                            event_type="Timeout",
                            fn=lambda e: timeout_future.resolve("timeout"),
                        ),
                    ],
                )
                idx, value = yield any_of(response_future, timeout_future)
                self.result_index = idx
                self.result_value = value
                self.completed = True
                return []

        client = RaceClient()
        sim = _make_sim(client, resolver)
        _trigger(sim, client)
        sim.run()

        assert client.completed
        assert client.result_index == 1
        assert client.result_value == "timeout"


class TestAllOfIntegration:
    """Test all_of in a full simulation (quorum patterns)."""

    def test_waits_for_all_replicas(self):
        replicas = [FutureResolver(f"Replica-{i}") for i in range(3)]

        class Client(Entity):
            def __init__(self):
                super().__init__("QuorumClient")
                self.results = None
                self.completed = False

            def handle_event(self, event: Event) -> Generator:
                futures = []
                side_effects = []
                for i, replica in enumerate(replicas):
                    f = SimFuture()
                    futures.append(f)
                    side_effects.append(
                        Event(
                            time=self.now,
                            event_type="Write",
                            target=replica,
                            context={"future": f, "delay": 0.1 * (i + 1), "value": f"ack-{i}"},
                        )
                    )
                yield 0.0, side_effects
                self.results = yield all_of(*futures)
                self.completed = True
                return []

        client = Client()
        sim = _make_sim(client, *replicas)
        _trigger(sim, client)
        sim.run()

        assert client.completed
        assert client.results == ["ack-0", "ack-1", "ack-2"]


class TestMultiStepFuture:
    """Test mixing delays and futures in the same generator."""

    def test_interleaved_delays_and_futures(self):
        resolver = FutureResolver()

        class Client(Entity):
            def __init__(self):
                super().__init__("MultiStep")
                self.steps = []
                self.completed = False

            def handle_event(self, event: Event) -> Generator:
                yield 0.1
                self.steps.append("after_delay_1")

                future1 = SimFuture()
                yield (
                    0.0,
                    [
                        Event(
                            time=self.now,
                            event_type="Req",
                            target=resolver,
                            context={"future": future1, "delay": 0.2, "value": "resp1"},
                        )
                    ],
                )
                resp1 = yield future1
                self.steps.append(f"after_future_1:{resp1}")

                yield 0.05
                self.steps.append("after_delay_2")

                future2 = SimFuture()
                yield (
                    0.0,
                    [
                        Event(
                            time=self.now,
                            event_type="Req",
                            target=resolver,
                            context={"future": future2, "delay": 0.1, "value": "resp2"},
                        )
                    ],
                )
                resp2 = yield future2
                self.steps.append(f"after_future_2:{resp2}")

                self.completed = True
                return []

        client = Client()
        sim = _make_sim(client, resolver)
        _trigger(sim, client)
        sim.run()

        assert client.completed
        assert client.steps == [
            "after_delay_1",
            "after_future_1:resp1",
            "after_delay_2",
            "after_future_2:resp2",
        ]


class TestCompletionHooksWithFuture:
    """Test that completion hooks fire after generator with futures finishes."""

    def test_completion_hook_fires(self):
        resolver = FutureResolver()
        hook_fired = []

        class Client(Entity):
            def __init__(self):
                super().__init__("HookedClient")
                self.completed = False

            def handle_event(self, event: Event) -> Generator:
                future = SimFuture()
                yield (
                    0.0,
                    [
                        Event(
                            time=self.now,
                            event_type="Req",
                            target=resolver,
                            context={"future": future, "delay": 0.1, "value": "done"},
                        )
                    ],
                )
                yield future
                self.completed = True
                return []

        client = Client()
        sim = _make_sim(client, resolver)

        trigger = Event(
            time=Instant.from_seconds(0),
            event_type="Go",
            target=client,
        )
        trigger.add_completion_hook(lambda t: hook_fired.append(t))
        sim.schedule(trigger)
        sim.run()

        assert client.completed
        assert len(hook_fired) == 1


class TestMultipleFutureResolves:
    """Test multiple events using futures in the same simulation."""

    def test_multiple_concurrent_futures(self):
        """Two clients with overlapping futures both resolve correctly."""
        resolver = FutureResolver()
        client1 = RequestResponseClient("Client1", resolver)
        client2 = RequestResponseClient("Client2", resolver)

        sim = _make_sim(client1, client2, resolver)
        _trigger(sim, client1, time_s=0.0)
        _trigger(sim, client2, time_s=0.0)
        sim.run()

        assert client1.completed
        assert client2.completed
        assert client1.response_value == {"status": "ok"}
        assert client2.response_value == {"status": "ok"}
