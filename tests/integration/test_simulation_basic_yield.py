from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant
from happysimulator.load.profile import Profile
from happysimulator.load.source import Source
from happysimulator.core.simulation import Simulation


class SideEffectCounterEntity(Entity):
    def __init__(self):
        super().__init__("sideeffectcounter")
        self.counter = 0

    def handle_event(self, event):
        self.counter += 1


class PingCounterEntity(Entity):
    def __init__(self, side_effect_counter: Entity):
        super().__init__("pingcounter")
        self.side_effect_counter = side_effect_counter
        self.first_counter = 0
        self.second_counter  = 0

    def handle_event(self, event: Event):
        self.first_counter += 1
        yield 1, None
        self.second_counter += 1

        # Yields a side effect (now) to be handled by the SideEffectCounterEntity
        yield 1, Event(time=self.now, event_type="Ping", target=self.side_effect_counter)
        return []


class ConstantOneProfile(Profile):
    """Returns a rate of 1.0 event per second."""
    def get_rate(self, time: Instant) -> float:
        if time <= Instant.from_seconds(60):
            return 1.0
        else:
            return 0


# --- 2. The Test Case ---

def test_basic_constant_simulation():
    """
    Verifies that a Simulation with a single Constant Source (1 event/sec)
    runs for exactly 60 seconds and generates roughly 60 events.
    """
    # A. CONFIGURATION

    # Setup the counter entities
    side_effect_counter = SideEffectCounterEntity()
    source_event_counter = PingCounterEntity(side_effect_counter)

    # Create the Source using the custom profile that drops rate to 0 after t=60
    profile = ConstantOneProfile()
    source = Source.with_profile(
        profile=profile, target=source_event_counter, event_type="Ping",
        poisson=False, name="PingSource",
    )

    # B. INITIALIZATION
    sim = Simulation(
        sources=[source],
        entities=[source_event_counter])

    # C. EXECUTION
    # Run the simulation
    sim.run()

    # D. ASSERTIONS
    # Note: With the discontinuous rate profile (1.0 for t<=60, 0 for t>60),
    # numerical integration can produce 61 events due to boundary handling.
    # This matches test_simulation_basic_counter.py which also expects 61.
    assert source_event_counter.first_counter == 61, \
        f"Expected a count of 61 in the first counter, but there were {source_event_counter.first_counter}"

    assert source_event_counter.second_counter == 61, \
        f"Expected a count of 61 in the second counter, but there were {source_event_counter.second_counter}"

    assert side_effect_counter.counter == 61, \
        f"Expected a count of 61 in the side effect counter, but there were {source_event_counter.second_counter}"
