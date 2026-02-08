from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant
from happysimulator.load.source import Source


class PingCounterEntity(Entity):
    def __init__(self, name):
        super().__init__(name)
        self.counter = 1

    def handle_event(self, event: Event):
        self.counter += 1


# --- 2. The Test Case ---

def test_basic_constant_simulation():
    """
    Verifies that a Simulation with a single Constant Source (1 event/sec)
    runs for exactly 60 seconds and generates roughly 60 events.
    """
    # A. CONFIGURATION
    sim_duration = 60.0

    # Setup the counter entity
    counter = PingCounterEntity("pingcounter")

    # Create the Source (Rate=1, Distribution=Constant)
    # This ensures exactly 1 event every 1.0s (t=1.0, t=2.0, etc.)
    source = Source.constant(rate=1, target=counter, event_type="Ping", name="PingSource")

    # B. INITIALIZATION
    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=Instant.from_seconds(sim_duration),
        sources=[source],
        entities=[counter]
    )

    # C. EXECUTION
    sim.run()

    # D. ASSERTIONS

    # We expect events at t=1, 2, ... 60 → 60 events.
    # Counter starts at 1 in __init__, so final = 1 + 60 = 61.
    assert source._nmb_generated == 61, \
        f"Expected 61 events, but source generated {source._nmb_generated}"

    assert counter.counter == 61, \
        f"Expected a count of 61 in the event counter, but there were {counter.counter}"