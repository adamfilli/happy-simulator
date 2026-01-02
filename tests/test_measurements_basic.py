from typing import List

from happysimulator.entities.entity import Entity
from happysimulator.load.constant_arrival_time_provider import ConstantArrivalTimeProvider
from happysimulator.load.event_provider import EventProvider
from happysimulator.load.source import Source
from happysimulator.utils.instant import Instant
from happysimulator.events.event import Event
from happysimulator.load.profile import Profile
from happysimulator.simulation import Simulation

class ConcurrencyTrackerEntity(Entity):
    def __init__(self):
        super().__init__("concurrencytracker")
        self.concurrency = 0
        self.first_counter = 0
        self.second_counter  = 0
    
    def handle_event(self, event: Event):
        self.first_counter += 1
        self.concurrency += 1
        yield 1, None
        self.second_counter += 1
        self.concurrency -= 1
        return []

class PingEvent(Event):
    def __init__(self, time: Instant, counter: Entity):
        super().__init__(time, "Ping", counter, None)

class ConstantOneProfile(Profile):
    """Returns a rate of 1.0 event per second."""
    def get_rate(self, time: Instant) -> float:
        if time <= Instant.from_seconds(60):
            return 1.0
        else:
            return 0

class PingProvider(EventProvider):
    def __init__(self, counter: ConcurrencyTrackerEntity):
        super().__init__()
        self.counter = counter
    
    def get_events(self, time: Instant) -> List[Event]:
        return [PingEvent(time, self.counter)]

# --- 2. The Test Case ---

def test_basic_constant_simulation():
    """
    Verifies that a Simulation with a single Constant Source (1 event/sec)
    runs for exactly 60 seconds and generates roughly 60 events.
    """
    # A. CONFIGURATION
    
    # Setup the counter entities
    source_event_counter = ConcurrencyTrackerEntity()
    
    # Setup the Source components
    profile = ConstantOneProfile()
    provider = PingProvider(source_event_counter)
    
    arrival_time_provider = ConstantArrivalTimeProvider(profile, Instant.Epoch)
    
    # Create the Source (Rate=1, Distribution=Constant)
    # This ensures exactly 1 event every 1.0s (t=1.0, t=2.0, etc.)
    source = Source(
        name="PingSource",
        event_provider=provider,
        arrival_time_provider=arrival_time_provider
    )

    # B. INITIALIZATION
    sim = Simulation(
        sources=[source],
        entities=[source_event_counter])

    # C. EXECUTION
    # Run the simulation
    sim.run()

    # D. ASSERTIONS