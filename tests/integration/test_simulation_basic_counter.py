from typing import List

from happysimulator.core.entity import Entity
from happysimulator.load.providers.constant_arrival import ConstantArrivalTimeProvider
from happysimulator.load.event_provider import EventProvider
from happysimulator.load.source import Source
from happysimulator.core.temporal import Instant
from happysimulator.core.event import Event
from happysimulator.load.profile import Profile
from happysimulator.core.simulation import Simulation  # Assuming your generic Simulation class

class PingCounterEntity(Entity):
    def __init__(self, name):
        super().__init__(name)
        self.counter = 1
    
    def handle_event(self, event: Event):
        self.counter += 1

class PingEvent(Event):
    """A simple ping event with no entity to be invoked."""
    def __init__(self, time: Instant, counter: PingCounterEntity):
        super().__init__(time=time, event_type="Ping", target=counter, callback=None)

class ConstantOneProfile(Profile):
    """Returns a rate of 1.0 event per second."""
    def get_rate(self, time: Instant) -> float:
        return 1.0

class PingProvider(EventProvider):
    def __init__(self, counter: PingCounterEntity):
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
    sim_duration = 60.0
    end_time = Instant.from_seconds(sim_duration)
    
    # Setup the counter entities
    counter = PingCounterEntity("pingcounter")
    
    # Setup the Source components
    profile = ConstantOneProfile()
    provider = PingProvider(counter)
    
    arrival_time_provider = ConstantArrivalTimeProvider(profile, start_time=Instant.Epoch)
    
    # Create the Source (Rate=1, Distribution=Constant)
    # This ensures exactly 1 event every 1.0s (t=1.0, t=2.0, etc.)
    source = Source(
        name="PingSource",
        event_provider=provider,
        arrival_time_provider=arrival_time_provider
    )

    # B. INITIALIZATION
    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=end_time,
        sources=[source],
        entities=[counter]
    )

    # C. EXECUTION
    # Run the simulation
    sim.run()

    # D. ASSERTIONS
    
    # 1. Verify the Source generated the expected number of events
    # We expect events at t=1, 2, ... 60. 
    assert source._nmb_generated == 61, \
        f"Expected 60 events, but source generated {source._nmb_generated}"
        
    assert counter.counter == 61, \
        f"Expected a count of 61 in the event counter, but there were {counter.counter}"