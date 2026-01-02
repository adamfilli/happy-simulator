import pytest
from typing import List

from happysimulator.entities.entity import Entity
from happysimulator.load.constant_arrival_time_provider import ConstantArrivalTimeProvider
from happysimulator.load.event_provider import EventProvider
from happysimulator.load.source import Source
from happysimulator.utils.instant import Instant
from happysimulator.events.event import Event
from happysimulator.load.profile import Profile
from happysimulator.simulation import Simulation  # Assuming your generic Simulation class


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
        yield 1, PingEvent(event.time, self.side_effect_counter)
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
    
    # Setup the counter entities
    side_effect_counter = SideEffectCounterEntity()
    source_event_counter = PingCounterEntity(side_effect_counter)
    
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
    
    # We expect events at t=1, 2, ... 60. 
        
    assert source_event_counter.first_counter == 60, \
        f"Expected a count of 60 in the first counter, but there were {source_event_counter.first_counter}"
    
    assert source_event_counter.second_counter == 60, \
        f"Expected a count of 59 in the second counter, but there were {source_event_counter.second_counter}"
        
    assert side_effect_counter.counter == 60, \
        f"Expected a count of 60 in the side effect counter, but there were {source_event_counter.second_counter}"