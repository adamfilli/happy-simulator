from happysimulator.components.common import Counter
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant
from happysimulator.load.source import Source


def test_basic_constant_simulation():
    """
    Verifies that a Simulation with a single Constant Source (1 event/sec)
    runs for exactly 60 seconds and generates exactly 60 events.
    """
    sim_duration = 60.0

    counter = Counter("pingcounter")

    source = Source.constant(rate=1, target=counter, event_type="Ping", name="PingSource")

    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=Instant.from_seconds(sim_duration),
        sources=[source],
        entities=[counter],
    )
    sim.run()

    # We expect events at t=1, 2, ... 60 → 60 events handled by the counter.
    # source._nmb_generated includes the Source's self-scheduling events.
    assert source._nmb_generated == 61, \
        f"Expected 61 generated, but source generated {source._nmb_generated}"

    assert counter.total == 60, \
        f"Expected 60 events counted, but there were {counter.total}"

    assert counter.by_type == {"Ping": 60}