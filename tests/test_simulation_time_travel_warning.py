import logging

from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant


def test_simulation_logs_warning_on_time_travel(caplog):
    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=Instant.from_seconds(20),
        sources=[],
        entities=[],
    )

    def first_callback(_: Event):
        return Event.once(
            time=Instant.from_seconds(5),
            event_type="BackInTime",
            fn=lambda __: [],
        )

    sim.schedule(
        Event.once(
            time=Instant.from_seconds(10),
            event_type="First",
            fn=first_callback,
        )
    )

    with caplog.at_level(logging.WARNING, logger="happysimulator.simulation"):
        sim.run()

    assert any("Time travel detected" in rec.message for rec in caplog.records)
