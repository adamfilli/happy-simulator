"""Periodic metric measurement using Source-based polling.

A Probe is a specialized Source that periodically samples a metric from
a target entity and stores the results in a Data container. Probes run
as daemon events, so they do not block auto-termination.

The metric is accessed via reflection (getattr), supporting both
attributes and callable properties.
"""

import logging

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant
from happysimulator.instrumentation.data import Data
from happysimulator.load.event_provider import EventProvider
from happysimulator.load.profile import Profile
from happysimulator.load.providers.constant_arrival import ConstantArrivalTimeProvider
from happysimulator.load.source import Source

logger = logging.getLogger(__name__)


class _ProbeProfile(Profile):
    """Internal profile providing constant rate for probe polling interval."""

    def __init__(self, interval_seconds: float):
        if interval_seconds <= 0:
            raise ValueError("Probe interval must be positive.")
        self.rate = 1.0 / interval_seconds
        self._interval = interval_seconds

    def get_rate(self, time: Instant) -> float:
        return self.rate


class _ProbeEventProvider(EventProvider):
    """Internal provider that creates measurement callback events."""

    def __init__(self, target: Entity, metric: str, data_sink: Data):
        super().__init__()
        self.target = target
        self.metric = metric
        self.data_sink = data_sink

    def _create_measurement_callback(self):
        """Create a callback that samples the metric and stores it."""
        target = self.target
        metric = self.metric
        data_sink = self.data_sink

        def measure_callback(event: Event) -> list[Event]:
            val = 0.0
            if hasattr(target, metric):
                raw_val = getattr(target, metric)
                if callable(raw_val):
                    val = raw_val()
                else:
                    val = raw_val
            else:
                logger.warning("Probe target '%s' has no attribute '%s'", target.name, metric)
            data_sink.add_stat(val, event.time)
            logger.debug("Probe sampled %s.%s = %s at %r", target.name, metric, val, event.time)
            return []

        return measure_callback

    def get_events(self, time: Instant) -> list[Event]:
        callback = self._create_measurement_callback()
        return [
            Event.once(
                time=time,
                event_type="probe_event",
                fn=callback,
                daemon=True,
            )
        ]


class Probe(Source):
    """Periodic metric sampler for monitoring entity state over time.

    Extends Source to poll a metric from a target entity at fixed intervals.
    The sampled values are stored in a Data container for post-simulation
    analysis or visualization.

    Probes run as daemon events, meaning they do not prevent the simulation
    from auto-terminating when all primary events are processed.

    Args:
        target: The entity to measure.
        metric: Attribute or property name to sample (accessed via getattr).
        data: Data container to store samples.
        interval: Seconds between measurements. Defaults to 1.0.
        start_time: When to begin probing. Defaults to Instant.Epoch.
    """

    def __init__(
        self,
        target: Entity,
        metric: str,
        data: Data,
        interval: float = 1.0,
        start_time: Instant | None = None,
    ):
        self.target = target
        self.metric = metric
        self.data_sink = data

        if start_time is None:
            start_time = Instant.Epoch

        profile = _ProbeProfile(interval)
        provider = _ProbeEventProvider(target, metric, data)
        arrival_time_provider = ConstantArrivalTimeProvider(profile, start_time=start_time)

        super().__init__(
            name=f"Probe_{target.name}_{metric}",
            event_provider=provider,
            arrival_time_provider=arrival_time_provider,
        )
        logger.info(
            "Probe created: target=%s metric=%s interval=%.3fs", target.name, metric, interval
        )

    @classmethod
    def on(cls, target: Entity, metric: str, interval: float = 1.0) -> tuple["Probe", Data]:
        """Create a Probe and its Data container in one call.

        Args:
            target: The entity to measure.
            metric: Attribute or property name to sample.
            interval: Seconds between measurements. Defaults to 1.0.

        Returns:
            (probe, data) tuple. Pass probe to Simulation(probes=[...]),
            use data for post-simulation analysis.
        """
        data = Data()
        probe = cls(target=target, metric=metric, data=data, interval=interval)
        return probe, data

    @classmethod
    def on_many(
        cls, target: Entity, metrics: list[str], interval: float = 1.0
    ) -> tuple[list["Probe"], dict[str, Data]]:
        """Create Probes for multiple metrics on the same target.

        Args:
            target: The entity to measure.
            metrics: List of attribute/property names to sample.
            interval: Seconds between measurements. Defaults to 1.0.

        Returns:
            (probes_list, data_dict) where data_dict is keyed by metric name.
        """
        probes: list[Probe] = []
        data_dict: dict[str, Data] = {}
        for metric in metrics:
            probe, data = cls.on(target, metric, interval=interval)
            probes.append(probe)
            data_dict[metric] = data
        return probes, data_dict
