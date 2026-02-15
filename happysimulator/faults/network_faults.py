"""Network-level fault injection.

Provides faults that manipulate network links and partitions:
- ``InjectLatency``: add extra latency to a link for a time window
- ``InjectPacketLoss``: increase packet loss rate for a time window
- ``NetworkPartition``: create a partition between groups for a time window
- ``RandomPartition``: Jepsen-style recurring random partitions
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from happysimulator.core.event import Event
from happysimulator.core.temporal import Duration, Instant
from happysimulator.distributions.latency_distribution import LatencyDistribution

if TYPE_CHECKING:
    from happysimulator.faults.fault import FaultContext

logger = logging.getLogger(__name__)


class _CompoundLatency(LatencyDistribution):
    """Adds two latency distributions together.

    Returns ``base.get_latency() + extra.get_latency()`` on each sample.
    Used internally by ``InjectLatency`` to layer extra latency on top
    of the existing link latency.
    """

    def __init__(self, base: LatencyDistribution, extra: LatencyDistribution) -> None:
        # Mean is sum of means (informational only)
        super().__init__(base._mean_latency + extra._mean_latency)
        self._base = base
        self._extra = extra

    def get_latency(self, current_time: Instant) -> Duration:
        base_dur = self._base.get_latency(current_time)
        extra_dur = self._extra.get_latency(current_time)
        return Duration.from_seconds(base_dur.to_seconds() + extra_dur.to_seconds())


@dataclass(frozen=True)
class InjectLatency:
    """Add extra latency to a network link for a time window.

    At ``start``, replaces the link's latency with a compound distribution
    that adds ``extra_ms`` milliseconds. At ``end``, restores the original.

    Attributes:
        source_name: Source entity name for the link.
        dest_name: Destination entity name for the link.
        extra_ms: Extra latency to add in milliseconds.
        start: Fault activation time in seconds.
        end: Fault deactivation time in seconds.
        network_name: Network to target. None = use first found.
    """

    source_name: str
    dest_name: str
    extra_ms: float
    start: float
    end: float
    network_name: str | None = None

    def generate_events(self, ctx: FaultContext) -> list[Event]:
        from happysimulator.distributions.constant import ConstantLatency

        network = self._resolve_network(ctx)
        link = network.get_link(self.source_name, self.dest_name)
        if link is None:
            raise ValueError(f"No link found: {self.source_name} -> {self.dest_name}")

        original_latency = link.latency
        extra_dist = ConstantLatency(self.extra_ms / 1000.0)
        src = self.source_name
        dst = self.dest_name

        def activate(e: Event) -> None:
            link.latency = _CompoundLatency(original_latency, extra_dist)
            logger.info(
                "[FaultInjection] Injected +%sms latency on %s -> %s at %s",
                self.extra_ms,
                src,
                dst,
                e.time,
            )

        def deactivate(e: Event) -> None:
            link.latency = original_latency
            logger.info(
                "[FaultInjection] Restored latency on %s -> %s at %s",
                src,
                dst,
                e.time,
            )

        return [
            Event.once(
                time=Instant.from_seconds(self.start),
                event_type=f"fault.latency.activate:{src}->{dst}",
                fn=activate,
                daemon=True,
            ),
            Event.once(
                time=Instant.from_seconds(self.end),
                event_type=f"fault.latency.deactivate:{src}->{dst}",
                fn=deactivate,
                daemon=True,
            ),
        ]

    def _resolve_network(self, ctx: FaultContext):
        if self.network_name is not None:
            return ctx.networks[self.network_name]
        if not ctx.networks:
            raise ValueError("No networks registered in simulation")
        return next(iter(ctx.networks.values()))


@dataclass(frozen=True)
class InjectPacketLoss:
    """Inject additional packet loss on a link for a time window.

    At ``start``, increases the link's ``packet_loss_rate``. At ``end``,
    restores the original rate.

    Attributes:
        source_name: Source entity name for the link.
        dest_name: Destination entity name for the link.
        loss_rate: Additional loss rate to add [0, 1].
        start: Fault activation time in seconds.
        end: Fault deactivation time in seconds.
        network_name: Network to target. None = use first found.
    """

    source_name: str
    dest_name: str
    loss_rate: float
    start: float
    end: float
    network_name: str | None = None

    def generate_events(self, ctx: FaultContext) -> list[Event]:
        network = self._resolve_network(ctx)
        link = network.get_link(self.source_name, self.dest_name)
        if link is None:
            raise ValueError(f"No link found: {self.source_name} -> {self.dest_name}")

        original_loss = link.packet_loss_rate
        src = self.source_name
        dst = self.dest_name
        extra = self.loss_rate

        def activate(e: Event) -> None:
            link.packet_loss_rate = min(1.0, original_loss + extra)
            logger.info(
                "[FaultInjection] Injected +%.1f%% packet loss on %s -> %s at %s",
                extra * 100,
                src,
                dst,
                e.time,
            )

        def deactivate(e: Event) -> None:
            link.packet_loss_rate = original_loss
            logger.info(
                "[FaultInjection] Restored packet loss on %s -> %s at %s",
                src,
                dst,
                e.time,
            )

        return [
            Event.once(
                time=Instant.from_seconds(self.start),
                event_type=f"fault.loss.activate:{src}->{dst}",
                fn=activate,
                daemon=True,
            ),
            Event.once(
                time=Instant.from_seconds(self.end),
                event_type=f"fault.loss.deactivate:{src}->{dst}",
                fn=deactivate,
                daemon=True,
            ),
        ]

    def _resolve_network(self, ctx: FaultContext):
        if self.network_name is not None:
            return ctx.networks[self.network_name]
        if not ctx.networks:
            raise ValueError("No networks registered in simulation")
        return next(iter(ctx.networks.values()))


@dataclass(frozen=True)
class NetworkPartition:
    """Create a network partition between two groups for a time window.

    At ``start``, calls ``network.partition()`` to block traffic between
    groups. At ``end``, heals the partition.

    Attributes:
        group_a: Entity names for group A.
        group_b: Entity names for group B.
        start: Partition start time in seconds.
        end: Partition end time in seconds.
        asymmetric: If True, only block A -> B traffic.
        network_name: Network to target. None = use first found.
    """

    group_a: list[str]
    group_b: list[str]
    start: float
    end: float
    asymmetric: bool = False
    network_name: str | None = None

    def generate_events(self, ctx: FaultContext) -> list[Event]:
        network = self._resolve_network(ctx)
        entities_a = [ctx.entities[n] for n in self.group_a]
        entities_b = [ctx.entities[n] for n in self.group_b]
        partition_handle = None
        asymmetric = self.asymmetric

        def activate(e: Event) -> None:
            nonlocal partition_handle
            partition_handle = network.partition(entities_a, entities_b, asymmetric=asymmetric)
            logger.info(
                "[FaultInjection] Network partition %s <-X-> %s at %s",
                self.group_a,
                self.group_b,
                e.time,
            )

        def deactivate(e: Event) -> None:
            if partition_handle is not None:
                partition_handle.heal()
                logger.info(
                    "[FaultInjection] Partition healed %s <-> %s at %s",
                    self.group_a,
                    self.group_b,
                    e.time,
                )

        return [
            Event.once(
                time=Instant.from_seconds(self.start),
                event_type="fault.partition.activate",
                fn=activate,
                daemon=True,
            ),
            Event.once(
                time=Instant.from_seconds(self.end),
                event_type="fault.partition.deactivate",
                fn=deactivate,
                daemon=True,
            ),
        ]

    def _resolve_network(self, ctx: FaultContext):
        if self.network_name is not None:
            return ctx.networks[self.network_name]
        if not ctx.networks:
            raise ValueError("No networks registered in simulation")
        return next(iter(ctx.networks.values()))


@dataclass(frozen=True)
class RandomPartition:
    """Jepsen-style random partition injection (recurring).

    Schedules fault/heal cycles using exponentially distributed intervals.
    Each cycle randomly splits nodes into two groups, creates a partition,
    then heals after a random repair time.

    The self-scheduling chain (like Source's self-perpetuation) uses
    ``Event.once()`` callbacks that schedule the next event.

    Attributes:
        nodes: Entity names that can be partitioned.
        mtbf: Mean time between failures in seconds.
        mttr: Mean time to repair in seconds.
        seed: Random seed for reproducibility.
        network_name: Network to target. None = use first found.
    """

    nodes: list[str]
    mtbf: float
    mttr: float
    seed: int | None = None
    network_name: str | None = None

    def generate_events(self, ctx: FaultContext) -> list[Event]:
        from happysimulator.core.sim_future import _get_active_heap

        network = self._resolve_network(ctx)
        rng = random.Random(self.seed)
        entities = {n: ctx.entities[n] for n in self.nodes}
        node_names = list(self.nodes)
        mtbf = self.mtbf
        mttr = self.mttr
        partition_handle = None

        def schedule_fault(at: float) -> Event:
            def do_fault(e: Event) -> None:
                nonlocal partition_handle

                # Random split: shuffle and divide
                rng.shuffle(node_names)
                split = max(1, len(node_names) // 2)
                group_a_names = node_names[:split]
                group_b_names = node_names[split:]

                group_a = [entities[n] for n in group_a_names]
                group_b = [entities[n] for n in group_b_names]

                partition_handle = network.partition(group_a, group_b)
                logger.info(
                    "[FaultInjection] Random partition %s <-X-> %s at %s",
                    group_a_names,
                    group_b_names,
                    e.time,
                )

                # Schedule heal
                repair_delay = rng.expovariate(1.0 / mttr)
                heal_time = e.time.to_seconds() + repair_delay
                heal_event = schedule_heal(heal_time)

                # Push heal event onto the heap via active context
                heap = _get_active_heap()
                if heap is not None:
                    heap.push(heal_event)

            return Event.once(
                time=Instant.from_seconds(at),
                event_type="fault.random_partition.fault",
                fn=do_fault,
                daemon=True,
            )

        def schedule_heal(at: float) -> Event:
            def do_heal(e: Event) -> None:
                nonlocal partition_handle
                if partition_handle is not None:
                    partition_handle.heal()
                    partition_handle = None
                    logger.info(
                        "[FaultInjection] Random partition healed at %s",
                        e.time,
                    )

                # Schedule next fault
                fault_delay = rng.expovariate(1.0 / mtbf)
                next_fault_time = e.time.to_seconds() + fault_delay
                next_event = schedule_fault(next_fault_time)

                heap = _get_active_heap()
                if heap is not None:
                    heap.push(next_event)

            return Event.once(
                time=Instant.from_seconds(at),
                event_type="fault.random_partition.heal",
                fn=do_heal,
                daemon=True,
            )

        # Schedule the first fault
        first_delay = rng.expovariate(1.0 / mtbf)
        first_time = ctx.start_time.to_seconds() + first_delay
        return [schedule_fault(first_time)]

    def _resolve_network(self, ctx: FaultContext):
        if self.network_name is not None:
            return ctx.networks[self.network_name]
        if not ctx.networks:
            raise ValueError("No networks registered in simulation")
        return next(iter(ctx.networks.values()))
