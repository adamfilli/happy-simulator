"""Auto-scaling controller that monitors load and adjusts backend count.

Provides periodic evaluation of backend utilization and automatically
scales the number of instances up or down within configured bounds.
Supports pluggable scaling policies and cooldown periods.

Example:
    from happysimulator.components.deployment import AutoScaler, TargetUtilization

    scaler = AutoScaler(
        name="scaler", load_balancer=lb,
        server_factory=lambda name: Server(name=name, service_time=ConstantLatency(0.1)),
        policy=TargetUtilization(target=0.7),
        min_instances=1, max_instances=10,
    )

    sim = Simulation(entities=[scaler, lb, ...])
    sim.schedule(scaler.start())
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Duration, Instant

logger = logging.getLogger(__name__)


@runtime_checkable
class ScalingPolicy(Protocol):
    """Protocol for scaling decision algorithms."""

    def evaluate(
        self,
        backends: list[Entity],
        current_count: int,
        min_instances: int,
        max_instances: int,
    ) -> int:
        """Evaluate and return desired instance count.

        Args:
            backends: Current backend entities.
            current_count: Current number of instances.
            min_instances: Minimum allowed instances.
            max_instances: Maximum allowed instances.

        Returns:
            Desired instance count.
        """
        ...


class TargetUtilization:
    """Scale to keep average utilization near a target.

    Calculates average utilization across backends and scales to bring
    utilization close to the target value.
    """

    def __init__(self, target: float = 0.7):
        if not 0 < target <= 1.0:
            raise ValueError(f"target must be in (0, 1], got {target}")
        self._target = target

    @property
    def target(self) -> float:
        return self._target

    def evaluate(
        self,
        backends: list[Entity],
        current_count: int,
        min_instances: int,
        max_instances: int,
    ) -> int:
        if not backends:
            return min_instances

        # Calculate average utilization
        utilizations = [b.utilization for b in backends if hasattr(b, "utilization")]

        if not utilizations:
            return current_count

        avg_util = sum(utilizations) / len(utilizations)

        # desired = ceil(current * avg_util / target)
        if self._target > 0:
            desired = int(current_count * avg_util / self._target + 0.5)
        else:
            desired = current_count

        return max(min_instances, min(max_instances, desired))


class StepScaling:
    """Step-based scaling with utilization thresholds.

    Each step defines a utilization threshold and the adjustment to make.
    Steps are evaluated from highest threshold to lowest.
    """

    def __init__(self, steps: list[tuple[float, int]]):
        """Initialize step scaling.

        Args:
            steps: List of (utilization_threshold, adjustment) tuples.
                   Sorted by threshold descending internally.
        """
        self._steps = sorted(steps, key=lambda s: s[0], reverse=True)

    def evaluate(
        self,
        backends: list[Entity],
        current_count: int,
        min_instances: int,
        max_instances: int,
    ) -> int:
        if not backends:
            return current_count

        utilizations = [b.utilization for b in backends if hasattr(b, "utilization")]

        if not utilizations:
            return current_count

        avg_util = sum(utilizations) / len(utilizations)

        for threshold, adjustment in self._steps:
            if avg_util >= threshold:
                desired = current_count + adjustment
                return max(min_instances, min(max_instances, desired))

        return current_count


class QueueDepthScaling:
    """Scale based on aggregate queue depth across backends."""

    def __init__(self, scale_out_threshold: int = 100, scale_in_threshold: int = 10):
        self._scale_out_threshold = scale_out_threshold
        self._scale_in_threshold = scale_in_threshold

    def evaluate(
        self,
        backends: list[Entity],
        current_count: int,
        min_instances: int,
        max_instances: int,
    ) -> int:
        total_depth = 0
        for b in backends:
            if hasattr(b, "depth"):
                total_depth += b.depth

        if total_depth >= self._scale_out_threshold:
            desired = current_count + 1
        elif total_depth <= self._scale_in_threshold and current_count > min_instances:
            desired = current_count - 1
        else:
            desired = current_count

        return max(min_instances, min(max_instances, desired))


@dataclass
class ScalingEvent:
    """Record of a scaling action."""

    time: Instant
    action: str  # "scale_out" or "scale_in"
    from_count: int
    to_count: int
    reason: str


@dataclass(frozen=True)
class AutoScalerStats:
    """Statistics tracked by AutoScaler."""

    evaluations: int = 0
    scale_out_count: int = 0
    scale_in_count: int = 0
    instances_added: int = 0
    instances_removed: int = 0
    cooldown_blocks: int = 0


class AutoScaler(Entity):
    """Auto-scaling controller for load balancer backends.

    Periodically evaluates backend utilization and adds/removes instances
    to maintain desired performance. Supports cooldown periods to prevent
    oscillation.

    Attributes:
        name: Scaler identifier.
        stats: Frozen statistics snapshot.
        scaling_history: List of scaling events.
    """

    def __init__(
        self,
        name: str,
        load_balancer: Entity,
        server_factory: Callable[[str], Entity],
        policy: ScalingPolicy | None = None,
        min_instances: int = 1,
        max_instances: int = 10,
        evaluation_interval: float = 10.0,
        scale_out_cooldown: float = 30.0,
        scale_in_cooldown: float = 60.0,
    ):
        """Initialize the auto scaler.

        Args:
            name: Scaler identifier.
            load_balancer: LoadBalancer to manage.
            server_factory: Callable that creates new server instances given a name.
            policy: Scaling decision policy (default TargetUtilization(0.7)).
            min_instances: Minimum backend count.
            max_instances: Maximum backend count.
            evaluation_interval: Seconds between evaluations.
            scale_out_cooldown: Seconds after scale-out before next action.
            scale_in_cooldown: Seconds after scale-in before next action.
        """
        super().__init__(name)

        self._load_balancer = load_balancer
        self._server_factory = server_factory
        self._policy = policy or TargetUtilization()
        self._min_instances = min_instances
        self._max_instances = max_instances
        self._evaluation_interval = evaluation_interval
        self._scale_out_cooldown = scale_out_cooldown
        self._scale_in_cooldown = scale_in_cooldown

        self._is_running = False
        self._last_scale_time: Instant | None = None
        self._last_scale_action: str | None = None
        self._next_instance_id = 0
        self._managed_servers: list[Entity] = []

        self._evaluations = 0
        self._scale_out_count = 0
        self._scale_in_count = 0
        self._instances_added = 0
        self._instances_removed = 0
        self._cooldown_blocks = 0
        self.scaling_history: list[ScalingEvent] = []

        logger.debug(
            "[%s] AutoScaler initialized: min=%d, max=%d, interval=%.1fs",
            name,
            min_instances,
            max_instances,
            evaluation_interval,
        )

    @property
    def stats(self) -> AutoScalerStats:
        """Return a frozen snapshot of current statistics."""
        return AutoScalerStats(
            evaluations=self._evaluations,
            scale_out_count=self._scale_out_count,
            scale_in_count=self._scale_in_count,
            instances_added=self._instances_added,
            instances_removed=self._instances_removed,
            cooldown_blocks=self._cooldown_blocks,
        )

    @property
    def load_balancer(self) -> Entity:
        return self._load_balancer

    @property
    def min_instances(self) -> int:
        return self._min_instances

    @property
    def max_instances(self) -> int:
        return self._max_instances

    @property
    def current_count(self) -> int:
        return len(self._load_balancer.all_backends)

    @property
    def is_running(self) -> bool:
        return self._is_running

    def start(self) -> Event:
        """Start periodic evaluation.

        Returns:
            The first evaluation event to schedule.
        """
        self._is_running = True
        return Event(
            time=self.now if self._clock is not None else Instant.Epoch,
            event_type="_autoscaler_evaluate",
            target=self,
            daemon=True,
            context={},
        )

    def stop(self) -> None:
        """Stop the auto scaler."""
        self._is_running = False

    def handle_event(self, event: Event) -> list[Event] | None:
        if event.event_type == "_autoscaler_evaluate":
            return self._evaluate()
        return None

    def _evaluate(self) -> list[Event]:
        """Run a scaling evaluation cycle."""
        if not self._is_running:
            return []

        self._evaluations += 1

        backends = self._load_balancer.all_backends
        current_count = len(backends)

        desired = self._policy.evaluate(
            backends,
            current_count,
            self._min_instances,
            self._max_instances,
        )

        if desired > current_count:
            self._try_scale_out(desired - current_count)
        elif desired < current_count:
            self._try_scale_in(current_count - desired)

        # Schedule next evaluation
        return [
            Event(
                time=self.now + Duration.from_seconds(self._evaluation_interval),
                event_type="_autoscaler_evaluate",
                target=self,
                daemon=True,
                context={},
            )
        ]

    def _in_cooldown(self, action: str) -> bool:
        """Check if we're in a cooldown period for the given action."""
        if self._last_scale_time is None:
            return False

        elapsed = (self.now - self._last_scale_time).to_seconds()
        if action == "scale_out":
            return elapsed < self._scale_out_cooldown
        return elapsed < self._scale_in_cooldown

    def _try_scale_out(self, count: int) -> None:
        """Attempt to add instances."""
        if self._in_cooldown("scale_out"):
            self._cooldown_blocks += 1
            return

        current = len(self._load_balancer.all_backends)
        # Clamp to max
        to_add = min(count, self._max_instances - current)
        if to_add <= 0:
            return

        for _ in range(to_add):
            self._next_instance_id += 1
            server_name = f"{self.name}_server_{self._next_instance_id}"
            new_server = self._server_factory(server_name)

            # Inject clock since Simulation only injects at init time
            if self._clock is not None:
                new_server.set_clock(self._clock)

            self._load_balancer.add_backend(new_server)
            self._managed_servers.append(new_server)

        new_count = len(self._load_balancer.all_backends)
        self._last_scale_time = self.now
        self._last_scale_action = "scale_out"
        self._scale_out_count += 1
        self._instances_added += to_add

        event = ScalingEvent(
            time=self.now,
            action="scale_out",
            from_count=current,
            to_count=new_count,
            reason=f"Added {to_add} instances",
        )
        self.scaling_history.append(event)

        logger.info(
            "[%s] Scale out: %d -> %d (added %d)",
            self.name,
            current,
            new_count,
            to_add,
        )

    def _try_scale_in(self, count: int) -> None:
        """Attempt to remove instances."""
        if self._in_cooldown("scale_in"):
            self._cooldown_blocks += 1
            return

        current = len(self._load_balancer.all_backends)
        # Clamp to min
        to_remove = min(count, current - self._min_instances)
        if to_remove <= 0:
            return

        # Remove most recently added managed servers
        for _ in range(to_remove):
            if self._managed_servers:
                server = self._managed_servers.pop()
                self._load_balancer.remove_backend(server)

        new_count = len(self._load_balancer.all_backends)
        self._last_scale_time = self.now
        self._last_scale_action = "scale_in"
        self._scale_in_count += 1
        self._instances_removed += to_remove

        event = ScalingEvent(
            time=self.now,
            action="scale_in",
            from_count=current,
            to_count=new_count,
            reason=f"Removed {to_remove} instances",
        )
        self.scaling_history.append(event)

        logger.info(
            "[%s] Scale in: %d -> %d (removed %d)",
            self.name,
            current,
            new_count,
            to_remove,
        )
