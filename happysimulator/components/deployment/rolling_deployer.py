"""Rolling deployment strategy for zero-downtime backend replacement.

Replaces backends one-by-one (or in batches) with new versions,
health-checking each before removing the old instance.

Example:
    from happysimulator.components.deployment import RollingDeployer

    deployer = RollingDeployer(
        name="deployer", load_balancer=lb,
        server_factory=lambda name: Server(name=name, ...),
        batch_size=1, health_check_interval=2.0,
    )

    sim = Simulation(entities=[deployer, lb, ...])
    sim.schedule(deployer.deploy())
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Duration, Instant

logger = logging.getLogger(__name__)


@dataclass
class DeploymentState:
    """Current state of a rolling deployment."""

    status: str = "idle"  # idle, in_progress, completed, rolled_back
    total_instances: int = 0
    replaced: int = 0
    failed: int = 0
    current_batch: int = 0


@dataclass(frozen=True)
class RollingDeployerStats:
    """Statistics tracked by RollingDeployer."""

    deployments_started: int = 0
    deployments_completed: int = 0
    deployments_rolled_back: int = 0
    instances_replaced: int = 0
    health_checks_performed: int = 0
    health_checks_passed: int = 0
    health_checks_failed: int = 0


class RollingDeployer(Entity):
    """Rolls out new backend versions one batch at a time.

    For each batch:
    1. Create new instances and add to load balancer
    2. Health check new instances until healthy_threshold consecutive passes
    3. Remove old instances from the batch
    4. If health checks fail too many times, rollback

    Attributes:
        name: Deployer identifier.
        stats: Frozen statistics snapshot.
        state: Current deployment state.
    """

    def __init__(
        self,
        name: str,
        load_balancer: Entity,
        server_factory: Callable[[str], Entity],
        batch_size: int = 1,
        health_check_interval: float = 2.0,
        healthy_threshold: int = 2,
        max_failures: int = 1,
    ):
        """Initialize the rolling deployer.

        Args:
            name: Deployer identifier.
            load_balancer: LoadBalancer whose backends to replace.
            server_factory: Creates new server instances.
            batch_size: Number of instances to replace per batch.
            health_check_interval: Seconds between health checks.
            healthy_threshold: Consecutive passes required.
            max_failures: Failures before rollback.
        """
        super().__init__(name)

        self._load_balancer = load_balancer
        self._server_factory = server_factory
        self._batch_size = batch_size
        self._health_check_interval = health_check_interval
        self._healthy_threshold = healthy_threshold
        self._max_failures = max_failures

        self._old_backends: list[Entity] = []
        self._new_backends: list[Entity] = []
        self._current_batch_old: list[Entity] = []
        self._current_batch_new: list[Entity] = []
        self._health_pass_count: dict[str, int] = {}
        self._health_fail_count: int = 0
        self._next_instance_id = 0

        self._deployments_started = 0
        self._deployments_completed = 0
        self._deployments_rolled_back = 0
        self._instances_replaced = 0
        self._health_checks_performed = 0
        self._health_checks_passed = 0
        self._health_checks_failed = 0
        self.state = DeploymentState()

        logger.debug(
            "[%s] RollingDeployer initialized: batch_size=%d, "
            "health_interval=%.1fs, healthy_threshold=%d, max_failures=%d",
            name,
            batch_size,
            health_check_interval,
            healthy_threshold,
            max_failures,
        )

    @property
    def stats(self) -> RollingDeployerStats:
        """Return a frozen snapshot of current statistics."""
        return RollingDeployerStats(
            deployments_started=self._deployments_started,
            deployments_completed=self._deployments_completed,
            deployments_rolled_back=self._deployments_rolled_back,
            instances_replaced=self._instances_replaced,
            health_checks_performed=self._health_checks_performed,
            health_checks_passed=self._health_checks_passed,
            health_checks_failed=self._health_checks_failed,
        )

    def deploy(self) -> Event:
        """Start a rolling deployment.

        Returns:
            The deployment start event.
        """
        return Event(
            time=self.now if self._clock is not None else Instant.Epoch,
            event_type="_rolling_deploy_start",
            target=self,
            context={},
        )

    def handle_event(self, event: Event) -> list[Event] | None:
        et = event.event_type

        if et == "_rolling_deploy_start":
            return self._start_deployment()
        if et == "_rolling_replace_batch":
            return self._replace_batch()
        if et == "_rolling_health_check":
            return self._run_health_check(event)
        if et == "_rolling_health_pass":
            return self._handle_health_pass(event)
        if et == "_rolling_health_timeout":
            return self._handle_health_timeout(event)
        if et == "_rolling_rollback":
            return self._rollback()
        if et == "_rolling_complete":
            return self._complete()
        return None

    def _start_deployment(self) -> list[Event]:
        """Begin the rolling deployment."""
        self._old_backends = list(self._load_balancer.all_backends)
        self._new_backends = []
        self._health_fail_count = 0
        self._health_pass_count = {}

        self.state = DeploymentState(
            status="in_progress",
            total_instances=len(self._old_backends),
        )
        self._deployments_started += 1

        logger.info(
            "[%s] Starting rolling deployment: %d instances to replace",
            self.name,
            len(self._old_backends),
        )

        return [
            Event(
                time=self.now,
                event_type="_rolling_replace_batch",
                target=self,
                context={},
            )
        ]

    def _replace_batch(self) -> list[Event]:
        """Create new instances for the current batch."""
        if not self._old_backends:
            # All batches done
            return [
                Event(
                    time=self.now,
                    event_type="_rolling_complete",
                    target=self,
                    context={},
                )
            ]

        self.state.current_batch += 1
        batch_count = min(self._batch_size, len(self._old_backends))

        self._current_batch_old = self._old_backends[:batch_count]
        self._old_backends = self._old_backends[batch_count:]
        self._current_batch_new = []

        events = []
        for _old_backend in self._current_batch_old:
            self._next_instance_id += 1
            new_name = f"{self.name}_v2_{self._next_instance_id}"
            new_server = self._server_factory(new_name)

            # Inject clock
            if self._clock is not None:
                new_server.set_clock(self._clock)

            self._load_balancer.add_backend(new_server)
            self._current_batch_new.append(new_server)
            self._new_backends.append(new_server)
            self._health_pass_count[new_name] = 0

        # Start health checking the new instances
        events.append(
            Event(
                time=self.now + Duration.from_seconds(self._health_check_interval),
                event_type="_rolling_health_check",
                target=self,
                context={},
            )
        )

        logger.debug(
            "[%s] Batch %d: created %d new instances",
            self.name,
            self.state.current_batch,
            batch_count,
        )

        return events

    def _run_health_check(self, event: Event) -> list[Event]:
        """Send health check probes to new instances in current batch."""
        if self.state.status != "in_progress":
            return []

        events = []
        for new_server in self._current_batch_new:
            self._health_checks_performed += 1

            # Send probe with completion hook
            probe = Event(
                time=self.now,
                event_type="health_check",
                target=new_server,
                context={
                    "metadata": {
                        "_deployer": self.name,
                        "_server_name": new_server.name,
                    },
                },
            )

            server_name = new_server.name

            def on_complete(finish_time: Instant, _name=server_name) -> Event:
                return Event(
                    time=finish_time,
                    event_type="_rolling_health_pass",
                    target=self,
                    context={"metadata": {"server_name": _name}},
                )

            probe.add_completion_hook(on_complete)
            events.append(probe)

            # Timeout
            events.append(
                Event(
                    time=self.now + Duration.from_seconds(self._health_check_interval),
                    event_type="_rolling_health_timeout",
                    target=self,
                    context={"metadata": {"server_name": server_name}},
                )
            )

        return events

    def _handle_health_pass(self, event: Event) -> list[Event]:
        """Handle a successful health check response."""
        server_name = event.context.get("metadata", {}).get("server_name")
        if not server_name or server_name not in self._health_pass_count:
            return []

        self._health_pass_count[server_name] += 1
        self._health_checks_passed += 1

        logger.debug(
            "[%s] Health pass for %s (%d/%d)",
            self.name,
            server_name,
            self._health_pass_count[server_name],
            self._healthy_threshold,
        )

        # Check if all new instances in batch are healthy
        all_healthy = all(
            self._health_pass_count.get(s.name, 0) >= self._healthy_threshold
            for s in self._current_batch_new
        )

        if all_healthy:
            # Remove old instances from this batch
            for old_server in self._current_batch_old:
                self._load_balancer.remove_backend(old_server)
                self.state.replaced += 1
                self._instances_replaced += 1

            logger.info(
                "[%s] Batch %d complete: %d instances replaced",
                self.name,
                self.state.current_batch,
                len(self._current_batch_old),
            )

            # Move to next batch
            return [
                Event(
                    time=self.now,
                    event_type="_rolling_replace_batch",
                    target=self,
                    context={},
                )
            ]

        # Schedule next health check
        return [
            Event(
                time=self.now + Duration.from_seconds(self._health_check_interval),
                event_type="_rolling_health_check",
                target=self,
                context={},
            )
        ]

    def _handle_health_timeout(self, event: Event) -> list[Event]:
        """Handle a health check timeout (failure)."""
        server_name = event.context.get("metadata", {}).get("server_name")
        if not server_name:
            return []

        # Only count if we haven't already passed for this server
        passes = self._health_pass_count.get(server_name, 0)
        if passes >= self._healthy_threshold:
            return []  # Already healthy, ignore timeout

        self._health_fail_count += 1
        self._health_checks_failed += 1
        self.state.failed += 1

        logger.warning(
            "[%s] Health check failed for %s (total failures=%d/%d)",
            self.name,
            server_name,
            self._health_fail_count,
            self._max_failures,
        )

        if self._health_fail_count > self._max_failures:
            return [
                Event(
                    time=self.now,
                    event_type="_rolling_rollback",
                    target=self,
                    context={},
                )
            ]

        return []

    def _rollback(self) -> list[Event]:
        """Roll back the deployment: remove new, restore old."""
        self.state.status = "rolled_back"
        self._deployments_rolled_back += 1

        # Remove all new instances
        for new_server in self._new_backends:
            self._load_balancer.remove_backend(new_server)

        # Re-add old instances that were removed
        for old_backend in self._current_batch_old:
            if old_backend.name not in {b.name for b in self._load_balancer.all_backends}:
                self._load_balancer.add_backend(old_backend)

        logger.info("[%s] Deployment rolled back", self.name)
        return []

    def _complete(self) -> list[Event]:
        """Mark deployment as completed."""
        self.state.status = "completed"
        self._deployments_completed += 1
        logger.info(
            "[%s] Deployment completed: %d instances replaced",
            self.name,
            self.state.replaced,
        )
        return []
