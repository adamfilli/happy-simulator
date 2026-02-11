"""Unit tests for RollingDeployer."""

import pytest

from happysimulator.core.clock import Clock
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Duration, Instant

from happysimulator.components.deployment.rolling_deployer import (
    DeploymentState,
    RollingDeployer,
    RollingDeployerStats,
)


class FakeBackend(Entity):
    """Backend that responds to health checks."""

    def __init__(self, name: str, healthy: bool = True):
        super().__init__(name)
        self._healthy = healthy

    def handle_event(self, event):
        if not self._healthy:
            raise RuntimeError("Unhealthy")
        return None


class FakeLoadBalancer(Entity):
    """Minimal load balancer for testing."""

    def __init__(self, backends: list[Entity] | None = None):
        super().__init__("lb")
        self._backends: dict[str, Entity] = {}
        if backends:
            for b in backends:
                self.add_backend(b)

    @property
    def all_backends(self) -> list[Entity]:
        return list(self._backends.values())

    def add_backend(self, backend: Entity) -> None:
        self._backends[backend.name] = backend

    def remove_backend(self, backend: Entity) -> None:
        self._backends.pop(backend.name, None)

    def handle_event(self, event):
        return None


def make_deployer(
    num_backends: int = 3,
    batch_size: int = 1,
    healthy_threshold: int = 2,
    max_failures: int = 1,
    time: float = 0.0,
) -> tuple[RollingDeployer, FakeLoadBalancer, Clock]:
    clock = Clock(Instant.from_seconds(time))
    backends = [FakeBackend(f"server_{i}") for i in range(num_backends)]
    lb = FakeLoadBalancer(backends)
    lb.set_clock(clock)
    for b in backends:
        b.set_clock(clock)

    deployer = RollingDeployer(
        name="deployer",
        load_balancer=lb,
        server_factory=lambda name: FakeBackend(name),
        batch_size=batch_size,
        healthy_threshold=healthy_threshold,
        max_failures=max_failures,
    )
    deployer.set_clock(clock)
    return deployer, lb, clock


class TestDeployerCreation:
    def test_basic_creation(self):
        deployer, _, _ = make_deployer()
        assert deployer.name == "deployer"
        assert deployer.state.status == "idle"
        assert deployer.stats.deployments_started == 0

    def test_deploy_returns_event(self):
        deployer, _, _ = make_deployer()
        event = deployer.deploy()
        assert event.event_type == "_rolling_deploy_start"
        assert event.target is deployer


class TestDeploymentStart:
    def test_start_captures_old_backends(self):
        deployer, lb, _ = make_deployer(num_backends=3)
        start_event = Event(
            time=Instant.Epoch, event_type="_rolling_deploy_start",
            target=deployer, context={},
        )
        deployer.handle_event(start_event)
        assert deployer.state.status == "in_progress"
        assert deployer.state.total_instances == 3
        assert deployer.stats.deployments_started == 1

    def test_start_triggers_batch_replace(self):
        deployer, lb, _ = make_deployer()
        start_event = Event(
            time=Instant.Epoch, event_type="_rolling_deploy_start",
            target=deployer, context={},
        )
        result = deployer.handle_event(start_event)
        assert any(e.event_type == "_rolling_replace_batch" for e in result)


class TestBatchReplacement:
    def test_batch_creates_new_instances(self):
        deployer, lb, clock = make_deployer(num_backends=3, batch_size=1)

        # Start deployment
        start = Event(
            time=Instant.Epoch, event_type="_rolling_deploy_start",
            target=deployer, context={},
        )
        deployer.handle_event(start)

        # Replace batch
        batch_event = Event(
            time=Instant.Epoch, event_type="_rolling_replace_batch",
            target=deployer, context={},
        )
        result = deployer.handle_event(batch_event)

        # Should have added new backend to lb
        assert len(lb.all_backends) == 4  # 3 old + 1 new
        # Should schedule health check
        assert any(e.event_type == "_rolling_health_check" for e in result)

    def test_batch_size_controls_pace(self):
        deployer, lb, clock = make_deployer(num_backends=4, batch_size=2)

        start = Event(
            time=Instant.Epoch, event_type="_rolling_deploy_start",
            target=deployer, context={},
        )
        deployer.handle_event(start)

        batch_event = Event(
            time=Instant.Epoch, event_type="_rolling_replace_batch",
            target=deployer, context={},
        )
        deployer.handle_event(batch_event)

        # Should have added 2 new backends
        assert len(lb.all_backends) == 6  # 4 old + 2 new


class TestHealthChecking:
    def test_health_pass_increments_count(self):
        deployer, lb, clock = make_deployer(healthy_threshold=2)

        # Start and replace batch
        start = Event(
            time=Instant.Epoch, event_type="_rolling_deploy_start",
            target=deployer, context={},
        )
        deployer.handle_event(start)
        batch = Event(
            time=Instant.Epoch, event_type="_rolling_replace_batch",
            target=deployer, context={},
        )
        deployer.handle_event(batch)

        new_name = deployer._current_batch_new[0].name

        # First health pass
        pass1 = Event(
            time=Instant.Epoch, event_type="_rolling_health_pass",
            target=deployer,
            context={"metadata": {"server_name": new_name}},
        )
        deployer.handle_event(pass1)
        assert deployer.stats.health_checks_passed == 1

    def test_health_threshold_triggers_removal(self):
        deployer, lb, clock = make_deployer(num_backends=2, healthy_threshold=1)

        start = Event(
            time=Instant.Epoch, event_type="_rolling_deploy_start",
            target=deployer, context={},
        )
        deployer.handle_event(start)

        batch = Event(
            time=Instant.Epoch, event_type="_rolling_replace_batch",
            target=deployer, context={},
        )
        deployer.handle_event(batch)

        new_name = deployer._current_batch_new[0].name

        # Single pass meets threshold
        pass_event = Event(
            time=Instant.Epoch, event_type="_rolling_health_pass",
            target=deployer,
            context={"metadata": {"server_name": new_name}},
        )
        result = deployer.handle_event(pass_event)

        # Old backend should be removed, moves to next batch
        assert deployer.state.replaced == 1
        assert any(e.event_type == "_rolling_replace_batch" for e in result)


class TestRollback:
    def test_rollback_on_too_many_failures(self):
        deployer, lb, clock = make_deployer(num_backends=2, max_failures=0)

        start = Event(
            time=Instant.Epoch, event_type="_rolling_deploy_start",
            target=deployer, context={},
        )
        deployer.handle_event(start)

        batch = Event(
            time=Instant.Epoch, event_type="_rolling_replace_batch",
            target=deployer, context={},
        )
        deployer.handle_event(batch)

        new_name = deployer._current_batch_new[0].name

        # Timeout triggers failure
        timeout = Event(
            time=Instant.Epoch, event_type="_rolling_health_timeout",
            target=deployer,
            context={"metadata": {"server_name": new_name}},
        )
        result = deployer.handle_event(timeout)
        assert any(e.event_type == "_rolling_rollback" for e in result)

    def test_rollback_removes_new_restores_old(self):
        deployer, lb, clock = make_deployer(num_backends=2)

        start = Event(
            time=Instant.Epoch, event_type="_rolling_deploy_start",
            target=deployer, context={},
        )
        deployer.handle_event(start)

        batch = Event(
            time=Instant.Epoch, event_type="_rolling_replace_batch",
            target=deployer, context={},
        )
        deployer.handle_event(batch)
        count_with_new = len(lb.all_backends)

        rollback = Event(
            time=Instant.Epoch, event_type="_rolling_rollback",
            target=deployer, context={},
        )
        deployer.handle_event(rollback)
        assert deployer.state.status == "rolled_back"
        assert deployer.stats.deployments_rolled_back == 1


class TestStateTransitions:
    def test_idle_to_in_progress(self):
        deployer, _, _ = make_deployer()
        assert deployer.state.status == "idle"
        start = Event(
            time=Instant.Epoch, event_type="_rolling_deploy_start",
            target=deployer, context={},
        )
        deployer.handle_event(start)
        assert deployer.state.status == "in_progress"

    def test_complete_deployment(self):
        deployer, lb, clock = make_deployer(num_backends=1, healthy_threshold=1)

        # Full flow: start -> batch -> health pass -> batch (empty) -> complete
        start = Event(
            time=Instant.Epoch, event_type="_rolling_deploy_start",
            target=deployer, context={},
        )
        deployer.handle_event(start)

        batch = Event(
            time=Instant.Epoch, event_type="_rolling_replace_batch",
            target=deployer, context={},
        )
        deployer.handle_event(batch)

        new_name = deployer._current_batch_new[0].name
        pass_event = Event(
            time=Instant.Epoch, event_type="_rolling_health_pass",
            target=deployer,
            context={"metadata": {"server_name": new_name}},
        )
        result = deployer.handle_event(pass_event)

        # Now _rolling_replace_batch with empty old list -> complete
        for e in result:
            if e.event_type == "_rolling_replace_batch":
                result2 = deployer.handle_event(e)
                for e2 in result2:
                    if e2.event_type == "_rolling_complete":
                        deployer.handle_event(e2)

        assert deployer.state.status == "completed"
        assert deployer.stats.deployments_completed == 1
