"""Unit tests for AutoScaler."""

import pytest

from happysimulator.core.clock import Clock
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Duration, Instant

from happysimulator.components.deployment.auto_scaler import (
    AutoScaler,
    AutoScalerStats,
    QueueDepthScaling,
    ScalingEvent,
    StepScaling,
    TargetUtilization,
)


class FakeBackend(Entity):
    """Backend with controllable utilization and queue depth."""

    def __init__(self, name: str, utilization: float = 0.5, depth: int = 0):
        super().__init__(name)
        self._utilization = utilization
        self._depth = depth

    @property
    def utilization(self) -> float:
        return self._utilization

    @property
    def depth(self) -> int:
        return self._depth

    def handle_event(self, event):
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


def make_scaler(
    backends: list[Entity] | None = None,
    policy=None,
    min_instances: int = 1,
    max_instances: int = 10,
    evaluation_interval: float = 10.0,
    scale_out_cooldown: float = 30.0,
    scale_in_cooldown: float = 60.0,
    time: float = 0.0,
) -> tuple[AutoScaler, FakeLoadBalancer, Clock]:
    clock = Clock(Instant.from_seconds(time))
    lb = FakeLoadBalancer(backends)
    lb.set_clock(clock)

    scaler = AutoScaler(
        name="scaler",
        load_balancer=lb,
        server_factory=lambda name: FakeBackend(name),
        policy=policy,
        min_instances=min_instances,
        max_instances=max_instances,
        evaluation_interval=evaluation_interval,
        scale_out_cooldown=scale_out_cooldown,
        scale_in_cooldown=scale_in_cooldown,
    )
    scaler.set_clock(clock)
    return scaler, lb, clock


class TestAutoScalerCreation:
    def test_basic_creation(self):
        scaler, _, _ = make_scaler()
        assert scaler.name == "scaler"
        assert scaler.min_instances == 1
        assert scaler.max_instances == 10
        assert not scaler.is_running

    def test_start_returns_event(self):
        scaler, _, _ = make_scaler()
        event = scaler.start()
        assert event.event_type == "_autoscaler_evaluate"
        assert scaler.is_running

    def test_stop(self):
        scaler, _, _ = make_scaler()
        scaler.start()
        scaler.stop()
        assert not scaler.is_running


class TestTargetUtilizationPolicy:
    def test_invalid_target(self):
        with pytest.raises(ValueError):
            TargetUtilization(target=0)
        with pytest.raises(ValueError):
            TargetUtilization(target=1.5)

    def test_scale_out_when_high_utilization(self):
        backends = [FakeBackend(f"s{i}", utilization=0.9) for i in range(3)]
        policy = TargetUtilization(target=0.7)
        desired = policy.evaluate(backends, 3, 1, 10)
        assert desired > 3

    def test_scale_in_when_low_utilization(self):
        backends = [FakeBackend(f"s{i}", utilization=0.2) for i in range(5)]
        policy = TargetUtilization(target=0.7)
        desired = policy.evaluate(backends, 5, 1, 10)
        assert desired < 5

    def test_respects_max_bound(self):
        backends = [FakeBackend(f"s{i}", utilization=0.99) for i in range(3)]
        policy = TargetUtilization(target=0.1)
        desired = policy.evaluate(backends, 3, 1, 5)
        assert desired <= 5

    def test_respects_min_bound(self):
        backends = [FakeBackend(f"s{i}", utilization=0.01) for i in range(3)]
        policy = TargetUtilization(target=0.7)
        desired = policy.evaluate(backends, 3, 2, 10)
        assert desired >= 2


class TestStepScalingPolicy:
    def test_step_up(self):
        backends = [FakeBackend("s1", utilization=0.85)]
        policy = StepScaling([(0.8, 2), (0.6, 1), (0.3, -1)])
        desired = policy.evaluate(backends, 2, 1, 10)
        assert desired == 4  # 2 + 2

    def test_step_down(self):
        backends = [FakeBackend("s1", utilization=0.2)]
        policy = StepScaling([(0.8, 2), (0.6, 1), (0.1, -1)])
        desired = policy.evaluate(backends, 3, 1, 10)
        assert desired == 2  # 3 - 1


class TestQueueDepthPolicy:
    def test_scale_out_high_depth(self):
        backends = [FakeBackend("s1", depth=150)]
        policy = QueueDepthScaling(scale_out_threshold=100)
        desired = policy.evaluate(backends, 2, 1, 10)
        assert desired == 3

    def test_scale_in_low_depth(self):
        backends = [FakeBackend("s1", depth=5)]
        policy = QueueDepthScaling(scale_in_threshold=10)
        desired = policy.evaluate(backends, 3, 1, 10)
        assert desired == 2


class TestScaleOut:
    def test_scale_out_adds_backend(self):
        backends = [FakeBackend(f"s{i}", utilization=0.95) for i in range(2)]
        scaler, lb, clock = make_scaler(
            backends=backends,
            policy=TargetUtilization(target=0.5),
            scale_out_cooldown=0.0,
        )

        eval_event = Event(
            time=Instant.Epoch, event_type="_autoscaler_evaluate",
            target=scaler, context={},
        )
        scaler._is_running = True
        scaler.handle_event(eval_event)

        assert len(lb.all_backends) > 2
        assert scaler.stats.scale_out_count >= 1
        assert scaler.stats.instances_added >= 1

    def test_max_bound_enforced(self):
        backends = [FakeBackend(f"s{i}", utilization=0.99) for i in range(5)]
        scaler, lb, clock = make_scaler(
            backends=backends,
            policy=TargetUtilization(target=0.1),
            max_instances=6,
            scale_out_cooldown=0.0,
        )

        scaler._is_running = True
        eval_event = Event(
            time=Instant.Epoch, event_type="_autoscaler_evaluate",
            target=scaler, context={},
        )
        scaler.handle_event(eval_event)
        assert len(lb.all_backends) <= 6


class TestScaleIn:
    def test_scale_in_removes_backend(self):
        backends = [FakeBackend(f"s{i}", utilization=0.1) for i in range(5)]
        scaler, lb, clock = make_scaler(
            backends=backends,
            policy=TargetUtilization(target=0.7),
            scale_in_cooldown=0.0,
        )

        # Need managed servers for scale-in to work
        for b in backends:
            scaler._managed_servers.append(b)

        scaler._is_running = True
        eval_event = Event(
            time=Instant.Epoch, event_type="_autoscaler_evaluate",
            target=scaler, context={},
        )
        scaler.handle_event(eval_event)
        assert len(lb.all_backends) < 5

    def test_min_bound_enforced(self):
        backends = [FakeBackend(f"s{i}", utilization=0.05) for i in range(3)]
        scaler, lb, clock = make_scaler(
            backends=backends,
            policy=TargetUtilization(target=0.7),
            min_instances=2,
            scale_in_cooldown=0.0,
        )

        for b in backends:
            scaler._managed_servers.append(b)

        scaler._is_running = True
        eval_event = Event(
            time=Instant.Epoch, event_type="_autoscaler_evaluate",
            target=scaler, context={},
        )
        scaler.handle_event(eval_event)
        assert len(lb.all_backends) >= 2


class TestCooldown:
    def test_cooldown_prevents_oscillation(self):
        backends = [FakeBackend(f"s{i}", utilization=0.95) for i in range(2)]
        scaler, lb, clock = make_scaler(
            backends=backends,
            policy=TargetUtilization(target=0.5),
            scale_out_cooldown=30.0,
        )
        scaler._is_running = True

        # First evaluation: scale out
        eval1 = Event(
            time=Instant.Epoch, event_type="_autoscaler_evaluate",
            target=scaler, context={},
        )
        scaler.handle_event(eval1)
        count_after_first = len(lb.all_backends)
        assert count_after_first > 2

        # Second evaluation at t=10s (within cooldown)
        clock._current_time = Instant.from_seconds(10.0)
        eval2 = Event(
            time=Instant.from_seconds(10.0), event_type="_autoscaler_evaluate",
            target=scaler, context={},
        )
        scaler.handle_event(eval2)
        assert len(lb.all_backends) == count_after_first
        assert scaler.stats.cooldown_blocks >= 1


class TestClockInjection:
    def test_new_instances_receive_clock(self):
        created_servers = []
        def factory(name):
            server = FakeBackend(name)
            created_servers.append(server)
            return server

        backends = [FakeBackend(f"s{i}", utilization=0.95) for i in range(2)]
        clock = Clock(Instant.Epoch)
        lb = FakeLoadBalancer(backends)
        lb.set_clock(clock)

        scaler = AutoScaler(
            name="scaler", load_balancer=lb,
            server_factory=factory,
            policy=TargetUtilization(target=0.5),
            scale_out_cooldown=0.0,
        )
        scaler.set_clock(clock)
        scaler._is_running = True

        eval_event = Event(
            time=Instant.Epoch, event_type="_autoscaler_evaluate",
            target=scaler, context={},
        )
        scaler.handle_event(eval_event)

        # Verify created servers have clock
        for server in created_servers:
            assert server._clock is clock


class TestScalingHistory:
    def test_history_records_events(self):
        backends = [FakeBackend(f"s{i}", utilization=0.95) for i in range(2)]
        scaler, lb, clock = make_scaler(
            backends=backends,
            policy=TargetUtilization(target=0.5),
            scale_out_cooldown=0.0,
        )
        scaler._is_running = True

        eval_event = Event(
            time=Instant.Epoch, event_type="_autoscaler_evaluate",
            target=scaler, context={},
        )
        scaler.handle_event(eval_event)

        assert len(scaler.scaling_history) > 0
        event = scaler.scaling_history[0]
        assert isinstance(event, ScalingEvent)
        assert event.action == "scale_out"
