"""Unit tests for CanaryDeployer."""

from happysimulator.components.deployment.canary_deployer import (
    CanaryDeployer,
    CanaryStage,
    ErrorRateEvaluator,
    LatencyEvaluator,
)
from happysimulator.core.clock import Clock
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant


class FakeBackend(Entity):
    """Backend with controllable metrics."""

    def __init__(self, name: str, error_rate: float = 0.0, avg_latency: float = 0.05):
        super().__init__(name)
        self._error_rate = error_rate
        self._avg_latency = avg_latency
        # Stats for ErrorRateEvaluator
        self.stats = _FakeStats(error_rate)

    @property
    def average_service_time(self) -> float:
        return self._avg_latency

    def handle_event(self, event):
        return None


class _FakeStats:
    def __init__(self, error_rate: float):
        total = 100
        self.requests_completed = int(total * (1 - error_rate))
        self.requests_rejected = int(total * error_rate)
        self.total_failures = 0


class FakeLoadBalancer(Entity):
    """Minimal load balancer with weighted strategy support."""

    def __init__(self, backends: list[Entity] | None = None):
        super().__init__("lb")
        self._backends: dict[str, Entity] = {}
        self._strategy = _FakeWeightedStrategy()
        if backends:
            for b in backends:
                self.add_backend(b)

    @property
    def all_backends(self) -> list[Entity]:
        return list(self._backends.values())

    @property
    def strategy(self):
        return self._strategy

    def add_backend(self, backend: Entity) -> None:
        self._backends[backend.name] = backend

    def remove_backend(self, backend: Entity) -> None:
        self._backends.pop(backend.name, None)

    def handle_event(self, event):
        return None


class _FakeWeightedStrategy:
    def __init__(self):
        self._weights: dict[str, int] = {}

    def set_weight(self, backend: Entity, weight: int) -> None:
        self._weights[backend.name] = weight

    def get_weight(self, backend: Entity) -> int:
        return self._weights.get(backend.name, 1)


class AlwaysHealthy:
    """Evaluator that always reports healthy."""

    def is_healthy(self, canary, baseline_backends):
        return True


class AlwaysUnhealthy:
    """Evaluator that always reports unhealthy."""

    def is_healthy(self, canary, baseline_backends):
        return False


def make_deployer(
    num_backends: int = 3,
    stages: list[CanaryStage] | None = None,
    metric_evaluator=None,
    evaluation_interval: float = 5.0,
    time: float = 0.0,
) -> tuple[CanaryDeployer, FakeLoadBalancer, Clock]:
    clock = Clock(Instant.from_seconds(time))
    backends = [FakeBackend(f"server_{i}") for i in range(num_backends)]
    lb = FakeLoadBalancer(backends)
    lb.set_clock(clock)
    for b in backends:
        b.set_clock(clock)

    deployer = CanaryDeployer(
        name="canary",
        load_balancer=lb,
        server_factory=lambda name: FakeBackend(name),
        stages=stages,
        metric_evaluator=metric_evaluator,
        evaluation_interval=evaluation_interval,
    )
    deployer.set_clock(clock)
    return deployer, lb, clock


class TestCanaryDeployerCreation:
    def test_basic_creation(self):
        deployer, _, _ = make_deployer()
        assert deployer.name == "canary"
        assert deployer.state.status == "idle"
        assert deployer.canary is None

    def test_deploy_returns_event(self):
        deployer, _, _ = make_deployer()
        event = deployer.deploy()
        assert event.event_type == "_canary_deploy_start"


class TestDeploymentStart:
    def test_start_creates_canary(self):
        deployer, lb, _clock = make_deployer()
        start = Event(
            time=Instant.Epoch,
            event_type="_canary_deploy_start",
            target=deployer,
            context={},
        )
        deployer.handle_event(start)

        assert deployer.state.status == "in_progress"
        assert deployer.canary is not None
        assert deployer.canary.name in [b.name for b in lb.all_backends]
        assert deployer.stats.deployments_started == 1


class TestProgressiveTrafficShift:
    def test_stages_advance_progressively(self):
        stages = [
            CanaryStage(traffic_percentage=0.01, evaluation_period=5.0),
            CanaryStage(traffic_percentage=0.05, evaluation_period=5.0),
            CanaryStage(traffic_percentage=1.0, evaluation_period=5.0),
        ]
        deployer, _lb, clock = make_deployer(
            stages=stages,
            metric_evaluator=AlwaysHealthy(),
            evaluation_interval=2.0,
        )

        # Start deployment
        start = Event(
            time=Instant.Epoch,
            event_type="_canary_deploy_start",
            target=deployer,
            context={},
        )
        deployer.handle_event(start)

        # Stage 1: start
        stage_event = Event(
            time=Instant.Epoch,
            event_type="_canary_stage_start",
            target=deployer,
            context={},
        )
        deployer.handle_event(stage_event)
        assert deployer.state.canary_traffic_pct == 0.01

        # Advance time past evaluation period
        clock._current_time = Instant.from_seconds(6.0)
        eval_event = Event(
            time=Instant.from_seconds(6.0),
            event_type="_canary_evaluate",
            target=deployer,
            context={},
        )
        result = deployer.handle_event(eval_event)

        # Should advance to next stage
        assert deployer.state.current_stage == 1
        assert any(e.event_type == "_canary_stage_start" for e in result)

    def test_traffic_weights_set(self):
        stages = [CanaryStage(traffic_percentage=0.05, evaluation_period=5.0)]
        deployer, lb, _clock = make_deployer(stages=stages, metric_evaluator=AlwaysHealthy())

        # Start and begin stage
        start = Event(
            time=Instant.Epoch,
            event_type="_canary_deploy_start",
            target=deployer,
            context={},
        )
        deployer.handle_event(start)

        stage = Event(
            time=Instant.Epoch,
            event_type="_canary_stage_start",
            target=deployer,
            context={},
        )
        deployer.handle_event(stage)

        # Canary should have a weight set
        canary = deployer.canary
        weight = lb.strategy.get_weight(canary)
        assert weight >= 1


class TestRollback:
    def test_rollback_on_unhealthy(self):
        stages = [CanaryStage(traffic_percentage=0.05, evaluation_period=10.0)]
        deployer, _lb, clock = make_deployer(
            stages=stages,
            metric_evaluator=AlwaysUnhealthy(),
        )

        start = Event(
            time=Instant.Epoch,
            event_type="_canary_deploy_start",
            target=deployer,
            context={},
        )
        deployer.handle_event(start)

        stage = Event(
            time=Instant.Epoch,
            event_type="_canary_stage_start",
            target=deployer,
            context={},
        )
        deployer.handle_event(stage)

        # Evaluate - should fail and trigger rollback
        clock._current_time = Instant.from_seconds(3.0)
        eval_event = Event(
            time=Instant.from_seconds(3.0),
            event_type="_canary_evaluate",
            target=deployer,
            context={},
        )
        result = deployer.handle_event(eval_event)
        assert any(e.event_type == "_canary_rollback" for e in result)

    def test_rollback_removes_canary(self):
        stages = [CanaryStage(traffic_percentage=0.05, evaluation_period=10.0)]
        deployer, lb, _clock = make_deployer(stages=stages, metric_evaluator=AlwaysUnhealthy())

        start = Event(
            time=Instant.Epoch,
            event_type="_canary_deploy_start",
            target=deployer,
            context={},
        )
        deployer.handle_event(start)
        canary_name = deployer.canary.name

        rollback = Event(
            time=Instant.Epoch,
            event_type="_canary_rollback",
            target=deployer,
            context={},
        )
        deployer.handle_event(rollback)

        assert deployer.state.status == "rolled_back"
        assert canary_name not in [b.name for b in lb.all_backends]
        assert deployer.stats.deployments_rolled_back == 1


class TestFullPromotion:
    def test_promotion_removes_old_backends(self):
        stages = [CanaryStage(traffic_percentage=1.0, evaluation_period=5.0)]
        deployer, lb, clock = make_deployer(
            num_backends=3,
            stages=stages,
            metric_evaluator=AlwaysHealthy(),
        )

        start = Event(
            time=Instant.Epoch,
            event_type="_canary_deploy_start",
            target=deployer,
            context={},
        )
        deployer.handle_event(start)
        old_names = [b.name for b in deployer._baseline_backends]

        # Start stage
        stage = Event(
            time=Instant.Epoch,
            event_type="_canary_stage_start",
            target=deployer,
            context={},
        )
        deployer.handle_event(stage)

        # Advance past evaluation
        clock._current_time = Instant.from_seconds(6.0)
        eval_event = Event(
            time=Instant.from_seconds(6.0),
            event_type="_canary_evaluate",
            target=deployer,
            context={},
        )
        result = deployer.handle_event(eval_event)

        # Should trigger promote
        promote_events = [e for e in result if e.event_type == "_canary_promote"]
        assert len(promote_events) == 1

        # Execute promote
        deployer.handle_event(promote_events[0])

        # Old backends removed
        current_names = [b.name for b in lb.all_backends]
        for old_name in old_names:
            assert old_name not in current_names

        # Complete
        complete = Event(
            time=Instant.from_seconds(6.0),
            event_type="_canary_complete",
            target=deployer,
            context={},
        )
        deployer.handle_event(complete)
        assert deployer.state.status == "completed"


class TestErrorRateEvaluator:
    def test_healthy_canary(self):
        evaluator = ErrorRateEvaluator(max_error_rate=0.1)
        canary = FakeBackend("canary", error_rate=0.02)
        baselines = [FakeBackend(f"b{i}", error_rate=0.01) for i in range(3)]
        assert evaluator.is_healthy(canary, baselines)

    def test_unhealthy_canary(self):
        evaluator = ErrorRateEvaluator(max_error_rate=0.05)
        canary = FakeBackend("canary", error_rate=0.1)
        baselines = [FakeBackend(f"b{i}", error_rate=0.01) for i in range(3)]
        assert not evaluator.is_healthy(canary, baselines)


class TestLatencyEvaluator:
    def test_healthy_canary(self):
        evaluator = LatencyEvaluator(max_latency=1.0, threshold_multiplier=2.0)
        canary = FakeBackend("canary", avg_latency=0.1)
        baselines = [FakeBackend(f"b{i}", avg_latency=0.05) for i in range(3)]
        assert evaluator.is_healthy(canary, baselines)

    def test_unhealthy_canary_high_latency(self):
        evaluator = LatencyEvaluator(max_latency=0.5)
        canary = FakeBackend("canary", avg_latency=0.8)
        baselines = [FakeBackend(f"b{i}", avg_latency=0.05) for i in range(3)]
        assert not evaluator.is_healthy(canary, baselines)


class TestCustomEvaluators:
    def test_custom_evaluator(self):
        class ThresholdEvaluator:
            def __init__(self, threshold: float):
                self._threshold = threshold

            def is_healthy(self, canary, baseline_backends):
                return canary.average_service_time < self._threshold

        stages = [CanaryStage(traffic_percentage=0.1, evaluation_period=5.0)]
        deployer, _lb, clock = make_deployer(
            stages=stages,
            metric_evaluator=ThresholdEvaluator(threshold=0.2),
        )

        start = Event(
            time=Instant.Epoch,
            event_type="_canary_deploy_start",
            target=deployer,
            context={},
        )
        deployer.handle_event(start)

        stage = Event(
            time=Instant.Epoch,
            event_type="_canary_stage_start",
            target=deployer,
            context={},
        )
        deployer.handle_event(stage)

        # Evaluate - canary has 0.05 latency < 0.2 threshold
        clock._current_time = Instant.from_seconds(1.0)
        eval_event = Event(
            time=Instant.from_seconds(1.0),
            event_type="_canary_evaluate",
            target=deployer,
            context={},
        )
        result = deployer.handle_event(eval_event)
        # Should continue evaluating (not rollback)
        assert not any(e.event_type == "_canary_rollback" for e in result)
