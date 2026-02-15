"""Canary deployment with progressive traffic shifting and metric evaluation.

Creates a canary instance and progressively shifts traffic through
configurable stages while monitoring health metrics. Rolls back
automatically if the canary degrades.

Example:
    from happysimulator.components.deployment import (
        CanaryDeployer, CanaryStage, ErrorRateEvaluator,
    )

    deployer = CanaryDeployer(
        name="canary", load_balancer=lb,
        server_factory=lambda name: Server(name=name, ...),
        stages=[
            CanaryStage(traffic_percentage=0.01, evaluation_period=10.0),
            CanaryStage(traffic_percentage=0.05, evaluation_period=10.0),
            CanaryStage(traffic_percentage=0.25, evaluation_period=10.0),
            CanaryStage(traffic_percentage=1.0, evaluation_period=10.0),
        ],
    )
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Duration, Instant

logger = logging.getLogger(__name__)


@dataclass
class CanaryStage:
    """Definition of a canary traffic stage.

    Attributes:
        traffic_percentage: Fraction of traffic to send to canary (0.0-1.0).
        evaluation_period: Seconds to observe at this stage before advancing.
    """

    traffic_percentage: float
    evaluation_period: float = 30.0


@dataclass
class CanaryState:
    """Current state of a canary deployment."""

    status: str = "idle"  # idle, in_progress, promoting, rolled_back, completed
    current_stage: int = 0
    total_stages: int = 0
    canary_traffic_pct: float = 0.0


@runtime_checkable
class MetricEvaluator(Protocol):
    """Protocol for evaluating canary health."""

    def is_healthy(self, canary: Entity, baseline_backends: list[Entity]) -> bool:
        """Evaluate whether the canary is healthy.

        Args:
            canary: The canary backend entity.
            baseline_backends: The existing baseline backends.

        Returns:
            True if the canary is performing acceptably.
        """
        ...


class ErrorRateEvaluator:
    """Evaluate canary health based on error/failure rates.

    Compares canary failure rate to baseline average. If canary exceeds
    the threshold multiplier over baseline, it's considered unhealthy.
    """

    def __init__(self, max_error_rate: float = 0.05, threshold_multiplier: float = 2.0):
        self._max_error_rate = max_error_rate
        self._threshold_multiplier = threshold_multiplier

    def is_healthy(self, canary: Entity, baseline_backends: list[Entity]) -> bool:
        canary_info = self._get_error_rate(canary)
        if canary_info > self._max_error_rate:
            return False

        if not baseline_backends:
            return canary_info <= self._max_error_rate

        baseline_rates = [self._get_error_rate(b) for b in baseline_backends]
        avg_baseline = sum(baseline_rates) / len(baseline_rates) if baseline_rates else 0

        if avg_baseline > 0:
            return canary_info <= avg_baseline * self._threshold_multiplier
        return canary_info <= self._max_error_rate

    def _get_error_rate(self, backend: Entity) -> float:
        if hasattr(backend, "stats"):
            stats = backend.stats
            total = getattr(stats, "requests_completed", 0) + getattr(stats, "requests_rejected", 0)
            failures = getattr(stats, "requests_rejected", 0) + getattr(stats, "total_failures", 0)
            if total > 0:
                return failures / total
        return 0.0


class LatencyEvaluator:
    """Evaluate canary health based on response latency.

    Compares canary latency to baseline. If canary exceeds threshold
    multiplier over baseline average, it's unhealthy.
    """

    def __init__(self, max_latency: float = 1.0, threshold_multiplier: float = 1.5):
        self._max_latency = max_latency
        self._threshold_multiplier = threshold_multiplier

    def is_healthy(self, canary: Entity, baseline_backends: list[Entity]) -> bool:
        canary_latency = self._get_avg_latency(canary)
        if canary_latency > self._max_latency:
            return False

        if not baseline_backends:
            return canary_latency <= self._max_latency

        baseline_latencies = [self._get_avg_latency(b) for b in baseline_backends]
        avg_baseline = (
            sum(baseline_latencies) / len(baseline_latencies) if baseline_latencies else 0
        )

        if avg_baseline > 0:
            return canary_latency <= avg_baseline * self._threshold_multiplier
        return canary_latency <= self._max_latency

    def _get_avg_latency(self, backend: Entity) -> float:
        if hasattr(backend, "average_service_time"):
            return backend.average_service_time
        return 0.0


@dataclass(frozen=True)
class CanaryDeployerStats:
    """Statistics tracked by CanaryDeployer."""

    deployments_started: int = 0
    deployments_completed: int = 0
    deployments_rolled_back: int = 0
    stages_completed: int = 0
    evaluations_performed: int = 0
    evaluations_passed: int = 0
    evaluations_failed: int = 0


class CanaryDeployer(Entity):
    """Canary deployment with progressive traffic shifting.

    Creates a canary instance and progressively shifts traffic through
    stages while monitoring metrics. Rolls back if metrics degrade.

    Default stages: [1%, 5%, 25%, 100%].

    Attributes:
        name: Deployer identifier.
        stats: Frozen statistics snapshot.
        state: Current deployment state.
    """

    DEFAULT_STAGES = [
        CanaryStage(traffic_percentage=0.01, evaluation_period=30.0),
        CanaryStage(traffic_percentage=0.05, evaluation_period=30.0),
        CanaryStage(traffic_percentage=0.25, evaluation_period=30.0),
        CanaryStage(traffic_percentage=1.0, evaluation_period=30.0),
    ]

    def __init__(
        self,
        name: str,
        load_balancer: Entity,
        server_factory: Callable[[str], Entity],
        stages: list[CanaryStage] | None = None,
        metric_evaluator: MetricEvaluator | None = None,
        evaluation_interval: float = 5.0,
    ):
        """Initialize the canary deployer.

        Args:
            name: Deployer identifier.
            load_balancer: LoadBalancer to manage.
            server_factory: Creates new server instances.
            stages: Traffic stages (default [1%, 5%, 25%, 100%]).
            metric_evaluator: Health evaluator (default ErrorRateEvaluator).
            evaluation_interval: Seconds between metric evaluations.
        """
        super().__init__(name)

        self._load_balancer = load_balancer
        self._server_factory = server_factory
        self._stages = stages or list(self.DEFAULT_STAGES)
        self._metric_evaluator = metric_evaluator or ErrorRateEvaluator()
        self._evaluation_interval = evaluation_interval

        self._canary: Entity | None = None
        self._baseline_backends: list[Entity] = []
        self._stage_start_time: Instant | None = None
        self._original_strategy = None

        self._deployments_started = 0
        self._deployments_completed = 0
        self._deployments_rolled_back = 0
        self._stages_completed = 0
        self._evaluations_performed = 0
        self._evaluations_passed = 0
        self._evaluations_failed = 0
        self.state = CanaryState()

        logger.debug(
            "[%s] CanaryDeployer initialized: stages=%d",
            name,
            len(self._stages),
        )

    @property
    def stats(self) -> CanaryDeployerStats:
        """Return a frozen snapshot of current statistics."""
        return CanaryDeployerStats(
            deployments_started=self._deployments_started,
            deployments_completed=self._deployments_completed,
            deployments_rolled_back=self._deployments_rolled_back,
            stages_completed=self._stages_completed,
            evaluations_performed=self._evaluations_performed,
            evaluations_passed=self._evaluations_passed,
            evaluations_failed=self._evaluations_failed,
        )

    @property
    def canary(self) -> Entity | None:
        """The canary instance, if active."""
        return self._canary

    def deploy(self) -> Event:
        """Start a canary deployment.

        Returns:
            The deployment start event.
        """
        return Event(
            time=self.now if self._clock is not None else Instant.Epoch,
            event_type="_canary_deploy_start",
            target=self,
            context={},
        )

    def handle_event(self, event: Event) -> list[Event] | None:
        et = event.event_type

        if et == "_canary_deploy_start":
            return self._start_deployment()
        if et == "_canary_stage_start":
            return self._start_stage()
        if et == "_canary_evaluate":
            return self._evaluate()
        if et == "_canary_promote":
            return self._promote()
        if et == "_canary_rollback":
            return self._do_rollback()
        if et == "_canary_complete":
            return self._complete()
        return None

    def _start_deployment(self) -> list[Event]:
        """Create canary and begin first stage."""
        self._baseline_backends = list(self._load_balancer.all_backends)

        # Create canary instance
        canary_name = f"{self.name}_canary"
        self._canary = self._server_factory(canary_name)

        if self._clock is not None:
            self._canary.set_clock(self._clock)

        self._load_balancer.add_backend(self._canary)

        self.state = CanaryState(
            status="in_progress",
            total_stages=len(self._stages),
        )
        self._deployments_started += 1

        logger.info("[%s] Canary deployment started", self.name)

        return [
            Event(
                time=self.now,
                event_type="_canary_stage_start",
                target=self,
                context={},
            )
        ]

    def _start_stage(self) -> list[Event]:
        """Configure traffic splitting for the current stage."""
        stage_idx = self.state.current_stage
        if stage_idx >= len(self._stages):
            return [
                Event(
                    time=self.now,
                    event_type="_canary_promote",
                    target=self,
                    context={},
                )
            ]

        stage = self._stages[stage_idx]
        self.state.canary_traffic_pct = stage.traffic_percentage
        self._stage_start_time = self.now

        # Set canary weight to achieve target traffic percentage
        self._set_traffic_weight(stage.traffic_percentage)

        logger.info(
            "[%s] Stage %d: %.0f%% traffic to canary",
            self.name,
            stage_idx + 1,
            stage.traffic_percentage * 100,
        )

        # Schedule periodic evaluation
        return [
            Event(
                time=self.now + Duration.from_seconds(self._evaluation_interval),
                event_type="_canary_evaluate",
                target=self,
                context={},
            )
        ]

    def _evaluate(self) -> list[Event]:
        """Evaluate canary health at current stage."""
        if self.state.status != "in_progress":
            return []

        self._evaluations_performed += 1

        is_healthy = self._metric_evaluator.is_healthy(
            self._canary,
            self._baseline_backends,
        )

        if not is_healthy:
            self._evaluations_failed += 1
            logger.warning("[%s] Canary evaluation failed", self.name)
            return [
                Event(
                    time=self.now,
                    event_type="_canary_rollback",
                    target=self,
                    context={},
                )
            ]

        self._evaluations_passed += 1

        # Check if stage evaluation period is complete
        stage = self._stages[self.state.current_stage]
        elapsed = (self.now - self._stage_start_time).to_seconds()

        if elapsed >= stage.evaluation_period:
            # Advance to next stage
            self._stages_completed += 1
            self.state.current_stage += 1

            if self.state.current_stage >= len(self._stages):
                return [
                    Event(
                        time=self.now,
                        event_type="_canary_promote",
                        target=self,
                        context={},
                    )
                ]

            return [
                Event(
                    time=self.now,
                    event_type="_canary_stage_start",
                    target=self,
                    context={},
                )
            ]

        # Continue evaluating
        return [
            Event(
                time=self.now + Duration.from_seconds(self._evaluation_interval),
                event_type="_canary_evaluate",
                target=self,
                context={},
            )
        ]

    def _promote(self) -> list[Event]:
        """Promote canary to production: remove old backends."""
        self.state.status = "promoting"

        for old_backend in self._baseline_backends:
            self._load_balancer.remove_backend(old_backend)

        # Reset weights
        self._reset_weights()

        return [
            Event(
                time=self.now,
                event_type="_canary_complete",
                target=self,
                context={},
            )
        ]

    def _do_rollback(self) -> list[Event]:
        """Roll back: remove canary, restore weights."""
        self.state.status = "rolled_back"
        self._deployments_rolled_back += 1

        if self._canary is not None:
            self._load_balancer.remove_backend(self._canary)

        # Reset weights
        self._reset_weights()

        logger.info("[%s] Canary deployment rolled back", self.name)
        return []

    def _complete(self) -> list[Event]:
        """Mark deployment as completed."""
        self.state.status = "completed"
        self._deployments_completed += 1
        logger.info("[%s] Canary deployment completed", self.name)
        return []

    def _set_traffic_weight(self, canary_pct: float) -> None:
        """Set canary weight to achieve the target traffic percentage.

        Uses WeightedRoundRobin strategy on the load balancer.
        """
        strategy = (
            self._load_balancer.strategy if hasattr(self._load_balancer, "strategy") else None
        )

        if strategy is not None and hasattr(strategy, "set_weight"):
            # Calculate weights: if canary gets pct, each baseline gets
            # (1-pct) / num_baseline of traffic
            num_baseline = len(self._baseline_backends)
            if num_baseline == 0:
                return

            # Use integer weights: canary gets proportional weight
            # e.g., for 1% with 3 baselines: canary=1, each baseline=33
            if canary_pct >= 1.0:
                # Full promotion: equal weights
                for b in self._baseline_backends:
                    strategy.set_weight(b, 1)
                if self._canary:
                    strategy.set_weight(self._canary, 1)
            else:
                # Scale to integer weights
                canary_weight = max(1, int(canary_pct * 100))
                baseline_weight = max(1, int((1.0 - canary_pct) * 100 / num_baseline))

                for b in self._baseline_backends:
                    strategy.set_weight(b, baseline_weight)
                if self._canary:
                    strategy.set_weight(self._canary, canary_weight)

    def _reset_weights(self) -> None:
        """Reset all backend weights to 1."""
        strategy = (
            self._load_balancer.strategy if hasattr(self._load_balancer, "strategy") else None
        )
        if strategy is not None and hasattr(strategy, "set_weight"):
            for b in self._load_balancer.all_backends:
                strategy.set_weight(b, 1)
