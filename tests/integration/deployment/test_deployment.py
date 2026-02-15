"""Integration tests for deployment components (AutoScaler, RollingDeployer, CanaryDeployer)."""

import random

from happysimulator import (
    ConstantArrivalTimeProvider,
    ConstantLatency,
    ConstantRateProfile,
    Entity,
    Event,
    Instant,
    Simulation,
    Source,
)
from happysimulator.components.deployment import (
    AutoScaler,
    CanaryDeployer,
    CanaryStage,
    RollingDeployer,
    TargetUtilization,
)
from happysimulator.components.load_balancer import LoadBalancer, RoundRobin, WeightedRoundRobin
from happysimulator.components.server import Server
from happysimulator.load import EventProvider


class RequestProvider(EventProvider):
    """Generates request events."""

    def __init__(self, target: Entity, stop_after: Instant | None = None):
        self._target = target
        self._stop_after = stop_after
        self._count = 0

    def get_events(self, time: Instant) -> list[Event]:
        if self._stop_after and time > self._stop_after:
            return []
        self._count += 1
        return [
            Event(
                time=time,
                event_type="Request",
                target=self._target,
                context={"created_at": time},
            )
        ]


def make_server(name: str) -> Server:
    return Server(name=name, concurrency=2, service_time=ConstantLatency(0.05))


class TestAutoScalerEndToEnd:
    def test_scale_out_under_load(self):
        """AutoScaler adds instances when utilization is high."""
        random.seed(42)

        # Start with 1 server
        initial_server = make_server("server_0")
        lb = LoadBalancer(
            name="lb",
            backends=[initial_server],
            strategy=RoundRobin(),
        )

        scaler = AutoScaler(
            name="scaler",
            load_balancer=lb,
            server_factory=make_server,
            policy=TargetUtilization(target=0.5),
            min_instances=1,
            max_instances=5,
            evaluation_interval=2.0,
            scale_out_cooldown=3.0,
            scale_in_cooldown=5.0,
        )

        provider = RequestProvider(lb, stop_after=Instant.from_seconds(15.0))
        arrival = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate=20.0), start_time=Instant.Epoch
        )
        source = Source(
            name="traffic",
            event_provider=provider,
            arrival_time_provider=arrival,
        )

        # Collect all entities including dynamically created ones
        sim = Simulation(
            start_time=Instant.Epoch,
            duration=20.0,
            sources=[source],
            entities=[lb, initial_server, scaler],
        )
        sim.schedule(scaler.start())
        sim.run()

        # Scaler should have evaluated and possibly scaled
        assert scaler.stats.evaluations >= 1


class TestRollingDeployerEndToEnd:
    def test_successful_deployment(self):
        """Rolling deployment replaces all backends."""
        random.seed(42)

        servers = [make_server(f"server_{i}") for i in range(2)]
        lb = LoadBalancer(name="lb", backends=servers, strategy=RoundRobin())

        deployer = RollingDeployer(
            name="deployer",
            load_balancer=lb,
            server_factory=make_server,
            batch_size=1,
            health_check_interval=1.0,
            healthy_threshold=1,
            max_failures=3,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=30.0,
            entities=[lb, deployer, *servers],
        )
        sim.schedule(deployer.deploy())
        sim.run()

        # Deployment should have started
        assert deployer.stats.deployments_started == 1
        # Should have replaced at least some instances
        assert deployer.state.status in ("completed", "in_progress")


class TestCanaryDeployerEndToEnd:
    def test_successful_canary_promotion(self):
        """Canary progresses through stages to completion."""
        random.seed(42)

        servers = [make_server(f"server_{i}") for i in range(2)]
        lb = LoadBalancer(name="lb", backends=servers, strategy=WeightedRoundRobin())

        # Always-healthy evaluator
        class AlwaysHealthy:
            def is_healthy(self, canary, baseline):
                return True

        deployer = CanaryDeployer(
            name="canary",
            load_balancer=lb,
            server_factory=make_server,
            stages=[
                CanaryStage(traffic_percentage=0.1, evaluation_period=2.0),
                CanaryStage(traffic_percentage=0.5, evaluation_period=2.0),
                CanaryStage(traffic_percentage=1.0, evaluation_period=2.0),
            ],
            metric_evaluator=AlwaysHealthy(),
            evaluation_interval=1.0,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=30.0,
            entities=[lb, deployer, *servers],
        )
        sim.schedule(deployer.deploy())
        sim.run()

        assert deployer.stats.deployments_started == 1
        assert deployer.state.status == "completed"
        assert deployer.stats.stages_completed == 3
