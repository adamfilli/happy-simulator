"""Load balancer components for traffic distribution.

This package provides load balancing abstractions with pluggable
strategies for distributing requests across backend servers.

Example:
    from happysimulator.components.load_balancer import (
        LoadBalancer,
        HealthChecker,
        RoundRobin,
        LeastConnections,
    )

    # Create backends
    servers = [Server(name=f"server_{i}", ...) for i in range(3)]

    # Create load balancer with round-robin strategy
    lb = LoadBalancer(
        name="api_lb",
        backends=servers,
        strategy=RoundRobin(),
    )

    # Optionally add health checking
    health_checker = HealthChecker(
        name="health_check",
        load_balancer=lb,
        interval=5.0,
        timeout=1.0,
    )
"""

from happysimulator.components.load_balancer.health_check import (
    BackendHealthState,
    HealthChecker,
    HealthCheckStats,
)
from happysimulator.components.load_balancer.load_balancer import (
    BackendInfo,
    LoadBalancer,
    LoadBalancerStats,
)
from happysimulator.components.load_balancer.strategies import (
    ConsistentHash,
    IPHash,
    LeastConnections,
    LeastResponseTime,
    LoadBalancingStrategy,
    PowerOfTwoChoices,
    Random,
    RoundRobin,
    WeightedLeastConnections,
    WeightedRoundRobin,
)

__all__ = [
    "BackendHealthState",
    "BackendInfo",
    "ConsistentHash",
    "HealthCheckStats",
    # Health Checking
    "HealthChecker",
    "IPHash",
    "LeastConnections",
    "LeastResponseTime",
    # Load Balancer
    "LoadBalancer",
    "LoadBalancerStats",
    # Strategies
    "LoadBalancingStrategy",
    "PowerOfTwoChoices",
    "Random",
    "RoundRobin",
    "WeightedLeastConnections",
    "WeightedRoundRobin",
]
