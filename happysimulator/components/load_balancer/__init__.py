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

from happysimulator.components.load_balancer.load_balancer import (
    LoadBalancer,
    LoadBalancerStats,
    BackendInfo,
)
from happysimulator.components.load_balancer.health_check import (
    HealthChecker,
    HealthCheckStats,
    BackendHealthState,
)
from happysimulator.components.load_balancer.strategies import (
    LoadBalancingStrategy,
    RoundRobin,
    WeightedRoundRobin,
    Random,
    LeastConnections,
    WeightedLeastConnections,
    LeastResponseTime,
    IPHash,
    ConsistentHash,
    PowerOfTwoChoices,
)

__all__ = [
    # Load Balancer
    "LoadBalancer",
    "LoadBalancerStats",
    "BackendInfo",
    # Health Checking
    "HealthChecker",
    "HealthCheckStats",
    "BackendHealthState",
    # Strategies
    "LoadBalancingStrategy",
    "RoundRobin",
    "WeightedRoundRobin",
    "Random",
    "LeastConnections",
    "WeightedLeastConnections",
    "LeastResponseTime",
    "IPHash",
    "ConsistentHash",
    "PowerOfTwoChoices",
]
