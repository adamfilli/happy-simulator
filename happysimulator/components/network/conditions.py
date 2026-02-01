"""Predefined network profiles for common scenarios.

Factory functions that create NetworkLink instances with realistic
characteristics for various network environments. Use these as starting
points for simulations or as building blocks for custom configurations.
"""

from happysimulator.components.network.link import NetworkLink
from happysimulator.distributions.constant import ConstantLatency
from happysimulator.distributions.exponential import ExponentialLatency


def local_network(name: str = "local") -> NetworkLink:
    """Create a local/loopback network link.

    Characteristics:
    - ~0.1ms latency (essentially instantaneous)
    - 1 Gbps bandwidth
    - No packet loss
    - No jitter

    Suitable for: localhost connections, in-process communication,
    same-machine services.

    Args:
        name: Identifier for the link.

    Returns:
        NetworkLink configured for local network conditions.
    """
    return NetworkLink(
        name=name,
        latency=ConstantLatency(0.0001),  # 0.1ms
        bandwidth_bps=1_000_000_000,  # 1 Gbps
        packet_loss_rate=0.0,
        jitter=None,
    )


def datacenter_network(name: str = "datacenter") -> NetworkLink:
    """Create a datacenter network link.

    Characteristics:
    - ~0.5ms latency
    - 10 Gbps bandwidth
    - No packet loss (reliable internal network)
    - Minimal jitter

    Suitable for: same-datacenter communication, rack-to-rack,
    internal microservices.

    Args:
        name: Identifier for the link.

    Returns:
        NetworkLink configured for datacenter network conditions.
    """
    return NetworkLink(
        name=name,
        latency=ConstantLatency(0.0005),  # 0.5ms
        bandwidth_bps=10_000_000_000,  # 10 Gbps
        packet_loss_rate=0.0,
        jitter=ConstantLatency(0.0001),  # 0.1ms jitter
    )


def cross_region_network(name: str = "cross_region") -> NetworkLink:
    """Create a cross-region/cross-datacenter network link.

    Characteristics:
    - ~50ms latency (continental distance)
    - 1 Gbps bandwidth
    - Very low packet loss (0.01%)
    - Moderate jitter

    Suitable for: multi-region deployments, cross-datacenter replication,
    geo-distributed systems.

    Args:
        name: Identifier for the link.

    Returns:
        NetworkLink configured for cross-region network conditions.
    """
    return NetworkLink(
        name=name,
        latency=ConstantLatency(0.050),  # 50ms
        bandwidth_bps=1_000_000_000,  # 1 Gbps
        packet_loss_rate=0.0001,  # 0.01%
        jitter=ExponentialLatency(0.005),  # 5ms mean jitter
    )


def internet_network(name: str = "internet") -> NetworkLink:
    """Create an internet/WAN network link.

    Characteristics:
    - ~100ms latency (intercontinental)
    - 100 Mbps bandwidth (typical business connection)
    - Low packet loss (0.1%)
    - Significant jitter

    Suitable for: public internet connections, client-server communication,
    external API calls.

    Args:
        name: Identifier for the link.

    Returns:
        NetworkLink configured for internet network conditions.
    """
    return NetworkLink(
        name=name,
        latency=ConstantLatency(0.100),  # 100ms
        bandwidth_bps=100_000_000,  # 100 Mbps
        packet_loss_rate=0.001,  # 0.1%
        jitter=ExponentialLatency(0.020),  # 20ms mean jitter
    )


def satellite_network(name: str = "satellite") -> NetworkLink:
    """Create a satellite network link.

    Characteristics:
    - ~600ms latency (geostationary orbit round-trip)
    - 10 Mbps bandwidth (limited satellite capacity)
    - Moderate packet loss (0.5%)
    - High jitter

    Suitable for: satellite internet, remote/maritime locations,
    backup connectivity.

    Args:
        name: Identifier for the link.

    Returns:
        NetworkLink configured for satellite network conditions.
    """
    return NetworkLink(
        name=name,
        latency=ConstantLatency(0.600),  # 600ms
        bandwidth_bps=10_000_000,  # 10 Mbps
        packet_loss_rate=0.005,  # 0.5%
        jitter=ExponentialLatency(0.050),  # 50ms mean jitter
    )


def lossy_network(
    loss_rate: float,
    name: str = "lossy",
    base_latency: float = 0.010,
) -> NetworkLink:
    """Create a network link with configurable packet loss.

    Useful for testing retry logic, fault tolerance, and degraded
    network conditions.

    Args:
        loss_rate: Probability [0, 1] of dropping each packet.
        name: Identifier for the link.
        base_latency: Base latency in seconds (default 10ms).

    Returns:
        NetworkLink configured with the specified loss rate.

    Raises:
        ValueError: If loss_rate is not in [0, 1].
    """
    if loss_rate < 0.0 or loss_rate > 1.0:
        raise ValueError(f"loss_rate must be in [0, 1], got {loss_rate}")

    return NetworkLink(
        name=name,
        latency=ConstantLatency(base_latency),
        bandwidth_bps=100_000_000,  # 100 Mbps
        packet_loss_rate=loss_rate,
        jitter=None,
    )


def slow_network(
    latency_seconds: float,
    name: str = "slow",
    bandwidth_bps: float = 1_000_000,
) -> NetworkLink:
    """Create a slow network link with configurable latency.

    Useful for testing timeout handling and latency-sensitive code.

    Args:
        latency_seconds: One-way latency in seconds.
        name: Identifier for the link.
        bandwidth_bps: Bandwidth in bits per second (default 1 Mbps).

    Returns:
        NetworkLink configured with the specified latency.
    """
    return NetworkLink(
        name=name,
        latency=ConstantLatency(latency_seconds),
        bandwidth_bps=bandwidth_bps,
        packet_loss_rate=0.0,
        jitter=None,
    )


def mobile_3g_network(name: str = "mobile_3g") -> NetworkLink:
    """Create a 3G mobile network link.

    Characteristics:
    - ~100ms latency
    - 2 Mbps bandwidth
    - Moderate packet loss (0.5%)
    - High jitter

    Suitable for: mobile app testing, degraded connectivity scenarios.

    Args:
        name: Identifier for the link.

    Returns:
        NetworkLink configured for 3G mobile network conditions.
    """
    return NetworkLink(
        name=name,
        latency=ConstantLatency(0.100),  # 100ms
        bandwidth_bps=2_000_000,  # 2 Mbps
        packet_loss_rate=0.005,  # 0.5%
        jitter=ExponentialLatency(0.030),  # 30ms mean jitter
    )


def mobile_4g_network(name: str = "mobile_4g") -> NetworkLink:
    """Create a 4G/LTE mobile network link.

    Characteristics:
    - ~50ms latency
    - 20 Mbps bandwidth
    - Low packet loss (0.1%)
    - Moderate jitter

    Suitable for: mobile app testing, typical mobile conditions.

    Args:
        name: Identifier for the link.

    Returns:
        NetworkLink configured for 4G mobile network conditions.
    """
    return NetworkLink(
        name=name,
        latency=ConstantLatency(0.050),  # 50ms
        bandwidth_bps=20_000_000,  # 20 Mbps
        packet_loss_rate=0.001,  # 0.1%
        jitter=ExponentialLatency(0.015),  # 15ms mean jitter
    )
