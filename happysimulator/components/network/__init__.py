"""Network simulation components.

This package provides abstractions for modeling network behavior including
latency, bandwidth constraints, packet loss, and network topologies.
"""

from happysimulator.components.network.conditions import (
    cross_region_network,
    datacenter_network,
    internet_network,
    local_network,
    lossy_network,
    mobile_3g_network,
    mobile_4g_network,
    satellite_network,
    slow_network,
)
from happysimulator.components.network.link import NetworkLink, NetworkLinkStats
from happysimulator.components.network.network import LinkStats, Network, Partition

__all__ = [
    "LinkStats",
    "Network",
    "NetworkLink",
    "NetworkLinkStats",
    "Partition",
    "cross_region_network",
    "datacenter_network",
    "internet_network",
    "local_network",
    "lossy_network",
    "mobile_3g_network",
    "mobile_4g_network",
    "satellite_network",
    "slow_network",
]
