"""Network simulation components.

This package provides abstractions for modeling network behavior including
latency, bandwidth constraints, packet loss, and network topologies.
"""

from happysimulator.components.network.link import NetworkLink
from happysimulator.components.network.network import Network
from happysimulator.components.network.conditions import (
    local_network,
    datacenter_network,
    cross_region_network,
    internet_network,
    satellite_network,
    lossy_network,
    slow_network,
    mobile_3g_network,
    mobile_4g_network,
)

__all__ = [
    "NetworkLink",
    "Network",
    "local_network",
    "datacenter_network",
    "cross_region_network",
    "internet_network",
    "satellite_network",
    "lossy_network",
    "slow_network",
    "mobile_3g_network",
    "mobile_4g_network",
]
