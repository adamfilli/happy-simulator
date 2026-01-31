"""Network simulation components.

This package provides abstractions for modeling network behavior including
latency, bandwidth constraints, packet loss, and network topologies.
"""

from happysimulator.components.network.link import NetworkLink
from happysimulator.components.network.network import Network

__all__ = ["NetworkLink", "Network"]
