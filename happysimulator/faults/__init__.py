"""Fault injection framework for declarative fault scheduling.

Provides fault types for nodes, networks, and resources, plus a
``FaultSchedule`` entity that generates activation/deactivation events
during simulation bootstrap.
"""

from happysimulator.faults.fault import (
    Fault,
    FaultContext,
    FaultHandle,
    FaultStats,
)
from happysimulator.faults.network_faults import (
    InjectLatency,
    InjectPacketLoss,
    NetworkPartition,
    RandomPartition,
)
from happysimulator.faults.node_faults import CrashNode, PauseNode
from happysimulator.faults.resource_faults import ReduceCapacity
from happysimulator.faults.schedule import FaultSchedule

__all__ = [
    "CrashNode",
    "Fault",
    "FaultContext",
    "FaultHandle",
    "FaultSchedule",
    "FaultStats",
    "InjectLatency",
    "InjectPacketLoss",
    "NetworkPartition",
    "PauseNode",
    "RandomPartition",
    "ReduceCapacity",
]
