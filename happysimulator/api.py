"""Curated public API surface for end-users.

This module exists to keep consumer imports short and stable:

    from happysimulator.api import Simulation, Queue, Source

The intent is to re-export only the most commonly used symbols.
More specialized utilities should continue to be imported from their
implementation modules.
"""

from __future__ import annotations

# Core
from happysimulator.core import Entity, Event, Instant, Simulation

# Components
from happysimulator.components import FIFOQueue, Queue, QueueDriver

# Instrumentation
from happysimulator.instrumentation import Data, Probe

# Load
from happysimulator.load import ConstantArrivalTimeProvider, EventProvider, Profile, Source

__all__ = [
    # Core
    "Entity",
    "Event",
    "Instant",
    "Simulation",
    # Components
    "Queue",
    "QueueDriver",
    "FIFOQueue",
    # Instrumentation
    "Data",
    "Probe",
    # Load
    "EventProvider",
    "Profile",
    "ConstantArrivalTimeProvider",
    "Source",
]
