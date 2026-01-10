"""happy-simulator: A discrete-event simulation library for Python."""

import logging
import os

level = os.environ.get("HS_LOGGING", "INFO")

def get_logging_level(level):
    switcher = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    return switcher.get(level.upper(), logging.INFO)


logging.basicConfig(level=get_logging_level(level),
                    format='%(asctime)s - %(module)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("happysimulator.log"),
                        logging.StreamHandler()
                    ])

# Core exports
from happysimulator.core import (
    Simulation,
    Event,
    Entity,
    Instant,
    Clock,
)

# Load generation
from happysimulator.load import (
    Source,
    EventProvider,
    ConstantArrivalTimeProvider,
    PoissonArrivalTimeProvider,
)

# Components
from happysimulator.components import (
    Queue,
    QueueDriver,
    QueuedResource,
    FIFOQueue,
    LIFOQueue,
    PriorityQueue,
)

# Distributions
from happysimulator.distributions import (
    ConstantLatency,
    ExponentialLatency,
)

# Instrumentation
from happysimulator.instrumentation import (
    Data,
    Probe,
)

__all__ = [
    # Core
    "Simulation",
    "Event",
    "Entity",
    "Instant",
    "Clock",
    # Load
    "Source",
    "EventProvider",
    "ConstantArrivalTimeProvider",
    "PoissonArrivalTimeProvider",
    # Components
    "Queue",
    "QueueDriver",
    "QueuedResource",
    "FIFOQueue",
    "LIFOQueue",
    "PriorityQueue",
    # Distributions
    "ConstantLatency",
    "ExponentialLatency",
    # Instrumentation
    "Data",
    "Probe",
]
