"""happy-simulator: A discrete-event simulation library for Python."""

__version__ = "0.1.0"

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

# Public facade exports (keeps common imports stable)
from happysimulator.api import (
    ConstantArrivalTimeProvider,
    Data,
    Entity,
    Event,
    EventProvider,
    FIFOQueue,
    Instant,
    Probe,
    Profile,
    Queue,
    QueueDriver,
    Simulation,
    Source,
)

# Additional top-level convenience exports
from happysimulator.core import Clock
from happysimulator.load import PoissonArrivalTimeProvider, ConstantRateProfile, LinearRampProfile
from happysimulator.components import LIFOQueue, PriorityQueue, QueuedResource
from happysimulator.distributions import ConstantLatency, ExponentialLatency

__all__ = [
    # Package metadata
    "__version__",
    # Core
    "Simulation",
    "Event",
    "Entity",
    "Instant",
    "Clock",
    # Load
    "Source",
    "EventProvider",
    "Profile",
    "ConstantRateProfile",
    "LinearRampProfile",
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
