"""happy-simulator: A discrete-event simulation library for Python."""

__version__ = "0.1.3"

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
from happysimulator.load import PoissonArrivalTimeProvider, ConstantRateProfile, LinearRampProfile, SpikeProfile
from happysimulator.components import LIFOQueue, PriorityQueue, QueuedResource, RandomRouter
from happysimulator.distributions import ConstantLatency, ExponentialLatency, PercentileFittedLatency
from happysimulator.distributions import ZipfDistribution, UniformDistribution, ValueDistribution
from happysimulator.load import DistributedFieldProvider
from happysimulator.core.temporal import Duration

__all__ = [
    # Package metadata
    "__version__",
    # Core
    "Simulation",
    "Event",
    "Entity",
    "Instant",
    "Duration",
    "Clock",
    # Load
    "Source",
    "EventProvider",
    "Profile",
    "ConstantRateProfile",
    "LinearRampProfile",
    "SpikeProfile",
    "ConstantArrivalTimeProvider",
    "PoissonArrivalTimeProvider",
    # Components
    "Queue",
    "QueueDriver",
    "QueuedResource",
    "FIFOQueue",
    "LIFOQueue",
    "PriorityQueue",
    "RandomRouter",
    # Distributions
    "ConstantLatency",
    "ExponentialLatency",
    "PercentileFittedLatency",
    "ZipfDistribution",
    "UniformDistribution",
    "ValueDistribution",
    # Load providers
    "DistributedFieldProvider",
    # Instrumentation
    "Data",
    "Probe",
]
