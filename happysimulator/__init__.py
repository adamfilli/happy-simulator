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

# Core simulation types
from happysimulator.core import (
    Clock,
    Entity,
    Event,
    Instant,
    Simulation,
    Simulatable,
    simulatable,
)
from happysimulator.core.temporal import Duration

# Load generation
from happysimulator.load import (
    ConstantArrivalTimeProvider,
    ConstantRateProfile,
    DistributedFieldProvider,
    EventProvider,
    LinearRampProfile,
    PoissonArrivalTimeProvider,
    Profile,
    Source,
    SpikeProfile,
)

# Components
from happysimulator.components import (
    FIFOQueue,
    LIFOQueue,
    PriorityQueue,
    Queue,
    QueueDriver,
    QueuedResource,
    RandomRouter,
)

# Distributions
from happysimulator.distributions import (
    ConstantLatency,
    ExponentialLatency,
    PercentileFittedLatency,
    UniformDistribution,
    ValueDistribution,
    ZipfDistribution,
)

# Instrumentation
from happysimulator.instrumentation import (
    Data,
    Probe,
)

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
    "Simulatable",
    "simulatable",
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
