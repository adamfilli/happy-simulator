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

# Re-exports for concise imports
from .arrival_distribution import ArrivalDistribution
from .generator import Generator
from .measurement import Measurement
from .simulation import Simulation
from .stat import Stat
from .time import Time

from .distribution.constant_latency import ConstantLatency
from .distribution.exponential_latency import ExponentialLatency
from .distribution.normal_latency import NormalLatency

from .profiles.constant_profile import ConstantProfile
from .profiles.rampup_profile import RampupProfile
from .profiles.sinusoid_profile import SinusoidProfile
from .profiles.spike_profile import SpikeProfile

from .entities import Client, Server, Queue, LifoQueue, QueuedServer

from .events.client_server_request_event import Request

__all__ = [
    "ArrivalDistribution",
    "Generator",
    "Measurement",
    "Simulation",
    "Stat",
    "Time",
    "ConstantLatency",
    "ExponentialLatency",
    "NormalLatency",
    "ConstantProfile",
    "RampupProfile",
    "SinusoidProfile",
    "SpikeProfile",
    "Client",
    "Server",
    "Queue",
    "LifoQueue",
    "QueuedServer",
    "Request",
]


