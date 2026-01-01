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
"""
from ..archive.arrival_distribution import ArrivalDistribution
from .load.source import Source
from ..archive.measurement import Measurement
from .simulation import Simulation
from ..archive.stat import Stat
from .utils.instant import Instant

from ..archive.constant_latency import ConstantLatency
from ..archive.exponential_latency import ExponentialLatency
from ..archive.normal_latency import NormalLatency

from ..archive.constant_profile import ConstantProfile
from ..archive.rampup_profile import RampupProfile
from ..archive.sinusoid_profile import SinusoidProfile
from ..archive.spike_profile import SpikeProfile

from .entities import Client, Server, Queue, LifoQueue, QueuedServer

from ..archive.client_server_request_event import Request

__all__ = [
    "ArrivalDistribution",
    "Source",
    "Measurement",
    "Simulation",
    "Stat",
    "Instant",
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
"""

