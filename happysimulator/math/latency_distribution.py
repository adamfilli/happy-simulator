import copy
from abc import ABC, abstractmethod

from happysimulator.utils.instant import Instant


class LatencyDistribution(ABC):
    def __init__(self, mean_latency: Instant):
        self._mean_latency = mean_latency.to_seconds()

    @abstractmethod
    def get_latency(self, current_time: Instant) -> Instant:
        pass

    def __add__(self, additional: float):
        new_instance = copy.deepcopy(self)
        new_instance._mean_latency += additional
        return new_instance

    def __sub__(self, subtraction: float):
        new_instance = copy.deepcopy(self)
        new_instance._mean_latency -= subtraction
        return new_instance