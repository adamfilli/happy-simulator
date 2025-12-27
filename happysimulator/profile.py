from abc import ABC, abstractmethod

from .time import Time

class Profile(ABC):
    @abstractmethod
    def get_rate(self, time: Time):
        pass