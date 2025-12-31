from abc import ABC, abstractmethod

from ..utils.instant import Instant

class Profile(ABC):
    @abstractmethod
    def get_rate(self, time: Instant) -> float:
        pass