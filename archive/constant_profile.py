from ..happysimulator.load.profile import Profile
from ..happysimulator.utils.instant import Instant


class ConstantProfile(Profile):
    def __init__(self, rate: float):
        if rate < 0.0:
            raise ValueError("Rate must be positive.")

        self._rate = rate

    def get_rate(self, time: Instant) -> float:
        return self._rate

    @classmethod
    def from_period(cls, time: Instant) -> 'ConstantProfile':
        return ConstantProfile(1 / time.to_seconds())
