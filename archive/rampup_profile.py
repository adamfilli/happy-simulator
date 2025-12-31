from ..happysimulator.load.profile import Profile
from ..happysimulator.utils.instant import Instant

class RampupProfile(Profile):
    """
    Event profile with a linear ramp-up pattern.

    Parameters:
    - starting_rate (float): The initial rate of requests before the ramp-up. Must be positive.
    - rampup_factor (float): The rate at which the request rate increases over time.

    Attributes:
    - _rate (float): The initial rate of requests.
    - _factor (float): The rate at which the request rate increases over time.

    Methods:
    - get_rate(time: Time) -> float: Returns the request rate at a given time.
    """
    def __init__(self, starting_rate: float, rampup_factor: float):
        if starting_rate <= 0:
            raise ValueError("Starting Rate must be positive.")

        self._rate = starting_rate
        self._factor = rampup_factor

    def get_rate(self, time: Instant) -> float:
        """
        Returns the request rate at a given time.

        Parameters:
        - time (Time): The time at which to get the request rate.

        Returns:
        - float: The request rate at the specified time.
        """
        return self._rate + (time.to_seconds() * self._factor)
