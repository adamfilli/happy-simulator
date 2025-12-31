from ..happysimulator.load.profile import Profile
from ..happysimulator.utils.instant import Instant

class SpikeProfile(Profile):
    """
    SpikeProfile simulates an event profile with a spike pattern.

    Parameters:
    - rampup_start (Time): The time when the ramp-up phase starts.
    - rampdown_start (Time): The time when the ramp-down phase starts.
    - starting_rate (float): The initial rate of requests before the ramp-up.
    - rampup_factor (float): The rate at which the request rate increases during the ramp-up phase.
    - rampdown_factor (float, optional): The rate at which the request rate decreases during the ramp-down phase. Defaults to 1.

    Attributes:
    - rampup_start (Time): The time when the ramp-up phase starts.
    - rampdown_start (Time): The time when the ramp-down phase starts.
    - starting_rate (float): The initial rate of requests before the ramp-up.
    - rampup_factor (float): The rate at which the request rate increases during the ramp-up phase.
    - rampdown_factor (float): The rate at which the request rate decreases during the ramp-down phase.
    - peak_rate (float): The peak request rate during the ramp-down phase.

    Methods:
    - get_rate(time: Time) -> float: Returns the request rate at a given time.
    """
    def __init__(self, rampup_start: Instant, rampdown_start: Instant, starting_rate: float, rampup_factor: float,
                 rampdown_factor: float = 1):
        self.rampup_start = rampup_start
        self.rampdown_start = rampdown_start
        self.starting_rate = starting_rate
        self.rampup_factor = rampup_factor
        self.rampdown_factor = rampdown_factor
        self.peak_rate = starting_rate + (rampdown_start - rampup_start).to_seconds() * rampup_factor

    def get_rate(self, time: Instant) -> float:
        """
        Returns the request rate at a given time.

        Parameters:
        - time (Time): The time at which to get the request rate.

        Returns:
        - float: The request rate at the specified time.
        """
        if time < self.rampup_start:
            return self.starting_rate
        elif self.rampup_start <= time < self.rampdown_start:
            return self.starting_rate + (self.rampup_factor * (time - self.rampup_start).to_seconds())
        else:  # After rampdown_start
            return max(self.starting_rate, self.peak_rate - (self.rampdown_factor * (time - self.rampdown_start).to_seconds()))