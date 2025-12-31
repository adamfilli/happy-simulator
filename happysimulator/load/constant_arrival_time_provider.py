import math
import numpy as np

from happysimulator.load.arrival_time_provider import ArrivalTimeProvider

class ConstantArrivalTimeProvider(ArrivalTimeProvider):
    """
    Generates deterministic events. 
    The system accumulates exactly 1.0 unit of 'rate' before triggering.
    If rate is constant r=2, events happen every 0.5s exactly.
    """
    def _get_target_integral_value(self) -> float:
        return 1.0