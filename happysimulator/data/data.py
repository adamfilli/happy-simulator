# happysimulator/data/data.py
from typing import Any, List, Tuple
from happysimulator.utils.instant import Instant

class Data:
    def __init__(self):
        self._samples: List[Tuple[float, Any]] = []

    def add_stat(self, value: Any, time: Instant):
        """
        Records a data point. 
        """
        self._samples.append((time.to_seconds(), value))

    @property
    def values(self):
        return self._samples