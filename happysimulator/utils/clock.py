from happysimulator.utils.instant import Instant

class Clock:
    def __init__(self, start_time: Instant):
        self._current_time = start_time

    @property
    def now(self) -> Instant:
        return self._current_time

    def update(self, time: Instant):
        self._current_time = time