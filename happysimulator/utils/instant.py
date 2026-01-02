from typing import Union

class Instant:
    def __init__(self, nanoseconds: int):
        self.nanoseconds = nanoseconds

    @classmethod
    def from_seconds(cls, seconds):
        if isinstance(seconds, int):
            return cls(seconds * 1_000_000_000)

        if isinstance(seconds, float):
            return cls(int(seconds * 1_000_000_000))

    def to_seconds(self) -> float:
        return float(self.nanoseconds) / 1_000_000_000

    def __add__(self, other: Union['Instant', int, float]):
        if isinstance(other, (int, float)):
            return Instant(self.nanoseconds + int(other * 1_000_000_000))
        elif isinstance(other, Instant):
            return Instant(self.nanoseconds + other.nanoseconds)
        return NotImplemented

    def __sub__(self, other: Union['Instant', int, float]):
        if isinstance(other, (int, float)):
            return Instant(self.nanoseconds - int(other * 1_000_000_000))
        elif isinstance(other, Instant):
            return Instant(self.nanoseconds - other.nanoseconds)
        return NotImplemented

    # Equality
    def __eq__(self, other):
        if not isinstance(other, Instant):
            return NotImplemented
        return self.nanoseconds == other.nanoseconds

    def __ne__(self, other):
        return not self.__eq__(other)

    # Less than
    def __lt__(self, other):
        if not isinstance(other, Instant):
            return NotImplemented
        return self.nanoseconds < other.nanoseconds

    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)

    def __gt__(self, other):
        return not self.__le__(other)

    def __ge__(self, other):
        return not self.__lt__(other)


# A singleton representing positive infinity for Instants.
class _InfiniteInstant(Instant):
    def __init__(self):
        # use float('inf') so arithmetic with it gives inf
        super().__init__(float('inf'))

    def __add__(self, other: Union['Instant', int, float]):
        if isinstance(other, (int, float, Instant)):
            return self
        return NotImplemented

    def __sub__(self, other: Union['Instant', int, float]):
        if isinstance(other, Instant) and other.nanoseconds == float('inf'):
            return NotImplemented
        if isinstance(other, (int, float, Instant)):
            return self
        return NotImplemented

    def __eq__(self, other):
        if not isinstance(other, Instant):
            return NotImplemented
        return other.nanoseconds == float('inf')

    def __lt__(self, other):
        if not isinstance(other, Instant):
            return NotImplemented
        return False

    def to_seconds(self) -> float:
        return float('inf')

    def __repr__(self):
        return "Instant.Infinity"

# Singleton instances
Instant.Infinity = _InfiniteInstant()
Instant.Epoch = Instant(0)
