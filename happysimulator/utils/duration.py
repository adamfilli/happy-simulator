from typing import Union


class Duration:
    def __init__(self, nanoseconds: int):
        self.nanoseconds = nanoseconds

    @classmethod
    def from_seconds(cls, seconds: Union[int, float]):
        if isinstance(seconds, int):
            return cls(seconds * 1_000_000_000)

        if isinstance(seconds, float):
            return cls(int(seconds * 1_000_000_000))

        raise TypeError("seconds must be int or float")

    def to_seconds(self) -> float:
        return float(self.nanoseconds) / 1_000_000_000

    # Arithmetic with other Durations or numeric seconds
    def __add__(self, other: Union['Duration', int, float]):
        if isinstance(other, (int, float)):
            return Duration(self.nanoseconds + int(other * 1_000_000_000))
        elif isinstance(other, Duration):
            return Duration(self.nanoseconds + other.nanoseconds)
        return NotImplemented

    def __radd__(self, other: Union['Duration', 'Instant', int, float]):
        # support Instant + Duration by handling Instant on the right-hand side
        try:
            from .instant import Instant
        except Exception:  # pragma: no cover - defensive import
            Instant = None

        if Instant is not None and isinstance(other, Instant):
            return Instant(other.nanoseconds + self.nanoseconds)

        return self.__add__(other)

    def __sub__(self, other: Union['Duration', int, float]):
        if isinstance(other, (int, float)):
            return Duration(self.nanoseconds - int(other * 1_000_000_000))
        elif isinstance(other, Duration):
            return Duration(self.nanoseconds - other.nanoseconds)
        return NotImplemented

    def __rsub__(self, other: Union['Instant', int, float]):
        try:
            from .instant import Instant
        except Exception:  # pragma: no cover - defensive import
            Instant = None

        if Instant is not None and isinstance(other, Instant):
            return Instant(other.nanoseconds - self.nanoseconds)

        if isinstance(other, (int, float)):
            return Duration(int(other * 1_000_000_000) - self.nanoseconds)

        return NotImplemented

    def __neg__(self):
        return Duration(-self.nanoseconds)

    def __abs__(self):
        return Duration(abs(self.nanoseconds))

    def __mul__(self, other: Union[int, float]):
        if isinstance(other, (int, float)):
            return Duration(int(self.nanoseconds * other))
        return NotImplemented

    def __rmul__(self, other: Union[int, float]):
        return self.__mul__(other)

    def __truediv__(self, other: Union[int, float, 'Duration']):
        if isinstance(other, (int, float)):
            return Duration(int(self.nanoseconds / other))
        if isinstance(other, Duration):
            return float(self.nanoseconds) / other.nanoseconds
        return NotImplemented

    # Equality & comparisons
    def __eq__(self, other):
        if not isinstance(other, Duration):
            return NotImplemented
        return self.nanoseconds == other.nanoseconds

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if not isinstance(other, Duration):
            return NotImplemented
        return self.nanoseconds < other.nanoseconds

    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)

    def __gt__(self, other):
        return not self.__le__(other)

    def __ge__(self, other):
        return not self.__lt__(other)

    def __repr__(self):
        return f"Duration({self.to_seconds()}s)"
