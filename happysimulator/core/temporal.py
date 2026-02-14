"""Time representation with nanosecond precision.

This module provides two distinct time concepts:

- Instant: A point in time (e.g., "the event occurs at T=5s")
- Duration: A length of time (e.g., "the operation takes 100ms")

Both use nanoseconds internally to avoid floating-point precision issues
that can cause non-deterministic behavior when comparing event times.

Special values:
- Instant.Epoch: Time zero (start of simulation)
- Instant.Infinity: Represents unbounded time (for auto-termination)
- Duration.ZERO: Zero-length duration
"""

from __future__ import annotations

from typing import Union


class Duration:
    """Immutable duration value with nanosecond precision.

    Represents a length of time, not a point in time. Use for latencies,
    timeouts, intervals, and other measurements of elapsed time.

    Stores time as an integer number of nanoseconds to avoid floating-point
    errors. Supports arithmetic with other Durations or float seconds.

    Attributes:
        nanoseconds: The duration in nanoseconds.
    """

    def __init__(self, nanoseconds: int):
        """Create a Duration from nanoseconds.

        Args:
            nanoseconds: Duration in nanoseconds.
        """
        self.nanoseconds = nanoseconds

    @classmethod
    def from_seconds(cls, seconds: int | float) -> Duration:
        """Create a Duration from a seconds value.

        Args:
            seconds: Duration in seconds (int or float).

        Returns:
            New Duration representing the given length of time.

        Raises:
            TypeError: If seconds is not int or float.
        """
        if isinstance(seconds, int):
            return cls(seconds * 1_000_000_000)

        if isinstance(seconds, float):
            return cls(int(seconds * 1_000_000_000))

        raise TypeError("seconds must be int or float")

    def to_seconds(self) -> float:
        """Convert this Duration to seconds as a float."""
        return float(self.nanoseconds) / 1_000_000_000

    def __add__(self, other: Union[Duration, int, float]) -> Duration:
        """Add two durations or a duration and seconds."""
        if isinstance(other, Duration):
            return Duration(self.nanoseconds + other.nanoseconds)
        if isinstance(other, (int, float)):
            return Duration(self.nanoseconds + int(other * 1_000_000_000))
        return NotImplemented

    def __radd__(self, other: Union[int, float]) -> Duration:
        """Support seconds + Duration."""
        if isinstance(other, (int, float)):
            return Duration(int(other * 1_000_000_000) + self.nanoseconds)
        return NotImplemented

    def __sub__(self, other: Union[Duration, int, float]) -> Duration:
        """Subtract durations or seconds from a duration."""
        if isinstance(other, Duration):
            return Duration(self.nanoseconds - other.nanoseconds)
        if isinstance(other, (int, float)):
            return Duration(self.nanoseconds - int(other * 1_000_000_000))
        return NotImplemented

    def __mul__(self, other: int | float) -> Duration:
        """Multiply duration by a scalar."""
        if isinstance(other, (int, float)):
            return Duration(int(self.nanoseconds * other))
        return NotImplemented

    def __rmul__(self, other: int | float) -> Duration:
        """Support scalar * Duration."""
        return self.__mul__(other)

    def __truediv__(self, other: int | float) -> Duration:
        """Divide duration by a scalar."""
        if isinstance(other, (int, float)):
            return Duration(int(self.nanoseconds / other))
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Duration):
            return NotImplemented
        return self.nanoseconds == other.nanoseconds

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __lt__(self, other: Duration) -> bool:
        if not isinstance(other, Duration):
            return NotImplemented
        return self.nanoseconds < other.nanoseconds

    def __le__(self, other: Duration) -> bool:
        if not isinstance(other, Duration):
            return NotImplemented
        return self.nanoseconds <= other.nanoseconds

    def __gt__(self, other: Duration) -> bool:
        if not isinstance(other, Duration):
            return NotImplemented
        return self.nanoseconds > other.nanoseconds

    def __ge__(self, other: Duration) -> bool:
        if not isinstance(other, Duration):
            return NotImplemented
        return self.nanoseconds >= other.nanoseconds

    def __hash__(self) -> int:
        return hash(self.nanoseconds)

    def __repr__(self) -> str:
        """Return a human-readable duration string.

        Format: D{hours}:{minutes}:{seconds}.{microseconds}
        Examples: D00:00:01.500000, D01:23:45.678901
        """
        total_us = abs(self.nanoseconds) // 1_000
        sign = "-" if self.nanoseconds < 0 else ""

        us = total_us % 1_000_000
        total_seconds = total_us // 1_000_000
        seconds = total_seconds % 60
        total_minutes = total_seconds // 60
        minutes = total_minutes % 60
        hours = total_minutes // 60

        return f"{sign}D{hours:02d}:{minutes:02d}:{seconds:02d}.{us:06d}"


# Singleton for zero duration
Duration.ZERO = Duration(0)


class Instant:
    """Immutable time point with nanosecond precision.

    Represents a specific point in time, not a duration. Use for event
    timestamps, scheduling times, and absolute time references.

    Stores time as an integer number of nanoseconds to avoid floating-point
    errors. Supports arithmetic with Durations or float seconds.

    Attributes:
        nanoseconds: The time value in nanoseconds since epoch.
    """

    def __init__(self, nanoseconds: int):
        """Create an Instant from nanoseconds since epoch.

        Args:
            nanoseconds: Time in nanoseconds since simulation start.
        """
        self.nanoseconds = nanoseconds

    @classmethod
    def from_seconds(cls, seconds: int | float) -> Instant:
        """Create an Instant from a seconds value.

        Args:
            seconds: Time in seconds (int or float).

        Returns:
            New Instant representing the given point in time.

        Raises:
            TypeError: If seconds is not int or float.
        """
        if isinstance(seconds, int):
            return cls(seconds * 1_000_000_000)

        if isinstance(seconds, float):
            return cls(int(seconds * 1_000_000_000))

        raise TypeError("seconds must be int or float")

    def to_seconds(self) -> float:
        """Convert this Instant to seconds as a float."""
        return float(self.nanoseconds) / 1_000_000_000

    def __add__(self, other: Union[Duration, int, float]) -> Instant:
        """Add a duration to an instant, producing a new instant.

        Instant + Duration = Instant (a point in time shifted by a duration)
        Instant + float = Instant (float interpreted as seconds)
        """
        if isinstance(other, Duration):
            return Instant(self.nanoseconds + other.nanoseconds)
        if isinstance(other, (int, float)):
            return Instant(self.nanoseconds + int(other * 1_000_000_000))
        return NotImplemented

    def __radd__(self, other: Union[int, float]) -> Instant:
        """Support seconds + Instant."""
        if isinstance(other, (int, float)):
            return Instant(int(other * 1_000_000_000) + self.nanoseconds)
        return NotImplemented

    def __sub__(self, other: Union[Instant, Duration, int, float]) -> Union[Instant, Duration]:
        """Subtract from an instant.

        Instant - Instant = Duration (the time between two points)
        Instant - Duration = Instant (a point in time shifted backwards)
        Instant - float = Instant (float interpreted as seconds)
        """
        if isinstance(other, Instant):
            return Duration(self.nanoseconds - other.nanoseconds)
        if isinstance(other, Duration):
            return Instant(self.nanoseconds - other.nanoseconds)
        if isinstance(other, (int, float)):
            return Instant(self.nanoseconds - int(other * 1_000_000_000))
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Instant):
            return NotImplemented
        return self.nanoseconds == other.nanoseconds

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __lt__(self, other: Instant) -> bool:
        if not isinstance(other, Instant):
            return NotImplemented
        return self.nanoseconds < other.nanoseconds

    def __le__(self, other: Instant) -> bool:
        if not isinstance(other, Instant):
            return NotImplemented
        return self.nanoseconds <= other.nanoseconds

    def __gt__(self, other: Instant) -> bool:
        if not isinstance(other, Instant):
            return NotImplemented
        return self.nanoseconds > other.nanoseconds

    def __ge__(self, other: Instant) -> bool:
        if not isinstance(other, Instant):
            return NotImplemented
        return self.nanoseconds >= other.nanoseconds

    def __hash__(self) -> int:
        return hash(self.nanoseconds)

    def __repr__(self) -> str:
        """Return a human-readable ISO-like time string with microsecond precision.

        Format: T{hours}:{minutes}:{seconds}.{microseconds}
        Examples: T00:00:01.500000, T01:23:45.678901
        """
        total_us = self.nanoseconds // 1_000

        us = total_us % 1_000_000
        total_seconds = total_us // 1_000_000
        seconds = total_seconds % 60
        total_minutes = total_seconds // 60
        minutes = total_minutes % 60
        hours = total_minutes // 60

        return f"T{hours:02d}:{minutes:02d}:{seconds:02d}.{us:06d}"


class _InfiniteInstant(Instant):
    """Singleton representing positive infinity for time comparisons.

    Used as the default end_time for auto-terminating simulations.
    Greater than all finite Instants. Arithmetic with infinity yields
    infinity (absorbing).
    """

    def __init__(self):
        import sys
        # Use maxsize as the nanoseconds value. All comparisons and arithmetic
        # are overridden so this value is never used directly, but it satisfies
        # the int type requirement of Instant.__init__.
        super().__init__(sys.maxsize)

    def __add__(self, other: Union[Duration, int, float]) -> Instant:
        if isinstance(other, (int, float, Duration)):
            return self
        return NotImplemented

    def __sub__(self, other: Union[Instant, Duration, int, float]) -> Union[Instant, Duration]:
        if isinstance(other, _InfiniteInstant):
            return NotImplemented
        if isinstance(other, (int, float, Duration)):
            return self
        if isinstance(other, Instant):
            # Infinity - finite instant = infinite duration (not really meaningful)
            return self
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Instant):
            return NotImplemented
        return isinstance(other, _InfiniteInstant)

    def __hash__(self) -> int:
        return hash(float('inf'))

    def __lt__(self, other: Instant) -> bool:
        if not isinstance(other, Instant):
            return NotImplemented
        return False

    def __le__(self, other: Instant) -> bool:
        if not isinstance(other, Instant):
            return NotImplemented
        return isinstance(other, _InfiniteInstant)

    def __gt__(self, other: Instant) -> bool:
        if not isinstance(other, Instant):
            return NotImplemented
        return not isinstance(other, _InfiniteInstant)

    def __ge__(self, other: Instant) -> bool:
        if not isinstance(other, Instant):
            return NotImplemented
        return True

    def to_seconds(self) -> float:
        return float('inf')

    def __repr__(self) -> str:
        return "Instant.Infinity"


# Singleton instances
Instant.Infinity = _InfiniteInstant()
Instant.Epoch = Instant(0)
