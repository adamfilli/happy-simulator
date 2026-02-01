"""HyperLogLog for cardinality (distinct count) estimation.

HyperLogLog is a probabilistic data structure for estimating the number of
distinct elements (cardinality) in a data stream. It uses a small, fixed
amount of memory regardless of the stream size.

Key properties:
- Space: O(2^precision) bytes (typically 1-16 KB)
- Update: O(1)
- Query: O(m) where m = 2^precision
- Error: ~1.04/√m standard error

This is ideal for:
- Counting unique visitors/users
- Distinct query counting
- Any cardinality estimation where exact count isn't needed

Reference:
    Flajolet, Fusy, Gandouet, Meunier. "HyperLogLog: the analysis of a
    near-optimal cardinality estimation algorithm" (2007)
"""

from __future__ import annotations

import hashlib
import math
import struct
import sys
from typing import Generic, TypeVar, Hashable

from happysimulator.sketching.base import CardinalitySketch

T = TypeVar('T', bound=Hashable)


def _count_leading_zeros(value: int, max_bits: int = 64) -> int:
    """Count leading zeros in a binary number.

    Args:
        value: The integer value.
        max_bits: Maximum number of bits to consider.

    Returns:
        Number of leading zeros.
    """
    if value == 0:
        return max_bits

    count = 0
    for i in range(max_bits - 1, -1, -1):
        if value & (1 << i):
            break
        count += 1
    return count


class HyperLogLog(CardinalitySketch, Generic[T]):
    """HyperLogLog for streaming cardinality estimation.

    Estimates the number of distinct elements in a stream using
    fixed memory. Uses multiple registers that track the maximum
    number of leading zeros in hashed values.

    Args:
        precision: Number of bits for register index (4-16).
            - precision=4: 16 registers, ~26% error
            - precision=10: 1024 registers, ~3.2% error
            - precision=14: 16384 registers, ~0.8% error
            Default is 14 for good accuracy.
        seed: Random seed for hash function initialization.

    Example:
        # Count unique visitors
        hll = HyperLogLog[str](precision=14)

        for visitor_id in visitor_stream:
            hll.add(visitor_id)

        unique_count = hll.cardinality()
        print(f"~{unique_count} unique visitors")
    """

    # Bias correction constants from the paper
    _ALPHA = {
        4: 0.673,
        5: 0.697,
        6: 0.709,
    }

    def __init__(self, precision: int = 14, seed: int | None = None):
        """Initialize HyperLogLog.

        Args:
            precision: Bits for register index (4-16). Default 14.
            seed: Random seed for reproducibility.

        Raises:
            ValueError: If precision not in [4, 16].
        """
        if not 4 <= precision <= 16:
            raise ValueError(f"precision must be in [4, 16], got {precision}")

        self._precision = precision
        self._num_registers = 1 << precision  # 2^precision
        self._registers = [0] * self._num_registers
        self._seed = seed if seed is not None else 0
        self._total_count = 0

    @property
    def precision(self) -> int:
        """Number of bits used for register indexing."""
        return self._precision

    @property
    def num_registers(self) -> int:
        """Number of registers (2^precision)."""
        return self._num_registers

    def _alpha(self) -> float:
        """Get bias correction factor alpha_m."""
        if self._precision in self._ALPHA:
            return self._ALPHA[self._precision]
        # For m >= 128: alpha_m = 0.7213 / (1 + 1.079/m)
        m = self._num_registers
        return 0.7213 / (1 + 1.079 / m)

    def _hash(self, item: T) -> int:
        """Hash an item to a 64-bit integer."""
        h = hashlib.sha256()
        # Include seed for reproducibility
        h.update(struct.pack(">Q", self._seed))
        # Hash the item (convert to bytes via repr for general hashables)
        h.update(repr(item).encode('utf-8'))
        return struct.unpack(">Q", h.digest()[:8])[0]

    def add(self, item: T, count: int = 1) -> None:
        """Add an item to the sketch.

        Note: count parameter is accepted but only the presence of the item
        matters for cardinality estimation, not how many times it's added.

        Args:
            item: The item to add. Must be hashable.
            count: Ignored for cardinality (only presence matters).
        """
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        if count == 0:
            return

        self._total_count += count

        # Hash the item
        hash_value = self._hash(item)

        # First p bits determine the register index
        register_idx = hash_value >> (64 - self._precision)

        # Remaining bits are used for the "run" (leading zeros + 1)
        remaining_bits = hash_value & ((1 << (64 - self._precision)) - 1)
        # Count leading zeros in the remaining bits, plus 1
        run_length = _count_leading_zeros(remaining_bits, 64 - self._precision) + 1

        # Update register with maximum run length seen
        self._registers[register_idx] = max(self._registers[register_idx], run_length)

    def cardinality(self) -> int:
        """Estimate the number of distinct items.

        Returns:
            Estimated count of unique items.
        """
        m = self._num_registers
        alpha = self._alpha()

        # Harmonic mean of 2^(-register)
        indicator = sum(2.0 ** (-r) for r in self._registers)
        raw_estimate = alpha * m * m / indicator

        # Apply corrections for small and large cardinalities
        if raw_estimate <= 2.5 * m:
            # Small range correction
            # Count registers with value 0
            zeros = self._registers.count(0)
            if zeros > 0:
                # Linear counting estimate
                raw_estimate = m * math.log(m / zeros)
        elif raw_estimate > (1 << 32) / 30:
            # Large range correction (for very large cardinalities)
            raw_estimate = -(1 << 32) * math.log(1 - raw_estimate / (1 << 32))

        return int(raw_estimate)

    def standard_error(self) -> float:
        """Theoretical standard error of the estimate.

        Returns:
            Expected relative error (e.g., 0.01 for 1% error).
        """
        return 1.04 / math.sqrt(self._num_registers)

    def merge(self, other: "HyperLogLog[T]") -> None:
        """Merge another HyperLogLog into this one.

        After merging, this sketch estimates the cardinality of the union
        of both streams.

        Args:
            other: Another HyperLogLog with same precision.

        Raises:
            TypeError: If other is not a HyperLogLog.
            ValueError: If other has different precision.
        """
        if not isinstance(other, HyperLogLog):
            raise TypeError(f"Can only merge with HyperLogLog, got {type(other).__name__}")
        if other._precision != self._precision:
            raise ValueError(
                f"Cannot merge: precision differs ({self._precision} vs {other._precision})"
            )

        # Take maximum of each register
        for i in range(self._num_registers):
            self._registers[i] = max(self._registers[i], other._registers[i])
        self._total_count += other._total_count

    @property
    def memory_bytes(self) -> int:
        """Estimated memory usage in bytes."""
        # Each register is an int (but typically small, <64)
        # In practice, could use bytes for registers
        return self._num_registers * 8 + sys.getsizeof(self)

    @property
    def item_count(self) -> int:
        """Total count of items added (not distinct count)."""
        return self._total_count

    def clear(self) -> None:
        """Reset the sketch to empty state."""
        self._registers = [0] * self._num_registers
        self._total_count = 0

    def __repr__(self) -> str:
        return (
            f"HyperLogLog(precision={self._precision}, "
            f"registers={self._num_registers}, "
            f"cardinality≈{self.cardinality()})"
        )
