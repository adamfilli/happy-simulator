"""Bloom Filter for probabilistic set membership testing.

A Bloom Filter is a space-efficient probabilistic data structure for testing
whether an element is a member of a set. It can have false positives (saying
an element is present when it isn't) but never false negatives (if it says
an element is not present, it definitely isn't).

Key properties:
- Space: O(n) bits where n is configurable
- Insert: O(k) where k is number of hash functions
- Query: O(k)
- False positive rate: (1 - e^(-kn/m))^k approximately
- No false negatives

This is ideal for:
- Cache invalidation (is this key possibly in cache?)
- Duplicate detection (have we possibly seen this before?)
- Spell checking (is this word possibly in dictionary?)
- Any case where false positives are acceptable but false negatives aren't

Reference:
    Bloom. "Space/Time Trade-offs in Hash Coding with Allowable Errors" (1970)
"""

from __future__ import annotations

import hashlib
import math
import struct
import sys
from typing import Generic, TypeVar, Hashable

from happysimulator.sketching.base import MembershipSketch

T = TypeVar('T', bound=Hashable)


def _optimal_num_hashes(size_bits: int, expected_items: int) -> int:
    """Calculate optimal number of hash functions.

    k_opt = (m/n) * ln(2)
    """
    if expected_items == 0:
        return 1
    return max(1, int(round((size_bits / expected_items) * math.log(2))))


def _optimal_size_bits(expected_items: int, fp_rate: float) -> int:
    """Calculate optimal bit array size for target false positive rate.

    m = -n * ln(p) / (ln(2))^2
    """
    if expected_items == 0:
        return 64
    return int(math.ceil(-expected_items * math.log(fp_rate) / (math.log(2) ** 2)))


class BloomFilter(MembershipSketch[T]):
    """Bloom Filter for probabilistic set membership.

    Space-efficient data structure for testing set membership. Can report
    false positives but never false negatives.

    Args:
        size_bits: Size of the bit array. Larger = lower false positive rate.
        num_hashes: Number of hash functions. If None, calculated optimally.
        seed: Random seed for hash function initialization.

    Alternatively, create from expected usage:
        BloomFilter.from_expected_items(n=10000, fp_rate=0.01)

    Example:
        # Check if URLs have been visited
        visited = BloomFilter.from_expected_items(n=1_000_000, fp_rate=0.01)

        for url in crawl_history:
            visited.add(url)

        if visited.contains(new_url):
            print("Probably visited before")
        else:
            print("Definitely not visited")
    """

    def __init__(
        self,
        size_bits: int,
        num_hashes: int | None = None,
        seed: int | None = None,
    ):
        """Initialize Bloom Filter.

        Args:
            size_bits: Size of the bit array. Must be > 0.
            num_hashes: Number of hash functions. If None, defaults to 7.
            seed: Random seed for reproducibility.

        Raises:
            ValueError: If size_bits <= 0 or num_hashes <= 0.
        """
        if size_bits <= 0:
            raise ValueError(f"size_bits must be positive, got {size_bits}")
        if num_hashes is not None and num_hashes <= 0:
            raise ValueError(f"num_hashes must be positive, got {num_hashes}")

        self._size_bits = size_bits
        self._num_hashes = num_hashes if num_hashes is not None else 7
        self._seed = seed if seed is not None else 0

        # Bit array stored as list of integers (64 bits each)
        num_words = (size_bits + 63) // 64
        self._bits: list[int] = [0] * num_words
        self._bits_set = 0  # Count of 1 bits
        self._total_count = 0  # Total adds

    @classmethod
    def from_expected_items(
        cls,
        n: int,
        fp_rate: float,
        seed: int | None = None,
    ) -> "BloomFilter[T]":
        """Create a filter sized for expected items and false positive rate.

        Args:
            n: Expected number of items to insert.
            fp_rate: Desired false positive rate (e.g., 0.01 for 1%).
            seed: Random seed for reproducibility.

        Returns:
            BloomFilter configured for the specified parameters.

        Raises:
            ValueError: If n < 0 or fp_rate not in (0, 1).
        """
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        if not 0 < fp_rate < 1:
            raise ValueError(f"fp_rate must be in (0, 1), got {fp_rate}")

        size_bits = _optimal_size_bits(n, fp_rate)
        num_hashes = _optimal_num_hashes(size_bits, n)
        return cls(size_bits=size_bits, num_hashes=num_hashes, seed=seed)

    @property
    def size_bits(self) -> int:
        """Size of the bit array in bits."""
        return self._size_bits

    @property
    def num_hashes(self) -> int:
        """Number of hash functions used."""
        return self._num_hashes

    def _hash(self, item: T, i: int) -> int:
        """Hash an item to a bit index.

        Uses double hashing: h(item, i) = (h1 + i * h2) mod m
        """
        h = hashlib.sha256()
        h.update(struct.pack(">QQ", self._seed, i))
        h.update(repr(item).encode('utf-8'))
        digest = h.digest()
        h1 = struct.unpack(">Q", digest[:8])[0]
        h2 = struct.unpack(">Q", digest[8:16])[0]
        return (h1 + i * h2) % self._size_bits

    def _set_bit(self, bit_idx: int) -> bool:
        """Set a bit and return True if it was previously 0."""
        word_idx = bit_idx // 64
        bit_pos = bit_idx % 64
        mask = 1 << bit_pos

        was_zero = (self._bits[word_idx] & mask) == 0
        self._bits[word_idx] |= mask
        return was_zero

    def _get_bit(self, bit_idx: int) -> bool:
        """Check if a bit is set."""
        word_idx = bit_idx // 64
        bit_pos = bit_idx % 64
        return bool(self._bits[word_idx] & (1 << bit_pos))

    def add(self, item: T, count: int = 1) -> None:
        """Add an item to the filter.

        Note: count > 1 has the same effect as count = 1 for membership.

        Args:
            item: The item to add. Must be hashable.
            count: Number of times to add (only presence matters).

        Raises:
            ValueError: If count is negative.
        """
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        if count == 0:
            return

        self._total_count += count

        for i in range(self._num_hashes):
            bit_idx = self._hash(item, i)
            if self._set_bit(bit_idx):
                self._bits_set += 1

    def contains(self, item: T) -> bool:
        """Test if an item is (probably) in the set.

        Args:
            item: The item to test.

        Returns:
            True if the item might be in the set (possible false positive).
            False if the item is definitely not in the set.
        """
        for i in range(self._num_hashes):
            bit_idx = self._hash(item, i)
            if not self._get_bit(bit_idx):
                return False
        return True

    def __contains__(self, item: T) -> bool:
        """Support 'in' operator."""
        return self.contains(item)

    @property
    def false_positive_rate(self) -> float:
        """Estimated current false positive rate.

        Returns:
            Probability that contains() returns True for an item not in the set.
        """
        if self._bits_set == 0:
            return 0.0

        # Estimated fill ratio
        fill_ratio = self._bits_set / self._size_bits

        # False positive probability: fill_ratio^num_hashes
        return fill_ratio ** self._num_hashes

    @property
    def fill_ratio(self) -> float:
        """Proportion of bits that are set."""
        return self._bits_set / self._size_bits

    def merge(self, other: "BloomFilter[T]") -> None:
        """Merge another Bloom Filter into this one (OR operation).

        After merging, this filter represents the union of both sets.

        Args:
            other: Another BloomFilter with same size and hash count.

        Raises:
            TypeError: If other is not a BloomFilter.
            ValueError: If other has different configuration.
        """
        if not isinstance(other, BloomFilter):
            raise TypeError(f"Can only merge with BloomFilter, got {type(other).__name__}")
        if other._size_bits != self._size_bits:
            raise ValueError(
                f"Cannot merge: size_bits differs ({self._size_bits} vs {other._size_bits})"
            )
        if other._num_hashes != self._num_hashes:
            raise ValueError(
                f"Cannot merge: num_hashes differs ({self._num_hashes} vs {other._num_hashes})"
            )
        if other._seed != self._seed:
            raise ValueError(f"Cannot merge: seeds differ ({self._seed} vs {other._seed})")

        # OR the bit arrays
        for i in range(len(self._bits)):
            self._bits[i] |= other._bits[i]

        # Recount bits (could track incrementally but this is simpler)
        self._bits_set = sum(bin(word).count('1') for word in self._bits)
        self._total_count += other._total_count

    @property
    def memory_bytes(self) -> int:
        """Estimated memory usage in bytes."""
        # Bit array: size_bits / 8 bytes (stored as 64-bit words)
        bit_bytes = len(self._bits) * 8
        return bit_bytes + sys.getsizeof(self)

    @property
    def item_count(self) -> int:
        """Total count of items added."""
        return self._total_count

    def clear(self) -> None:
        """Reset the filter to empty state."""
        self._bits = [0] * len(self._bits)
        self._bits_set = 0
        self._total_count = 0

    def __repr__(self) -> str:
        return (
            f"BloomFilter(size_bits={self._size_bits}, "
            f"num_hashes={self._num_hashes}, "
            f"fill={self.fill_ratio:.1%}, "
            f"fp_rate≈{self.false_positive_rate:.3%})"
        )
