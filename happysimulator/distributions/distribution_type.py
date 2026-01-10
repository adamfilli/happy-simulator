"""Enumeration of supported distribution types.

Used to configure arrival patterns and latency distributions without
directly referencing implementation classes.
"""

from enum import Enum, auto


class DistributionType(Enum):
    """Distribution type identifiers.

    POISSON: Exponential inter-arrival times (memoryless, random).
    CONSTANT: Deterministic inter-arrival times (perfectly regular).
    """
    POISSON = auto()
    CONSTANT = auto()