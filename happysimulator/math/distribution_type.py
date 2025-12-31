from enum import Enum, auto

class DistributionType(Enum):
    POISSON = auto()   # Exponential inter-arrival times (Memoryless, Random)
    CONSTANT = auto()  # Deterministic inter-arrival times (Perfectly smooth)