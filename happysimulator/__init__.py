"""happy-simulator: A discrete-event simulation library for Python."""

__version__ = "0.1.3"

import logging

# Library best practice: silent by default
# Users must explicitly enable logging using the configure functions below
logging.getLogger("happysimulator").addHandler(logging.NullHandler())

# Core simulation types
from happysimulator.core import (
    Clock,
    Entity,
    Event,
    Instant,
    Simulation,
    Simulatable,
    simulatable,
)
from happysimulator.core.temporal import Duration

# Load generation
from happysimulator.load import (
    ConstantArrivalTimeProvider,
    ConstantRateProfile,
    DistributedFieldProvider,
    EventProvider,
    LinearRampProfile,
    PoissonArrivalTimeProvider,
    Profile,
    Source,
    SpikeProfile,
)

# Components
from happysimulator.components import (
    FIFOQueue,
    LIFOQueue,
    PriorityQueue,
    Queue,
    QueueDriver,
    QueuedResource,
    RandomRouter,
    RateLimiterPolicy,
    RateLimitedEntity,
    RateLimitedEntityStats,
    TokenBucketPolicy,
    LeakyBucketPolicy,
    SlidingWindowPolicy,
    FixedWindowPolicy,
    AdaptivePolicy,
    RateAdjustmentReason,
    RateSnapshot,
)

# Distributions
from happysimulator.distributions import (
    ConstantLatency,
    ExponentialLatency,
    PercentileFittedLatency,
    UniformDistribution,
    ValueDistribution,
    ZipfDistribution,
)

# Instrumentation
from happysimulator.instrumentation import (
    Data,
    Probe,
)

# Sketching algorithms
from happysimulator.sketching import (
    TopK,
    CountMinSketch,
    TDigest,
    HyperLogLog,
    BloomFilter,
    ReservoirSampler,
    FrequencyEstimate,
)

# Sketching entity wrappers
from happysimulator.components.sketching import (
    SketchCollector,
    TopKCollector,
    QuantileEstimator,
)

# Logging configuration utilities
from happysimulator.logging_config import (
    configure_from_env,
    disable_logging,
    enable_console_logging,
    enable_file_logging,
    enable_json_file_logging,
    enable_json_logging,
    enable_timed_file_logging,
    set_level,
    set_module_level,
)

__all__ = [
    # Package metadata
    "__version__",
    # Core
    "Simulation",
    "Event",
    "Entity",
    "Instant",
    "Duration",
    "Clock",
    "Simulatable",
    "simulatable",
    # Load
    "Source",
    "EventProvider",
    "Profile",
    "ConstantRateProfile",
    "LinearRampProfile",
    "SpikeProfile",
    "ConstantArrivalTimeProvider",
    "PoissonArrivalTimeProvider",
    # Components
    "Queue",
    "QueueDriver",
    "QueuedResource",
    "FIFOQueue",
    "LIFOQueue",
    "PriorityQueue",
    "RandomRouter",
    # Rate limiters
    "RateLimiterPolicy",
    "RateLimitedEntity",
    "RateLimitedEntityStats",
    "TokenBucketPolicy",
    "LeakyBucketPolicy",
    "SlidingWindowPolicy",
    "FixedWindowPolicy",
    "AdaptivePolicy",
    "RateAdjustmentReason",
    "RateSnapshot",
    # Distributions
    "ConstantLatency",
    "ExponentialLatency",
    "PercentileFittedLatency",
    "ZipfDistribution",
    "UniformDistribution",
    "ValueDistribution",
    # Load providers
    "DistributedFieldProvider",
    # Instrumentation
    "Data",
    "Probe",
    # Sketching algorithms
    "TopK",
    "CountMinSketch",
    "TDigest",
    "HyperLogLog",
    "BloomFilter",
    "ReservoirSampler",
    "FrequencyEstimate",
    # Sketching entity wrappers
    "SketchCollector",
    "TopKCollector",
    "QuantileEstimator",
    # Logging configuration
    "configure_from_env",
    "disable_logging",
    "enable_console_logging",
    "enable_file_logging",
    "enable_json_file_logging",
    "enable_json_logging",
    "enable_timed_file_logging",
    "set_level",
    "set_module_level",
]
