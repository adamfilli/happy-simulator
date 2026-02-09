"""happy-simulator: A discrete-event simulation library for Python."""

__version__ = "0.1.3"

import logging

# Library best practice: silent by default
# Users must explicitly enable logging using the configure functions below
logging.getLogger("happysimulator").addHandler(logging.NullHandler())

# Core simulation types
from happysimulator.core import (
    CallbackEntity,
    Clock,
    Entity,
    Event,
    Instant,
    NullEntity,
    SimFuture,
    any_of,
    all_of,
    Simulation,
    Simulatable,
    simulatable,
    SimulationControl,
    SimulationState,
    BreakpointContext,
    TimeBreakpoint,
    EventCountBreakpoint,
    ConditionBreakpoint,
    MetricBreakpoint,
    EventTypeBreakpoint,
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
    Counter,
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
    Sink,
    TokenBucketPolicy,
    LeakyBucketPolicy,
    SlidingWindowPolicy,
    FixedWindowPolicy,
    AdaptivePolicy,
    RateAdjustmentReason,
    RateSnapshot,
    Inductor,
    InductorStats,
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
    BucketedData,
    Data,
    EntitySummary,
    LatencyTracker,
    Probe,
    QueueStats,
    SimulationSummary,
    ThroughputTracker,
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
    "CallbackEntity",
    "NullEntity",
    "Instant",
    "Duration",
    "Clock",
    "Simulatable",
    "simulatable",
    # SimFuture
    "SimFuture",
    "any_of",
    "all_of",
    # Simulation control
    "SimulationControl",
    "SimulationState",
    "BreakpointContext",
    "TimeBreakpoint",
    "EventCountBreakpoint",
    "ConditionBreakpoint",
    "MetricBreakpoint",
    "EventTypeBreakpoint",
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
    "Counter",
    "Queue",
    "QueueDriver",
    "QueuedResource",
    "FIFOQueue",
    "LIFOQueue",
    "PriorityQueue",
    "RandomRouter",
    "Sink",
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
    "Inductor",
    "InductorStats",
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
    "BucketedData",
    "Data",
    "EntitySummary",
    "LatencyTracker",
    "Probe",
    "QueueStats",
    "SimulationSummary",
    "ThroughputTracker",
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
