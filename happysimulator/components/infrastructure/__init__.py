"""Low-level infrastructure primitives for distributed system simulation.

Provides components modeling OS and hardware-level behaviors that affect
distributed system performance: disk I/O, page caching, CPU scheduling,
garbage collection, TCP transport, and DNS resolution.
"""

from happysimulator.components.infrastructure.disk_io import (
    DiskIO,
    DiskIOStats,
    DiskProfile,
    HDD,
    SSD,
    NVMe,
)
from happysimulator.components.infrastructure.page_cache import (
    PageCache,
    PageCacheStats,
)
from happysimulator.components.infrastructure.cpu_scheduler import (
    CPUScheduler,
    CPUSchedulerStats,
    CPUTask,
    SchedulingPolicy,
    FairShare,
    PriorityPreemptive,
)
from happysimulator.components.infrastructure.garbage_collector import (
    GarbageCollector,
    GCStats,
    GCStrategy,
    StopTheWorld,
    ConcurrentGC,
    GenerationalGC,
)
from happysimulator.components.infrastructure.tcp_connection import (
    TCPConnection,
    TCPStats,
    CongestionControl,
    AIMD,
    Cubic,
    BBR,
)
from happysimulator.components.infrastructure.dns_resolver import (
    DNSResolver,
    DNSRecord,
    DNSStats,
)

__all__ = [
    # DiskIO
    "DiskIO",
    "DiskIOStats",
    "DiskProfile",
    "HDD",
    "SSD",
    "NVMe",
    # PageCache
    "PageCache",
    "PageCacheStats",
    # CPUScheduler
    "CPUScheduler",
    "CPUSchedulerStats",
    "CPUTask",
    "SchedulingPolicy",
    "FairShare",
    "PriorityPreemptive",
    # GarbageCollector
    "GarbageCollector",
    "GCStats",
    "GCStrategy",
    "StopTheWorld",
    "ConcurrentGC",
    "GenerationalGC",
    # TCPConnection
    "TCPConnection",
    "TCPStats",
    "CongestionControl",
    "AIMD",
    "Cubic",
    "BBR",
    # DNSResolver
    "DNSResolver",
    "DNSRecord",
    "DNSStats",
]
