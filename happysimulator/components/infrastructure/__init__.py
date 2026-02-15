"""Low-level infrastructure primitives for distributed system simulation.

Provides components modeling OS and hardware-level behaviors that affect
distributed system performance: disk I/O, page caching, CPU scheduling,
garbage collection, TCP transport, and DNS resolution.
"""

from happysimulator.components.infrastructure.cpu_scheduler import (
    CPUScheduler,
    CPUSchedulerStats,
    CPUTask,
    FairShare,
    PriorityPreemptive,
    SchedulingPolicy,
)
from happysimulator.components.infrastructure.disk_io import (
    HDD,
    SSD,
    DiskIO,
    DiskIOStats,
    DiskProfile,
    NVMe,
)
from happysimulator.components.infrastructure.dns_resolver import (
    DNSRecord,
    DNSResolver,
    DNSStats,
)
from happysimulator.components.infrastructure.garbage_collector import (
    ConcurrentGC,
    GarbageCollector,
    GCStats,
    GCStrategy,
    GenerationalGC,
    StopTheWorld,
)
from happysimulator.components.infrastructure.page_cache import (
    PageCache,
    PageCacheStats,
)
from happysimulator.components.infrastructure.tcp_connection import (
    AIMD,
    BBR,
    CongestionControl,
    Cubic,
    TCPConnection,
    TCPStats,
)

__all__ = [
    "AIMD",
    "BBR",
    "HDD",
    "SSD",
    # CPUScheduler
    "CPUScheduler",
    "CPUSchedulerStats",
    "CPUTask",
    "ConcurrentGC",
    "CongestionControl",
    "Cubic",
    "DNSRecord",
    # DNSResolver
    "DNSResolver",
    "DNSStats",
    # DiskIO
    "DiskIO",
    "DiskIOStats",
    "DiskProfile",
    "FairShare",
    "GCStats",
    "GCStrategy",
    # GarbageCollector
    "GarbageCollector",
    "GenerationalGC",
    "NVMe",
    # PageCache
    "PageCache",
    "PageCacheStats",
    "PriorityPreemptive",
    "SchedulingPolicy",
    "StopTheWorld",
    # TCPConnection
    "TCPConnection",
    "TCPStats",
]
