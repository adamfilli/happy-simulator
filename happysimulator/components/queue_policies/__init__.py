"""Advanced queue policies for sophisticated queue management.

This module provides active queue management (AQM) algorithms and
specialized queuing strategies beyond basic FIFO/LIFO.

Policies:
    CoDelQueue: Controlled Delay - manages queue delay, not size
    REDQueue: Random Early Detection - probabilistic early dropping
    FairQueue: Per-flow fair queuing with round-robin
    WeightedFairQueue: Weighted fair queuing with priority classes
    DeadlineQueue: Priority by deadline with expiration
    AdaptiveLIFO: LIFO under congestion, FIFO otherwise

Example:
    from happysimulator.components.queue_policies import (
        CoDelQueue,
        REDQueue,
        FairQueue,
        DeadlineQueue,
    )

    # CoDel for controlling latency
    codel = CoDelQueue(target_delay=0.005, interval=0.100)

    # RED for congestion control
    red = REDQueue(min_threshold=10, max_threshold=30)

    # Fair queuing for multi-tenant systems
    fair = FairQueue(get_flow_id=lambda e: e.context["tenant_id"])
"""

from happysimulator.components.queue_policies.adaptive_lifo import (
    AdaptiveLIFO,
    AdaptiveLIFOStats,
)
from happysimulator.components.queue_policies.codel import (
    CoDelQueue,
    CoDelStats,
)
from happysimulator.components.queue_policies.deadline_queue import (
    DeadlineQueue,
    DeadlineQueueStats,
)
from happysimulator.components.queue_policies.fair_queue import (
    FairQueue,
    FairQueueStats,
)
from happysimulator.components.queue_policies.red import (
    REDQueue,
    REDStats,
)
from happysimulator.components.queue_policies.weighted_fair_queue import (
    WeightedFairQueue,
    WeightedFairQueueStats,
)

__all__ = [
    # Adaptive LIFO
    "AdaptiveLIFO",
    "AdaptiveLIFOStats",
    # CoDel
    "CoDelQueue",
    "CoDelStats",
    # Deadline Queue
    "DeadlineQueue",
    "DeadlineQueueStats",
    # Fair Queue
    "FairQueue",
    "FairQueueStats",
    # RED
    "REDQueue",
    "REDStats",
    # Weighted Fair Queue
    "WeightedFairQueue",
    "WeightedFairQueueStats",
]
