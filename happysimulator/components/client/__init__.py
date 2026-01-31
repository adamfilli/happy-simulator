"""Client components for request handling simulation.

This package provides client abstractions with timeout handling,
retry policies, connection pooling, and response tracking.
"""

from happysimulator.components.client.client import Client, ClientStats
from happysimulator.components.client.connection_pool import (
    Connection,
    ConnectionPool,
    ConnectionPoolStats,
)
from happysimulator.components.client.pooled_client import PooledClient, PooledClientStats
from happysimulator.components.client.retry import (
    RetryPolicy,
    NoRetry,
    FixedRetry,
    ExponentialBackoff,
    DecorrelatedJitter,
)

__all__ = [
    "Client",
    "ClientStats",
    "Connection",
    "ConnectionPool",
    "ConnectionPoolStats",
    "PooledClient",
    "PooledClientStats",
    "RetryPolicy",
    "NoRetry",
    "FixedRetry",
    "ExponentialBackoff",
    "DecorrelatedJitter",
]
