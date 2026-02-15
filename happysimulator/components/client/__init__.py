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
    DecorrelatedJitter,
    ExponentialBackoff,
    FixedRetry,
    NoRetry,
    RetryPolicy,
)

__all__ = [
    "Client",
    "ClientStats",
    "Connection",
    "ConnectionPool",
    "ConnectionPoolStats",
    "DecorrelatedJitter",
    "ExponentialBackoff",
    "FixedRetry",
    "NoRetry",
    "PooledClient",
    "PooledClientStats",
    "RetryPolicy",
]
