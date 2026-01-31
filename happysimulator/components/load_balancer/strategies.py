"""Load balancing strategies for distributing requests across backends.

Provides pluggable algorithms that determine which backend should handle
each incoming request. Includes common patterns like round-robin, least
connections, and consistent hashing.

Example:
    from happysimulator.components.load_balancer import LoadBalancer, RoundRobin

    lb = LoadBalancer(
        name="api_lb",
        backends=[server1, server2, server3],
        strategy=RoundRobin(),
    )
"""

import hashlib
import logging
import random
from typing import Callable, Protocol, runtime_checkable

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event

logger = logging.getLogger(__name__)


@runtime_checkable
class LoadBalancingStrategy(Protocol):
    """Protocol for load balancing algorithms.

    Implementations select which backend should handle a given request
    from the list of available (healthy) backends.
    """

    def select(self, backends: list[Entity], request: Event) -> Entity | None:
        """Select a backend to handle the request.

        Args:
            backends: List of available backends to choose from.
            request: The incoming request event.

        Returns:
            The selected backend, or None if no backend available.
        """
        ...


class RoundRobin:
    """Cycles through backends in sequential order.

    Simple and fair distribution that gives each backend an equal
    share of requests over time. Best when backends have similar
    capacity and request handling times are uniform.
    """

    def __init__(self):
        self._index = 0

    def select(self, backends: list[Entity], request: Event) -> Entity | None:
        """Select the next backend in round-robin order."""
        if not backends:
            return None

        backend = backends[self._index % len(backends)]
        self._index += 1
        return backend

    def reset(self) -> None:
        """Reset the round-robin counter."""
        self._index = 0


class WeightedRoundRobin:
    """Round-robin with weighted distribution.

    Backends with higher weights receive proportionally more requests.
    Useful when backends have different capacities.

    Example:
        # server1 gets 3x the traffic of server2
        strategy = WeightedRoundRobin()
        strategy.set_weight(server1, 3)
        strategy.set_weight(server2, 1)
    """

    def __init__(self):
        self._weights: dict[str, int] = {}  # backend.name -> weight
        self._current_weights: dict[str, int] = {}
        self._default_weight = 1

    def set_weight(self, backend: Entity, weight: int) -> None:
        """Set the weight for a backend.

        Args:
            backend: The backend entity.
            weight: Weight value (higher = more traffic).

        Raises:
            ValueError: If weight is less than 1.
        """
        if weight < 1:
            raise ValueError(f"weight must be >= 1, got {weight}")
        self._weights[backend.name] = weight

    def get_weight(self, backend: Entity) -> int:
        """Get the weight for a backend."""
        return self._weights.get(backend.name, self._default_weight)

    def select(self, backends: list[Entity], request: Event) -> Entity | None:
        """Select backend using weighted round-robin (smooth weighted)."""
        if not backends:
            return None

        # Initialize current weights for new backends
        for b in backends:
            if b.name not in self._current_weights:
                self._current_weights[b.name] = 0

        # Smooth weighted round-robin algorithm
        total_weight = sum(self.get_weight(b) for b in backends)

        # Increase current weight by effective weight
        for b in backends:
            self._current_weights[b.name] += self.get_weight(b)

        # Select backend with highest current weight
        selected = max(backends, key=lambda b: self._current_weights[b.name])

        # Decrease selected backend's current weight by total
        self._current_weights[selected.name] -= total_weight

        return selected


class Random:
    """Random backend selection.

    Simple strategy that randomly selects from available backends.
    Provides good distribution over many requests without maintaining
    state.
    """

    def select(self, backends: list[Entity], request: Event) -> Entity | None:
        """Select a random backend."""
        if not backends:
            return None
        return random.choice(backends)


class LeastConnections:
    """Selects backend with fewest active connections.

    Directs traffic to the least loaded backend, which helps balance
    load when request handling times vary. Requires backends to expose
    their current connection count.

    The strategy looks for an `active_connections` or `active_requests`
    property on backends, falling back to 0 if not found.
    """

    def _get_connections(self, backend: Entity) -> int:
        """Get the number of active connections for a backend."""
        # Try common property names
        if hasattr(backend, 'active_connections'):
            return backend.active_connections
        if hasattr(backend, 'active_requests'):
            return backend.active_requests
        if hasattr(backend, 'in_flight_count'):
            return backend.in_flight_count
        # Check for stats object
        if hasattr(backend, 'stats'):
            stats = backend.stats
            if hasattr(stats, 'active_requests'):
                return stats.active_requests
        return 0

    def select(self, backends: list[Entity], request: Event) -> Entity | None:
        """Select the backend with fewest active connections."""
        if not backends:
            return None

        # Find backend with minimum connections
        # In case of tie, pick the first one (stable selection)
        return min(backends, key=self._get_connections)


class WeightedLeastConnections:
    """Least connections with weights.

    Combines connection count with backend weights. A backend with
    weight 2 and 4 connections is equivalent to a backend with weight 1
    and 2 connections.

    Score = connections / weight (lower is better)
    """

    def __init__(self):
        self._weights: dict[str, int] = {}
        self._default_weight = 1

    def set_weight(self, backend: Entity, weight: int) -> None:
        """Set the weight for a backend."""
        if weight < 1:
            raise ValueError(f"weight must be >= 1, got {weight}")
        self._weights[backend.name] = weight

    def get_weight(self, backend: Entity) -> int:
        """Get the weight for a backend."""
        return self._weights.get(backend.name, self._default_weight)

    def _get_connections(self, backend: Entity) -> int:
        """Get the number of active connections for a backend."""
        if hasattr(backend, 'active_connections'):
            return backend.active_connections
        if hasattr(backend, 'active_requests'):
            return backend.active_requests
        if hasattr(backend, 'in_flight_count'):
            return backend.in_flight_count
        if hasattr(backend, 'stats'):
            stats = backend.stats
            if hasattr(stats, 'active_requests'):
                return stats.active_requests
        return 0

    def _get_score(self, backend: Entity) -> float:
        """Calculate weighted score (lower is better)."""
        connections = self._get_connections(backend)
        weight = self.get_weight(backend)
        return connections / weight

    def select(self, backends: list[Entity], request: Event) -> Entity | None:
        """Select backend with lowest weighted connection score."""
        if not backends:
            return None
        return min(backends, key=self._get_score)


class LeastResponseTime:
    """Selects backend with lowest recent average response time.

    Tracks response times and directs traffic to the fastest backend.
    Uses exponential moving average to weight recent responses more
    heavily.
    """

    def __init__(self, alpha: float = 0.3):
        """Initialize least response time strategy.

        Args:
            alpha: Smoothing factor for EMA (0-1). Higher values give
                   more weight to recent observations.
        """
        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self._alpha = alpha
        self._response_times: dict[str, float] = {}  # EMA of response times
        self._default_response_time = 0.0

    def record_response_time(self, backend: Entity, response_time: float) -> None:
        """Record a response time observation for a backend.

        Args:
            backend: The backend that handled the request.
            response_time: Time taken to handle the request in seconds.
        """
        name = backend.name
        if name in self._response_times:
            # Exponential moving average
            self._response_times[name] = (
                self._alpha * response_time +
                (1 - self._alpha) * self._response_times[name]
            )
        else:
            self._response_times[name] = response_time

    def get_response_time(self, backend: Entity) -> float:
        """Get the current average response time for a backend."""
        return self._response_times.get(backend.name, self._default_response_time)

    def select(self, backends: list[Entity], request: Event) -> Entity | None:
        """Select backend with lowest average response time."""
        if not backends:
            return None

        # Backends without data get priority (to collect data)
        unknown = [b for b in backends if b.name not in self._response_times]
        if unknown:
            return random.choice(unknown)

        return min(backends, key=self.get_response_time)


class IPHash:
    """Consistent hashing based on client identifier.

    Routes requests from the same client to the same backend, useful
    for session affinity. Falls back to round-robin if no key found.
    """

    def __init__(self, get_key: Callable[[Event], str] | None = None):
        """Initialize IP hash strategy.

        Args:
            get_key: Function to extract routing key from request.
                     Defaults to looking for 'client_id' in metadata.
        """
        self._get_key = get_key or self._default_get_key
        self._fallback = RoundRobin()

    def _default_get_key(self, request: Event) -> str | None:
        """Default key extraction from request metadata."""
        metadata = request.context.get("metadata", {})
        # Try common key names
        for key in ["client_id", "client_ip", "session_id", "user_id"]:
            if key in metadata:
                return str(metadata[key])
        return None

    def select(self, backends: list[Entity], request: Event) -> Entity | None:
        """Select backend based on hashed client key."""
        if not backends:
            return None

        key = self._get_key(request)
        if key is None:
            # Fall back to round-robin
            return self._fallback.select(backends, request)

        # Hash the key and map to backend
        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
        index = hash_value % len(backends)
        return backends[index]


class ConsistentHash:
    """Consistent hashing with virtual nodes.

    Provides stable routing that minimizes remapping when backends
    are added or removed. Each backend is mapped to multiple points
    on a hash ring (virtual nodes) for better distribution.
    """

    def __init__(
        self,
        virtual_nodes: int = 100,
        get_key: Callable[[Event], str] | None = None,
    ):
        """Initialize consistent hash strategy.

        Args:
            virtual_nodes: Number of virtual nodes per backend.
            get_key: Function to extract routing key from request.
        """
        if virtual_nodes < 1:
            raise ValueError(f"virtual_nodes must be >= 1, got {virtual_nodes}")
        self._virtual_nodes = virtual_nodes
        self._get_key = get_key or self._default_get_key
        self._ring: list[tuple[int, str]] = []  # (hash, backend_name)
        self._backends: dict[str, Entity] = {}
        self._fallback = RoundRobin()

    def _default_get_key(self, request: Event) -> str | None:
        """Default key extraction from request metadata."""
        metadata = request.context.get("metadata", {})
        for key in ["client_id", "client_ip", "session_id", "user_id", "key"]:
            if key in metadata:
                return str(metadata[key])
        return None

    def _hash(self, key: str) -> int:
        """Compute hash value for a key."""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def add_backend(self, backend: Entity) -> None:
        """Add a backend to the hash ring."""
        self._backends[backend.name] = backend
        # Add virtual nodes
        for i in range(self._virtual_nodes):
            virtual_key = f"{backend.name}:{i}"
            hash_value = self._hash(virtual_key)
            self._ring.append((hash_value, backend.name))
        # Sort ring by hash value
        self._ring.sort(key=lambda x: x[0])

    def remove_backend(self, backend: Entity) -> None:
        """Remove a backend from the hash ring."""
        if backend.name in self._backends:
            del self._backends[backend.name]
            self._ring = [(h, n) for h, n in self._ring if n != backend.name]

    def _rebuild_ring(self, backends: list[Entity]) -> None:
        """Rebuild the hash ring from the current backends."""
        current_names = {b.name for b in backends}
        ring_names = {n for _, n in self._ring}

        # Add new backends
        for backend in backends:
            if backend.name not in ring_names:
                self.add_backend(backend)

        # Remove missing backends
        for name in ring_names:
            if name not in current_names:
                self._ring = [(h, n) for h, n in self._ring if n != name]
                if name in self._backends:
                    del self._backends[name]

    def select(self, backends: list[Entity], request: Event) -> Entity | None:
        """Select backend using consistent hashing."""
        if not backends:
            return None

        # Ensure ring is up to date
        self._rebuild_ring(backends)

        key = self._get_key(request)
        if key is None:
            return self._fallback.select(backends, request)

        if not self._ring:
            return self._fallback.select(backends, request)

        # Find first node on ring >= hash(key)
        hash_value = self._hash(key)

        for ring_hash, backend_name in self._ring:
            if ring_hash >= hash_value:
                if backend_name in self._backends:
                    return self._backends[backend_name]

        # Wrap around to first node
        first_name = self._ring[0][1]
        return self._backends.get(first_name)


class PowerOfTwoChoices:
    """Pick two random backends, choose the one with fewer connections.

    A probabilistic approach that provides near-optimal load balancing
    with low overhead. Better than pure random, simpler than least
    connections.

    Reference: "The Power of Two Choices in Randomized Load Balancing"
    by Mitzenmacher, Richa, and Sitaraman.
    """

    def _get_connections(self, backend: Entity) -> int:
        """Get the number of active connections for a backend."""
        if hasattr(backend, 'active_connections'):
            return backend.active_connections
        if hasattr(backend, 'active_requests'):
            return backend.active_requests
        if hasattr(backend, 'in_flight_count'):
            return backend.in_flight_count
        if hasattr(backend, 'stats'):
            stats = backend.stats
            if hasattr(stats, 'active_requests'):
                return stats.active_requests
        return 0

    def select(self, backends: list[Entity], request: Event) -> Entity | None:
        """Select from two random backends the one with fewer connections."""
        if not backends:
            return None

        if len(backends) == 1:
            return backends[0]

        # Pick two random backends
        choice1, choice2 = random.sample(backends, 2)

        # Return the one with fewer connections
        if self._get_connections(choice1) <= self._get_connections(choice2):
            return choice1
        return choice2
