# Component Library Implementation Plan

This document outlines the plan for building a comprehensive component library for simulating software engineering and distributed systems patterns in happy-simulator.

## Design Principles

All components follow the established patterns:
- **Composition over inheritance**: Combine smaller entities into larger abstractions
- **Protocol-based design**: Use `Simulatable` protocol for duck-typing compatibility
- **Generator-friendly**: Express delays naturally with `yield` statements
- **Clock injection**: Components receive time via `set_clock()`
- **Completion hooks**: Enable loose coupling between components
- **Transparent internals**: Hide implementation complexity from external callers

---

## Phase 1: Network Abstractions

**Goal**: Model network behavior including latency, bandwidth limits, and partitions.

### 1.1 Network Link (`components/network/link.py`)
Models a point-to-point network connection with configurable characteristics.

```python
class NetworkLink(Entity):
    """Simulates network transmission delay and bandwidth constraints."""

    def __init__(
        self,
        name: str,
        latency: LatencyDistribution,      # One-way delay distribution
        bandwidth_bps: float | None,        # Bits per second (None = infinite)
        packet_loss_rate: float = 0.0,      # Probability [0, 1] of dropping
        jitter: LatencyDistribution | None = None,
    ): ...
```

**Behavior**:
- Incoming events are delayed by `latency` + transmission time (based on payload size / bandwidth)
- Packet loss randomly drops events with configured probability
- Jitter adds random variation to latency
- Tracks: `bytes_transmitted`, `packets_sent`, `packets_dropped`, `current_utilization`

### 1.2 Network (`components/network/network.py`)
Manages a topology of network links between entities.

```python
class Network(Entity):
    """Routes events through configured network topology."""

    def __init__(self, name: str, default_link: NetworkLink | None = None): ...

    def add_link(self, source: Entity, dest: Entity, link: NetworkLink) -> None: ...
    def add_bidirectional_link(self, a: Entity, b: Entity, link: NetworkLink) -> None: ...
    def partition(self, group_a: list[Entity], group_b: list[Entity]) -> None: ...
    def heal_partition(self) -> None: ...
```

**Behavior**:
- Routes events through appropriate links based on source/destination
- Supports network partitions (events dropped between partitioned groups)
- Default link used when no specific link configured

### 1.3 Network Conditions (`components/network/conditions.py`)
Predefined network profiles for common scenarios.

```python
# Factory functions for common network profiles
def local_network() -> NetworkLink:  # ~0.1ms latency, 1Gbps
def datacenter_network() -> NetworkLink:  # ~0.5ms latency, 10Gbps
def cross_region_network() -> NetworkLink:  # ~50ms latency, 1Gbps
def internet_network() -> NetworkLink:  # ~100ms latency with jitter, packet loss
def satellite_network() -> NetworkLink:  # ~600ms latency, low bandwidth
def lossy_network(loss_rate: float) -> NetworkLink:  # Configurable packet loss
```

---

## Phase 2: Servers and Concurrency

**Goal**: Provide configurable server abstractions with various concurrency models.

### 2.1 Server (`components/server/server.py`)
Base server abstraction with configurable concurrency.

```python
class Server(QueuedResource):
    """Server with configurable concurrency and service time distribution."""

    def __init__(
        self,
        name: str,
        concurrency: int = 1,                    # Max concurrent requests
        service_time: LatencyDistribution = ConstantLatency(0.01),
        queue_policy: QueuePolicy = FIFOQueue(),
        queue_capacity: int | None = None,
    ): ...

    @property
    def active_requests(self) -> int: ...
    @property
    def depth(self) -> int: ...  # Queue depth
    @property
    def utilization(self) -> float: ...  # active / concurrency
```

**Behavior**:
- Accepts up to `concurrency` simultaneous requests
- Additional requests queued according to `queue_policy`
- Service time drawn from distribution per request
- Completion hooks trigger next request processing
- Tracks: `requests_completed`, `requests_rejected`, `total_service_time`

### 2.2 Concurrency Models (`components/server/concurrency.py`)
Different concurrency control strategies.

```python
class ConcurrencyModel(Protocol):
    """Protocol for concurrency control strategies."""
    def acquire(self) -> bool: ...
    def release(self) -> None: ...
    def has_capacity(self) -> bool: ...
    @property
    def available(self) -> int: ...
    @property
    def active(self) -> int: ...

class FixedConcurrency(ConcurrencyModel):
    """Fixed number of concurrent slots."""
    def __init__(self, max_concurrent: int): ...

class DynamicConcurrency(ConcurrencyModel):
    """Adjustable concurrency limit (e.g., for autoscaling)."""
    def __init__(self, initial: int, min_limit: int, max_limit: int): ...
    def set_limit(self, new_limit: int) -> None: ...

class WeightedConcurrency(ConcurrencyModel):
    """Requests consume variable 'weight' from capacity pool."""
    def __init__(self, total_capacity: int): ...
    def acquire(self, weight: int = 1) -> bool: ...
    def release(self, weight: int = 1) -> None: ...
```

### 2.3 Thread Pool (`components/server/thread_pool.py`)
Models a thread/worker pool with task scheduling.

```python
class ThreadPool(Entity):
    """Simulates a pool of worker threads processing tasks."""

    def __init__(
        self,
        name: str,
        num_workers: int,
        queue_policy: QueuePolicy = FIFOQueue(),
        queue_capacity: int | None = None,
    ): ...

    def submit(self, task: Event) -> None: ...

    @property
    def active_workers(self) -> int: ...
    @property
    def idle_workers(self) -> int: ...
    @property
    def queued_tasks(self) -> int: ...
```

**Behavior**:
- Tasks are events with processing time in context or derived from handler
- Workers pick tasks from queue when idle
- Tracks: `tasks_completed`, `tasks_rejected`, `worker_utilization`

### 2.4 Async Server (`components/server/async_server.py`)
Event-loop style server (like Node.js or asyncio).

```python
class AsyncServer(Entity):
    """Non-blocking server that multiplexes many connections on single thread."""

    def __init__(
        self,
        name: str,
        max_connections: int = 10000,
        cpu_work_distribution: LatencyDistribution = ConstantLatency(0),
    ): ...
```

**Behavior**:
- Handles many concurrent requests (high max_connections)
- CPU-bound work blocks all requests (simulated with yield)
- I/O-bound work (network calls) is non-blocking
- Tracks: `active_connections`, `peak_connections`

---

## Phase 3: Clients

**Goal**: Client abstractions with retry logic, connection pooling, and timeout handling.

### 3.1 Client (`components/client/client.py`)
Base client for making requests to servers.

```python
class Client(Entity):
    """Client that sends requests and handles responses."""

    def __init__(
        self,
        name: str,
        target: Entity,
        timeout: float | None = None,
        retry_policy: RetryPolicy | None = None,
    ): ...

    def send_request(self, payload: Any = None) -> Event: ...
```

**Behavior**:
- Sends requests to target entity
- Optional timeout triggers retry or failure callback
- Tracks in-flight requests
- Tracks: `requests_sent`, `responses_received`, `timeouts`, `retries`

### 3.2 Retry Policies (`components/client/retry.py`)

```python
class RetryPolicy(Protocol):
    """Determines retry behavior on failure."""
    def should_retry(self, attempt: int, error: Exception | None) -> bool: ...
    def get_delay(self, attempt: int) -> float: ...

class NoRetry(RetryPolicy):
    """Never retry."""

class FixedRetry(RetryPolicy):
    """Retry with fixed delay."""
    def __init__(self, max_attempts: int, delay: float): ...

class ExponentialBackoff(RetryPolicy):
    """Exponential backoff with optional jitter."""
    def __init__(
        self,
        max_attempts: int,
        initial_delay: float,
        max_delay: float,
        multiplier: float = 2.0,
        jitter: float = 0.0,  # Random factor [0, jitter] added
    ): ...

class DecorrelatedJitter(RetryPolicy):
    """AWS-style decorrelated jitter backoff."""
    def __init__(self, max_attempts: int, base_delay: float, max_delay: float): ...
```

### 3.3 Connection Pool (`components/client/connection_pool.py`)

```python
class ConnectionPool(Entity):
    """Manages a pool of reusable connections to a target."""

    def __init__(
        self,
        name: str,
        target: Entity,
        min_connections: int = 0,
        max_connections: int = 10,
        connection_timeout: float = 5.0,
        idle_timeout: float = 60.0,
        connection_latency: LatencyDistribution = ConstantLatency(0.01),
    ): ...

    def acquire(self) -> Generator[float, None, Connection]: ...
    def release(self, connection: Connection) -> None: ...

    @property
    def active_connections(self) -> int: ...
    @property
    def idle_connections(self) -> int: ...
    @property
    def pending_requests(self) -> int: ...
```

**Behavior**:
- Maintains pool of warm connections
- Creates new connections up to max when pool exhausted
- Requests wait in queue when pool at capacity
- Idle connections closed after timeout
- Tracks: `connections_created`, `connections_closed`, `wait_time`

### 3.4 Pooled Client (`components/client/pooled_client.py`)

```python
class PooledClient(Entity):
    """Client that uses connection pooling for requests."""

    def __init__(
        self,
        name: str,
        connection_pool: ConnectionPool,
        timeout: float | None = None,
        retry_policy: RetryPolicy | None = None,
    ): ...
```

---

## Phase 4: Load Balancers

**Goal**: Distribute traffic across multiple backend servers.

### 4.1 Load Balancer Base (`components/load_balancer/load_balancer.py`)

```python
class LoadBalancer(Entity):
    """Base load balancer that distributes requests to backends."""

    def __init__(
        self,
        name: str,
        backends: list[Entity],
        strategy: LoadBalancingStrategy,
        health_check_interval: float | None = None,
    ): ...

    def add_backend(self, backend: Entity, weight: int = 1) -> None: ...
    def remove_backend(self, backend: Entity) -> None: ...
    def mark_unhealthy(self, backend: Entity) -> None: ...
    def mark_healthy(self, backend: Entity) -> None: ...

    @property
    def healthy_backends(self) -> list[Entity]: ...
```

### 4.2 Load Balancing Strategies (`components/load_balancer/strategies.py`)

```python
class LoadBalancingStrategy(Protocol):
    """Selects a backend for each request."""
    def select(self, backends: list[Entity], request: Event) -> Entity | None: ...

class RoundRobin(LoadBalancingStrategy):
    """Cycles through backends in order."""

class WeightedRoundRobin(LoadBalancingStrategy):
    """Round-robin with weighted distribution."""
    def __init__(self, weights: dict[Entity, int]): ...

class Random(LoadBalancingStrategy):
    """Random backend selection."""

class LeastConnections(LoadBalancingStrategy):
    """Selects backend with fewest active connections."""

class WeightedLeastConnections(LoadBalancingStrategy):
    """Least connections with weights."""

class LeastResponseTime(LoadBalancingStrategy):
    """Selects backend with lowest recent average response time."""

class IPHash(LoadBalancingStrategy):
    """Consistent hashing based on client identifier."""
    def __init__(self, get_key: Callable[[Event], str]): ...

class ConsistentHash(LoadBalancingStrategy):
    """Consistent hashing with virtual nodes."""
    def __init__(self, virtual_nodes: int = 100): ...

class PowerOfTwoChoices(LoadBalancingStrategy):
    """Pick two random backends, choose the one with fewer connections."""
```

### 4.3 Health Checking (`components/load_balancer/health_check.py`)

```python
class HealthChecker(Entity):
    """Periodically checks backend health."""

    def __init__(
        self,
        name: str,
        load_balancer: LoadBalancer,
        interval: float,
        timeout: float,
        healthy_threshold: int = 2,   # Consecutive successes to mark healthy
        unhealthy_threshold: int = 3,  # Consecutive failures to mark unhealthy
    ): ...
```

---

## Phase 5: Circuit Breakers and Resilience

**Goal**: Implement resilience patterns to handle failures gracefully.

### 5.1 Circuit Breaker (`components/resilience/circuit_breaker.py`)

```python
class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery

class CircuitBreaker(Entity):
    """Implements circuit breaker pattern."""

    def __init__(
        self,
        name: str,
        target: Entity,
        failure_threshold: int = 5,      # Failures before opening
        success_threshold: int = 2,      # Successes to close from half-open
        timeout: float = 30.0,           # Time in open state before half-open
        failure_predicate: Callable[[Event], bool] | None = None,
    ): ...

    @property
    def state(self) -> CircuitState: ...
    @property
    def failure_count(self) -> int: ...
```

**Behavior**:
- CLOSED: Forward requests, track failures
- OPEN: Reject requests immediately (fast fail)
- HALF_OPEN: Allow limited requests to test recovery
- Tracks: `total_requests`, `failures`, `rejections`, `state_changes`

### 5.2 Bulkhead (`components/resilience/bulkhead.py`)

```python
class Bulkhead(Entity):
    """Isolates resources to prevent cascade failures."""

    def __init__(
        self,
        name: str,
        target: Entity,
        max_concurrent: int,
        max_wait_queue: int = 0,
        max_wait_time: float | None = None,
    ): ...
```

**Behavior**:
- Limits concurrent requests to target
- Optional waiting queue with timeout
- Rejects when both concurrent and queue limits reached

### 5.3 Timeout Wrapper (`components/resilience/timeout.py`)

```python
class TimeoutWrapper(Entity):
    """Wraps target with timeout handling."""

    def __init__(
        self,
        name: str,
        target: Entity,
        timeout: float,
        on_timeout: Callable[[Event], Event | None] | None = None,
    ): ...
```

### 5.4 Fallback (`components/resilience/fallback.py`)

```python
class Fallback(Entity):
    """Provides fallback behavior on failure."""

    def __init__(
        self,
        name: str,
        primary: Entity,
        fallback: Entity | Callable[[Event], Event],
        failure_predicate: Callable[[Event], bool] | None = None,
    ): ...
```

### 5.5 Hedge (`components/resilience/hedge.py`)

```python
class Hedge(Entity):
    """Sends redundant requests after delay to reduce tail latency."""

    def __init__(
        self,
        name: str,
        target: Entity,
        hedge_delay: float,  # Delay before sending hedge request
        max_hedges: int = 1,
    ): ...
```

---

## Phase 6: Advanced Queue Policies

**Goal**: Implement sophisticated queue management algorithms.

### 6.1 CoDel (Controlled Delay) (`components/queue_policies/codel.py`)

```python
class CoDelQueue(QueuePolicy):
    """CoDel active queue management algorithm."""

    def __init__(
        self,
        target_delay: float = 0.005,  # 5ms target
        interval: float = 0.100,       # 100ms interval
        capacity: int | None = None,
    ): ...
```

**Behavior**:
- Tracks packet sojourn time (time spent in queue)
- Drops packets when delay consistently exceeds target
- Adapts drop rate based on congestion level

### 6.2 RED (Random Early Detection) (`components/queue_policies/red.py`)

```python
class REDQueue(QueuePolicy):
    """Random Early Detection queue management."""

    def __init__(
        self,
        min_threshold: int,     # Start probabilistic dropping
        max_threshold: int,     # 100% drop rate
        max_probability: float = 0.1,
        capacity: int | None = None,
    ): ...
```

### 6.3 Fair Queue (`components/queue_policies/fair_queue.py`)

```python
class FairQueue(QueuePolicy):
    """Per-flow fair queuing."""

    def __init__(
        self,
        get_flow_id: Callable[[Event], str],
        max_flows: int | None = None,
        per_flow_capacity: int | None = None,
    ): ...
```

**Behavior**:
- Maintains separate queue per flow
- Round-robin dequeue across flows
- Prevents single flow from monopolizing

### 6.4 Weighted Fair Queue (`components/queue_policies/weighted_fair_queue.py`)

```python
class WeightedFairQueue(QueuePolicy):
    """Weighted fair queuing with priority classes."""

    def __init__(
        self,
        get_flow_id: Callable[[Event], str],
        get_weight: Callable[[str], int],
        capacity: int | None = None,
    ): ...
```

### 6.5 Deadline Queue (`components/queue_policies/deadline_queue.py`)

```python
class DeadlineQueue(QueuePolicy):
    """Priority by deadline, drops expired items."""

    def __init__(
        self,
        get_deadline: Callable[[Event], Instant],
        capacity: int | None = None,
    ): ...

    def pop(self) -> Event | None:
        """Returns earliest deadline item, or None if all expired."""
```

### 6.6 Adaptive LIFO (`components/queue_policies/adaptive_lifo.py`)

```python
class AdaptiveLIFO(QueuePolicy):
    """LIFO under congestion, FIFO otherwise."""

    def __init__(
        self,
        congestion_threshold: int,  # Queue depth to switch to LIFO
        capacity: int | None = None,
    ): ...
```

---

## Phase 7: Synchronization Primitives

**Goal**: Model concurrency control mechanisms.

### 7.1 Mutex (`components/sync/mutex.py`)

```python
class Mutex(Entity):
    """Mutual exclusion lock with queued waiters."""

    def __init__(self, name: str): ...

    def acquire(self) -> Generator[float, None, None]:
        """Blocks until lock acquired."""

    def release(self) -> list[Event]:
        """Releases lock, wakes next waiter."""

    @property
    def is_locked(self) -> bool: ...
    @property
    def waiters(self) -> int: ...
```

**Usage Pattern**:
```python
def handle_event(self, event):
    yield from self.mutex.acquire()
    try:
        yield 0.01  # Critical section
    finally:
        return self.mutex.release()
```

### 7.2 Semaphore (`components/sync/semaphore.py`)

```python
class Semaphore(Entity):
    """Counting semaphore for resource limiting."""

    def __init__(self, name: str, initial_count: int): ...

    def acquire(self, count: int = 1) -> Generator[float, None, None]: ...
    def release(self, count: int = 1) -> list[Event]: ...

    @property
    def available(self) -> int: ...
```

### 7.3 Read-Write Lock (`components/sync/rwlock.py`)

```python
class RWLock(Entity):
    """Read-write lock allowing concurrent reads or exclusive write."""

    def __init__(self, name: str, max_readers: int | None = None): ...

    def acquire_read(self) -> Generator[float, None, None]: ...
    def acquire_write(self) -> Generator[float, None, None]: ...
    def release_read(self) -> list[Event]: ...
    def release_write(self) -> list[Event]: ...

    @property
    def active_readers(self) -> int: ...
    @property
    def is_write_locked(self) -> bool: ...
```

### 7.4 Condition Variable (`components/sync/condition.py`)

```python
class Condition(Entity):
    """Condition variable for complex synchronization."""

    def __init__(self, name: str, lock: Mutex): ...

    def wait(self) -> Generator[float, None, None]: ...
    def notify(self) -> list[Event]: ...
    def notify_all(self) -> list[Event]: ...
```

### 7.5 Barrier (`components/sync/barrier.py`)

```python
class Barrier(Entity):
    """Synchronization point for multiple processes."""

    def __init__(self, name: str, parties: int): ...

    def wait(self) -> Generator[float, None, None]:
        """Blocks until all parties have called wait()."""
```

---

## Phase 8: Data Stores

**Goal**: Model various storage systems with realistic behavior.

### 8.1 Key-Value Store (`components/datastore/kv_store.py`)

```python
class KVStore(Entity):
    """In-memory key-value store with latency simulation."""

    def __init__(
        self,
        name: str,
        read_latency: LatencyDistribution = ConstantLatency(0.001),
        write_latency: LatencyDistribution = ConstantLatency(0.005),
        capacity: int | None = None,  # Max keys
    ): ...

    def get(self, key: str) -> Generator[float, None, Any]: ...
    def put(self, key: str, value: Any) -> Generator[float, None, None]: ...
    def delete(self, key: str) -> Generator[float, None, bool]: ...

    @property
    def size(self) -> int: ...
```

### 8.2 Cached Store (`components/datastore/cached_store.py`)

```python
class CacheEvictionPolicy(Protocol):
    def on_access(self, key: str) -> None: ...
    def evict(self) -> str | None: ...

class LRUEviction(CacheEvictionPolicy): ...
class LFUEviction(CacheEvictionPolicy): ...
class TTLEviction(CacheEvictionPolicy):
    def __init__(self, ttl: float): ...

class CachedStore(Entity):
    """Cache layer in front of a backing store."""

    def __init__(
        self,
        name: str,
        backing_store: Entity,
        cache_capacity: int,
        eviction_policy: CacheEvictionPolicy,
        read_latency: LatencyDistribution = ConstantLatency(0.0001),
        write_through: bool = True,
    ): ...

    @property
    def hit_rate(self) -> float: ...
    @property
    def miss_rate(self) -> float: ...
```

### 8.3 Replicated Store (`components/datastore/replicated_store.py`)

```python
class ConsistencyLevel(Enum):
    ONE = 1
    QUORUM = "quorum"
    ALL = "all"

class ReplicatedStore(Entity):
    """Distributed key-value store with configurable consistency."""

    def __init__(
        self,
        name: str,
        replicas: list[KVStore],
        read_consistency: ConsistencyLevel = ConsistencyLevel.QUORUM,
        write_consistency: ConsistencyLevel = ConsistencyLevel.QUORUM,
    ): ...
```

**Behavior**:
- Reads/writes to multiple replicas based on consistency level
- QUORUM = majority of replicas
- Tracks: `reads`, `writes`, `read_latency_p99`, `write_latency_p99`

### 8.4 Sharded Store (`components/datastore/sharded_store.py`)

```python
class ShardingStrategy(Protocol):
    def get_shard(self, key: str, num_shards: int) -> int: ...

class HashSharding(ShardingStrategy): ...
class RangeSharding(ShardingStrategy): ...
class ConsistentHashSharding(ShardingStrategy): ...

class ShardedStore(Entity):
    """Horizontally partitioned key-value store."""

    def __init__(
        self,
        name: str,
        shards: list[KVStore],
        sharding_strategy: ShardingStrategy = HashSharding(),
    ): ...
```

### 8.5 Database (`components/datastore/database.py`)

```python
class Database(Entity):
    """Relational database with connection pool and query execution."""

    def __init__(
        self,
        name: str,
        max_connections: int = 100,
        query_latency: Callable[[str], LatencyDistribution] | LatencyDistribution,
        lock_manager: LockManager | None = None,
    ): ...

    def execute(self, query: str) -> Generator[float, None, Any]: ...
    def begin_transaction(self) -> Generator[float, None, Transaction]: ...
```

---

## Phase 9: Rate Limiters (Extensions)

**Goal**: Extend existing rate limiters with additional algorithms.

### 9.1 Fixed Window (`components/rate_limiter/fixed_window.py`)

```python
class FixedWindowRateLimiter(Entity):
    """Simple fixed time window rate limiting."""

    def __init__(
        self,
        name: str,
        target: Entity,
        requests_per_window: int,
        window_size: float,  # seconds
    ): ...
```

### 9.2 Adaptive Rate Limiter (`components/rate_limiter/adaptive.py`)

```python
class AdaptiveRateLimiter(Entity):
    """Adjusts rate limit based on downstream health signals."""

    def __init__(
        self,
        name: str,
        target: Entity,
        initial_rate: float,
        min_rate: float,
        max_rate: float,
        increase_factor: float = 1.1,
        decrease_factor: float = 0.5,
        window_size: float = 1.0,
    ): ...
```

**Behavior**:
- Increases rate on success
- Decreases rate on failure/timeout
- Implements AIMD (Additive Increase, Multiplicative Decrease)

### 9.3 Distributed Rate Limiter (`components/rate_limiter/distributed.py`)

```python
class DistributedRateLimiter(Entity):
    """Rate limiter that coordinates across multiple instances."""

    def __init__(
        self,
        name: str,
        target: Entity,
        backing_store: KVStore,
        global_limit: int,
        window_size: float,
    ): ...
```

---

## Phase 10: Messaging & Pub/Sub

**Goal**: Model asynchronous messaging patterns.

### 10.1 Message Queue (`components/messaging/message_queue.py`)

```python
class MessageQueue(Entity):
    """Persistent message queue with acknowledgment."""

    def __init__(
        self,
        name: str,
        queue_policy: QueuePolicy = FIFOQueue(),
        delivery_latency: LatencyDistribution = ConstantLatency(0.001),
        redelivery_delay: float = 30.0,
        max_redeliveries: int = 3,
    ): ...

    def publish(self, message: Event) -> Generator[float, None, None]: ...
    def subscribe(self, consumer: Entity) -> None: ...
    def acknowledge(self, message_id: str) -> None: ...
    def reject(self, message_id: str, requeue: bool = True) -> None: ...
```

### 10.2 Topic (`components/messaging/topic.py`)

```python
class Topic(Entity):
    """Pub/sub topic with multiple subscribers."""

    def __init__(
        self,
        name: str,
        delivery_latency: LatencyDistribution = ConstantLatency(0.001),
    ): ...

    def publish(self, message: Event) -> Generator[float, None, None]: ...
    def subscribe(self, subscriber: Entity) -> None: ...
    def unsubscribe(self, subscriber: Entity) -> None: ...
```

### 10.3 Dead Letter Queue (`components/messaging/dlq.py`)

```python
class DeadLetterQueue(Entity):
    """Stores messages that failed processing."""

    def __init__(self, name: str, source_queue: MessageQueue): ...

    @property
    def messages(self) -> list[Event]: ...
```

---

## Implementation Order

### Milestone 1: Core Infrastructure
1. Network Link & Network (Phase 1)
2. Server with concurrency (Phase 2.1, 2.2)
3. Basic Client with retry (Phase 3.1, 3.2)

### Milestone 2: Traffic Management
4. Load Balancer with strategies (Phase 4)
5. Advanced Queue Policies: CoDel, RED (Phase 6.1, 6.2)
6. Connection Pool (Phase 3.3)

### Milestone 3: Resilience
7. Circuit Breaker (Phase 5.1)
8. Bulkhead, Timeout, Fallback (Phase 5.2-5.4)
9. Hedge (Phase 5.5)

### Milestone 4: Synchronization & Storage
10. Mutex, Semaphore, RWLock (Phase 7.1-7.3)
11. KV Store, Cached Store (Phase 8.1, 8.2)
12. Thread Pool (Phase 2.3)

### Milestone 5: Advanced Patterns
13. Replicated Store, Sharded Store (Phase 8.3, 8.4)
14. Fair Queue, Weighted Fair Queue (Phase 6.3, 6.4)
15. Distributed Rate Limiter (Phase 9.3)

### Milestone 6: Messaging
16. Message Queue (Phase 10.1)
17. Topic, Dead Letter Queue (Phase 10.2, 10.3)

I want Profiles to be pluggable into variables very easily, so that a component can have a variable parameter according to simulation
time

---

## Testing Strategy

Each component should have:

1. **Unit tests**: Verify behavior in isolation with `ConstantArrivalTimeProvider` and `ConstantLatency`
2. **Integration tests**: Verify composition with other components
3. **Statistical tests**: Verify distributions and probabilistic behavior (e.g., load balancer distribution)
4. **Example scenarios**: Runnable examples demonstrating realistic usage

---

## File Organization

```
happysimulator/
└── components/
    ├── network/
    │   ├── __init__.py
    │   ├── link.py
    │   ├── network.py
    │   └── conditions.py
    ├── server/
    │   ├── __init__.py
    │   ├── server.py
    │   ├── concurrency.py
    │   ├── thread_pool.py
    │   └── async_server.py
    ├── client/
    │   ├── __init__.py
    │   ├── client.py
    │   ├── retry.py
    │   ├── connection_pool.py
    │   └── pooled_client.py
    ├── load_balancer/
    │   ├── __init__.py
    │   ├── load_balancer.py
    │   ├── strategies.py
    │   └── health_check.py
    ├── resilience/
    │   ├── __init__.py
    │   ├── circuit_breaker.py
    │   ├── bulkhead.py
    │   ├── timeout.py
    │   ├── fallback.py
    │   └── hedge.py
    ├── queue_policies/
    │   ├── __init__.py
    │   ├── codel.py
    │   ├── red.py
    │   ├── fair_queue.py
    │   ├── weighted_fair_queue.py
    │   ├── deadline_queue.py
    │   └── adaptive_lifo.py
    ├── sync/
    │   ├── __init__.py
    │   ├── mutex.py
    │   ├── semaphore.py
    │   ├── rwlock.py
    │   ├── condition.py
    │   └── barrier.py
    ├── datastore/
    │   ├── __init__.py
    │   ├── kv_store.py
    │   ├── cached_store.py
    │   ├── replicated_store.py
    │   ├── sharded_store.py
    │   └── database.py
    ├── rate_limiter/
    │   ├── __init__.py          # Re-export existing
    │   ├── fixed_window.py
    │   ├── adaptive.py
    │   └── distributed.py
    └── messaging/
        ├── __init__.py
        ├── message_queue.py
        ├── topic.py
        └── dlq.py
```

---

## API Exports

Update `happysimulator/__init__.py` to export all components:

```python
# Network
from .components.network import NetworkLink, Network, local_network, datacenter_network

# Server
from .components.server import Server, ThreadPool, AsyncServer
from .components.server import FixedConcurrency, DynamicConcurrency

# Client
from .components.client import Client, PooledClient, ConnectionPool
from .components.client import NoRetry, FixedRetry, ExponentialBackoff

# Load Balancer
from .components.load_balancer import LoadBalancer, HealthChecker
from .components.load_balancer import RoundRobin, LeastConnections, ConsistentHash

# Resilience
from .components.resilience import CircuitBreaker, Bulkhead, TimeoutWrapper, Fallback, Hedge

# Queue Policies
from .components.queue_policies import CoDelQueue, REDQueue, FairQueue, DeadlineQueue

# Sync
from .components.sync import Mutex, Semaphore, RWLock, Condition, Barrier

# Datastore
from .components.datastore import KVStore, CachedStore, ReplicatedStore, ShardedStore

# Messaging
from .components.messaging import MessageQueue, Topic, DeadLetterQueue
```
