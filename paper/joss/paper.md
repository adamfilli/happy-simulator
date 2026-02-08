---
title: "happy-simulator: A Discrete-Event Simulation Library for Distributed Systems Engineering"
tags:
  - Python
  - discrete-event simulation
  - distributed systems
  - queueing theory
  - resilience engineering
authors:
  - name: Adam Filli
    orcid: 0000-0000-0000-0000
    corresponding: true
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 8 February 2026
bibliography: paper.bib
---

# Summary

`happy-simulator` is a discrete-event simulation (DES) library for Python 3.13+
that enables software engineers to model distributed systems and observe emergent
behavior before deployment. The library provides a generator-based process model
where `yield` statements represent actual simulated delays, nanosecond-precision
integer time to eliminate floating-point drift, and a component library of over
100 pre-built distributed-systems primitives spanning caches, circuit breakers,
rate limiters, queue policies, load balancers, and streaming algorithms. Engineers
compose these components into system models and run simulations to discover
failure modes---such as metastable collapse and retry amplification---that are
difficult to predict analytically and dangerous to encounter in production.

# Statement of Need

Software engineers designing distributed systems face a reasoning gap between two
established tools. On one side, analytical queueing theory
[@kleinrock1975queueing; @harchol2013performance] provides closed-form results
for idealized models (e.g., M/M/1 steady-state queue length) but cannot capture
the feedback loops and cascading failures characteristic of real systems. On the
other side, load testing and chaos engineering [@basiri2016chaos] exercise running
systems but require a deployed environment, arrive late in the development cycle,
and cannot exhaustively explore parameter spaces.

Discrete-event simulation occupies a natural middle ground, allowing engineers to
model system behavior at design time with arbitrary fidelity. However, existing
Python DES libraries---notably SimPy [@simpy] and salabim
[@salabim]---are designed for operations research and manufacturing workflows.
They provide generic process and resource primitives but lack vocabulary for
distributed-systems concerns: there are no built-in circuit breakers, retry
policies, cache eviction strategies, or queue management algorithms like CoDel
[@nichols2012codel]. Engineers must implement these from scratch for each model,
which discourages adoption and increases the likelihood of modeling errors.

Network simulators such as ns-3 [@riley2010ns3] and OMNeT++ [@varga2010omnet]
operate at the packet level, a granularity too fine for application-level
reasoning about service interactions, timeout behavior, and resource contention.

`happy-simulator` fills this gap by providing an application-level DES engine
with batteries-included distributed-systems primitives. Engineers describe
systems using familiar concepts---servers, clients, caches, load balancers---and
the simulation reveals emergent behaviors that resist analytical prediction. The
library's component library encodes established patterns from production systems
engineering [@nygard2018release], making them composable and reusable across
simulation models.

# Key Design Decisions

**Generator-based process model.** Entity behavior is expressed as Python
generators where each `yield` statement pauses the process for a specified
duration. This maps naturally to multi-step request processing (network hops,
computation, I/O waits) without callback nesting. Generators can also yield
side-effect events for modeling concurrent interactions during a single process.

**Nanosecond integer time.** Simulation time is represented as 64-bit integer
nanoseconds via the `Instant` class, eliminating floating-point accumulation
errors that cause event-ordering bugs in long-running simulations. All arithmetic
is exact, and comparison operators work without epsilon tolerances.

**Uniform target-based dispatch.** Every event carries a `target` entity
reference. The simulation loop dispatches events by calling
`target.handle_event(event)`, producing a single code path for all event types.
Function-based dispatch is supported via `Event.once()`, which wraps a callable
in a lightweight `CallbackEntity`.

**Composition over inheritance.** Complex components are built by composing
simpler ones. For example, `QueuedResource` combines a queue policy with
processing logic, and `CachedStore` layers a cache eviction policy over a backing
store. This enables mixing and matching---any of 10 eviction policies with any of
4 write-through strategies---without combinatorial class hierarchies.

# Component Library

`happy-simulator` ships with over 100 component implementations organized into
13 categories, summarized in \autoref{tab:components}.

: Component library categories with implementation counts. \label{tab:components}

| Category               | Implementations | Examples                                          |
|:-----------------------|:---------------:|:--------------------------------------------------|
| Queue policies         | 6               | CoDel, RED, fair, weighted fair, deadline, adaptive LIFO |
| Cache eviction         | 10              | LRU, LFU, SLRU, 2Q, clock, sampled-LRU, TTL      |
| Cache write policies   | 4               | Write-through, write-back, write-around            |
| Resilience patterns    | 5               | Circuit breaker, bulkhead, timeout, fallback, hedge |
| Rate limiters          | 6               | Token bucket, leaky bucket, sliding window, adaptive |
| Load balancers         | 9               | Round-robin, least-connections, power-of-two, consistent hash |
| Server models          | 7               | Async, thread pool, fixed/dynamic/weighted concurrency |
| Client models          | 8               | Connection pool, exponential backoff, decorrelated jitter |
| Network conditions     | 9               | Datacenter, cross-region, satellite, mobile 3G/4G  |
| Sync primitives        | 5               | Mutex, semaphore, read-write lock, condition, barrier |
| Messaging              | 3               | Message queue, pub/sub topic, dead-letter queue     |
| Streaming algorithms   | 6               | TopK, CountMinSketch, TDigest, HyperLogLog, Bloom filter |
| Datastore              | 8               | KV store, sharded store, replicated store, multi-tier cache |

# Illustrative Example

Metastable failures---where a system enters a self-sustaining degraded state that
persists even after the triggering condition resolves---are a significant concern
in distributed systems [@huang2022metastable]. These failures are notoriously
difficult to reason about because they emerge from feedback loops between
components.

The included `examples/gc_caused_collapse.py` demonstrates how a single
1-second garbage collection pause at 70% server utilization triggers permanent
system collapse. The server has 30% spare capacity and a generous 500ms client
timeout (5x the 100ms service time). During the GC pause, all in-flight requests
exceed the timeout. Clients retry, creating a load spike that exceeds the
server's capacity. The resulting queue buildup causes further timeouts, which
cause further retries---a positive feedback loop that sustains collapse
indefinitely. A control simulation without retries shows immediate recovery after
the same GC pause, isolating retry amplification as the mechanism.

This scenario, expressed in approximately 150 lines of `happy-simulator` code,
reveals behavior that would be difficult to predict from the individual
component specifications alone: 30% spare capacity is insufficient to absorb a
transient disruption when retry amplification is present. The simulation enables
engineers to discover this before deployment and evaluate mitigations (e.g.,
retry budgets, circuit breakers) in the same model.

# AI Usage Disclosure

Generative AI tools (Claude, Anthropic) were used as a development aid during
the creation of both the software and this manuscript. All generated content was
reviewed and validated by the author.

# Acknowledgements

The author thanks the open-source simulation community, particularly the
developers of SimPy and salabim, whose work informed the design of this library.

# References
