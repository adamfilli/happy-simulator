# Happy Simulator vs SimPy: Differentiation

## Core Architectural Difference

**SimPy is a process-based DES.** You write standalone generator functions:
```python
# SimPy: processes are free-floating generators
def customer(env, server):
    yield env.timeout(random.expo(1.0))  # arrive
    req = server.request()
    yield req                             # wait for server
    yield env.timeout(random.expo(0.5))  # service
    server.release(req)
```
There are no entities, no event types, no message routing. Processes coordinate through shared resources and bare events. It's generic — used for manufacturing lines, hospital queues, traffic flow, anything.

**Happy-simulator is an actor-based DES.** Stateful entities receive typed, routed events:
```python
# Happy-simulator: entities react to messages
class Server(Entity):
    def handle_event(self, event: Event):
        yield 0.1
        return [Event(time=self.now, event_type="Done", target=self.downstream, context=event.context)]
```

## Comparison Table

| | SimPy | Happy-simulator |
|---|---|---|
| **Paradigm** | Process-centric (generators *are* the actors) | Entity/actor-centric (stateful objects receive messages) |
| **Events** | Bare synchronization primitives | Rich domain objects (type, target, context, tracing, completion hooks) |
| **Dispatch** | None — processes just run | Target-based routing through event heap (it's a message bus) |
| **Domain** | Generic (factories, hospitals, traffic) | Software systems (networks, caches, rate limiters, queues) |
| **Component library** | Almost none | NetworkLink, QueuedResource, Inductor, CachedStore, rate limiters, sketching, etc. |
| **Observability** | DIY | Data, Probe, LatencyTracker, BucketedData, SimulationSummary |
| **Debugging** | None | Pause/step/breakpoints, event hooks, trace spans |
| **Time** | Float | Nanosecond integer (deterministic) |

## Why the Entity Model Matters

The entity model maps naturally to how software systems actually work — servers, brokers, caches, load balancers are all **stateful things that react to messages**. SimPy forces you to model everything as processes that happen to share resources, which gets awkward fast for component-based architectures.

## SimFuture and SimPy

The proposed SimFuture feature borrows one *mechanism* from SimPy (yielding on events, not just delays) but doesn't change the fundamental architecture. SimFuture fills a gap in happy-simulator's generator semantics — it's like how many languages borrowed garbage collection from Lisp without becoming Lisp.

SimPy's event model:
```python
event = env.event()
value = yield event          # park until triggered
event.succeed(result)        # resume the yielder

req = resource.request()
yield req                    # park until capacity available

yield env.any_of([event1, event2])   # first to fire
yield env.all_of([event1, event2])   # all must fire
```

Happy-simulator's proposed SimFuture (adapted to the Entity model):
```python
future = SimFuture()
value = yield future              # park until resolved
future.resolve(result)            # resume the yielder

grant = yield resource.acquire()  # park until capacity available

yield any_of(timeout, response)   # first to resolve
yield all_of(ack1, ack2, ack3)    # quorum wait
```

The mapping is direct, but the context is completely different — SimFuture lives inside `Entity.handle_event()` generators and integrates with the `ProcessContinuation` system, target-based dispatch, and the full event tracing infrastructure.
