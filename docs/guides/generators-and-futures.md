# Generators & Futures

Generators let you express multi-step processing with natural `yield` syntax. SimFutures extend this to cross-entity coordination.

## Yield Forms

Inside `handle_event()`, you can yield in three ways:

### Yield a delay

```python
def handle_event(self, event):
    yield 0.1  # pause for 100ms of simulation time
    # ... resume here after 0.1s
```

### Yield a delay with side-effect events

```python
def handle_event(self, event):
    yield 0.01, [Event(time=self.now, event_type="Ack", target=client)]
    # pause for 10ms AND schedule the Ack event immediately
```

This is useful for sending an acknowledgement before continuing with more processing.

### Yield a SimFuture

```python
def handle_event(self, event):
    future = SimFuture()
    yield 0.0, [Event(time=self.now, event_type="Req", target=server,
                       context={"reply": future})]
    response = yield future  # park until future.resolve(value)
```

The generator is suspended until another entity calls `future.resolve(value)`.

## Return Events

When a generator finishes, its return value is scheduled as events:

```python
def handle_event(self, event):
    yield 0.1
    return [Event(time=self.now, event_type="Done", target=self.downstream)]
```

## SimFuture

`SimFuture` enables request-response patterns across entities.

### Basic Request-Response

```python
from happysimulator import SimFuture

class Client(Entity):
    def handle_event(self, event):
        future = SimFuture()
        yield 0.0, [Event(time=self.now, event_type="Req", target=self.server,
                          context={"reply": future})]
        response = yield future
        print(f"Got response: {response}")

class Server(Entity):
    def handle_event(self, event):
        yield 0.05  # process
        event.context["reply"].resolve("OK")
```

### Timeout with `any_of`

Race a response against a timeout:

```python
from happysimulator import SimFuture, any_of

class Client(Entity):
    def handle_event(self, event):
        response = SimFuture()
        timeout = SimFuture()

        yield 0.0, [
            Event(time=self.now, event_type="Req", target=self.server,
                  context={"reply": response}),
            Event.once(time=self.now + Duration.from_seconds(1.0),
                       event_type="Timeout",
                       fn=lambda e: timeout.resolve("timeout")),
        ]

        index, value = yield any_of(response, timeout)
        if index == 0:
            print(f"Got response: {value}")
        else:
            print("Request timed out")
```

`any_of()` resolves with `(index, value)` — the index of whichever future resolved first.

### Quorum with `all_of`

Wait for all futures to resolve:

```python
from happysimulator import SimFuture, all_of

class Coordinator(Entity):
    def handle_event(self, event):
        futures = [SimFuture() for _ in range(3)]
        yield 0.0, [
            Event(time=self.now, event_type="Req", target=replica,
                  context={"reply": f})
            for replica, f in zip(self.replicas, futures)
        ]
        results = yield all_of(*futures)  # [value1, value2, value3]
```

## Rules

- Each `SimFuture` can only be yielded by **one** generator — yielding the same future from two generators raises an error
- Pre-resolved futures resume immediately when yielded
- There is no `fail()` — simulated failures are just values (e.g., `future.resolve(Error("timeout"))`)

## Next Steps

- [Load Generation](load-generation.md) — Source factories and custom providers
- [Queuing & Resources](queuing-and-resources.md) — buffering, concurrency, and rate limiting
