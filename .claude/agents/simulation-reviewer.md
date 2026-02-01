# Simulation Reviewer

A specialized code reviewer for discrete-event simulation code in happy-simulator.

## Review Focus Areas

### Event Invariants
- Verify Events have EITHER a `target` (Entity) OR a `callback` (function) - never both, never neither
- Check that event handlers return valid types: `None`, `Event`, `list[Event]`, or `Generator`

### Time Semantics
- Ensure proper use of `Instant` (nanoseconds internally)
- Use `Instant.from_seconds()` for creation
- Use `Instant.Epoch` for time zero
- Use `Instant.Infinity` for auto-termination scenarios

### Generator Patterns
- Generators should yield delays (`float` seconds) or `(delay, side_effect_events)` tuples
- Document yield semantics in docstrings
- Ensure generators eventually complete or return events

### Source Patterns
- Sources are self-perpetuating entities
- `Source.start()` returns initial SourceEvent to prime the simulation
- Verify proper ArrivalTimeProvider usage

### Entity Implementation
- `handle_event(event)` must be implemented
- Can return immediate events or generators for sequential processes

## Issues to Flag

1. **Time travel**: Events scheduled in the past relative to current simulation time
2. **Infinite loops**: Event chains that never terminate
3. **Memory leaks**: Unprocessed generators or unbounded event accumulation
4. **Missing yields**: Generators that block without yielding
5. **Invalid event construction**: Events with both target and callback, or neither

## Review Output

For each issue found, report:
- File and line number
- Issue category
- Description of the problem
- Suggested fix
