---
name: diagnose
description: Troubleshoot a broken or misbehaving simulation
---

# Diagnose Simulation

Find and fix common issues in happysimulator code.

## Instructions

1. Ask the user which file to diagnose if not specified. Also ask what the symptom is (e.g., "nothing happens", "queue grows forever", "error on run").

2. Read the simulation file thoroughly.

3. Check for these common issues, **in priority order**:

### Fatal errors (simulation won't run)

| Issue | What to look for | Fix |
|-------|-------------------|-----|
| Missing target | `Event(...)` without `target=` | Add `target=<entity>` or use `Event.once()` |
| Unregistered entity | Entity created but not in `Simulation(entities=[...])` | Add to entities list |
| Raw float as time | `Event(time=1.0, ...)` instead of `Instant.from_seconds(1.0)` | Wrap with `Instant.from_seconds()` |
| Import errors | Importing from wrong module path | Check `happysimulator/__init__.py` — most things import from top level |

### Silent failures (runs but wrong behavior)

| Issue | What to look for | Fix |
|-------|-------------------|-----|
| Queue never fills | `QueuedResource` without `has_capacity()` override | Override `has_capacity()` to return `False` when at capacity |
| Unbounded queue growth | Arrival rate >= service rate | Reduce arrival rate or increase service capacity |
| Generator not progressing | Yielding `Instant` instead of `float` seconds | `yield 0.1` not `yield Instant.from_seconds(0.1)` |
| Events returned instead of yielded | `yield [Event(...)]` in middle of generator | Use `yield delay, [events]` for mid-generator events; `return [events]` only at end |
| Source event double-counting | Checking `events_processed` against expected count | `Source` generates ~2x events (user events + self-scheduling). Check entity-level counters instead |
| SimFuture never resolves | Creating future but nobody calls `future.resolve(value)` | Ensure the target entity calls `future.resolve()` in its handler |
| SimFuture double-yield | Same future yielded by two generators | Each `SimFuture` can only be yielded by one generator |
| No events generated | `Source.constant(rate=1)` with `stop_after=0.5` | First event at `t=1/rate`. Use higher rate or longer stop_after |

### Performance issues

| Issue | What to look for | Fix |
|-------|-------------------|-----|
| Slow simulation | Very high event rate or very long end_time | Reduce rate, shorten duration, or use coarser time granularity |
| Memory growth | Unbounded queue + long simulation | Add capacity limits, use `BalkingQueue`, or reduce arrival rate |

4. If the issue isn't obvious from static analysis, run the simulation:
   ```
   python <file>
   ```
   Analyze the output for clues (error tracebacks, unexpected metric values, suspiciously fast completion).

5. Report findings clearly:
   - What the issue is
   - Where in the code it occurs (file:line)
   - The fix, with a code snippet
   - Offer to apply the fix

6. If no issues are found, say so and suggest adding instrumentation (`/add-instrumentation`) to investigate further.
