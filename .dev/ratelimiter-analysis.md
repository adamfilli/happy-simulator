# Rate Limiter Audit: Rejection Behavior

## Context

Rate limiters in real systems typically do one of three things when a request exceeds the limit:
1. **Queue** the request and process it later (delay)
2. **Reject with feedback** — return a response (e.g., HTTP 429) so the caller can retry
3. **Drop silently** — discard the request with no feedback

Option 3 (silent drop) is the least realistic for most use cases. Real HTTP APIs return 429s, real load balancers return 503s, real message brokers apply backpressure. Silent drops model packet loss at the network layer, but not application-layer rate limiting.

---

## Audit Findings

### Summary Table

| Limiter | File | On Limit Exceeded | Caller Gets Feedback? |
|---------|------|-------------------|-----------------------|
| **Token Bucket** | `components/token_bucket_rate_limiter.py` | Silent drop (`return []`) | No |
| **Leaky Bucket** | `components/leaky_bucket_rate_limiter.py` | Queue if space; silent drop if full (`return []`) | No |
| **Sliding Window** | `components/sliding_window_rate_limiter.py` | Silent drop (`return []`) | No |
| **Fixed Window** | `components/rate_limiter/fixed_window.py` | Silent drop (`return []`) | No |
| **Adaptive** | `components/rate_limiter/adaptive.py` | Silent drop (`return []`) | No |
| **Distributed** | `components/rate_limiter/distributed.py` | Silent drop (`return []`) | No |

### Key Observation

**All 6 limiters return `[]` (empty event list) when rejecting.** This means:
- The original caller/source entity receives **no signal** that the request was dropped
- There is no "rejected" or "throttled" event sent back upstream
- The request simply vanishes from the simulation
- Stats are tracked internally (`requests_dropped`), but no simulation participant can react to the rejection

The **Leaky Bucket** is the only limiter that queues requests (delaying them via scheduled leak events). But even it silently drops when the queue is full — no rejection event is sent back.

### Detailed Behavior Per Limiter

**Token Bucket** (`token_bucket_rate_limiter.py:162-170`)
- Refills tokens based on elapsed time, consumes 1 per request
- `if self._tokens >= 1.0` → forward; else → `return []`
- Tracks `dropped_times` but produces no rejection event

**Leaky Bucket** (`leaky_bucket_rate_limiter.py:130-150`)
- Best of the bunch: queues requests and leaks them at a fixed rate
- `if len(self._queue) < self._capacity` → queue + schedule leak; else → `return []`
- Queue-full drops are still silent

**Sliding Window** (`sliding_window_rate_limiter.py:136-164`)
- Maintains timestamp log, prunes expired entries
- `if current_count < self._max_requests` → forward; else → `return []`

**Fixed Window** (`rate_limiter/fixed_window.py:166-200`)
- Counts per discrete time window, resets at boundaries
- `if self._current_window_count < self._requests_per_window` → forward; else → `return []`

**Adaptive (AIMD)** (`rate_limiter/adaptive.py:284-318`)
- Token bucket with dynamic refill rate (AIMD adjustments)
- `if self._tokens >= 1.0` → forward; else → `return []`
- Has `record_success()`/`record_failure()` for rate adjustment, but these are called *by the user*, not by the limiter itself on rejection

**Distributed** (`rate_limiter/distributed.py:267-308`)
- Generator-based: yields during backing store access
- Checks global counter in shared KVStore
- `if allowed` → forward; else → `return []`

### What "Drop" Means in Practice

When a limiter returns `[]`:
1. The simulation loop has nothing to schedule — the event chain ends
2. If a `Source` generated the request, the Source doesn't know it was dropped
3. If a client entity sent the request, it receives no response event
4. No timeout or retry logic can fire because there's no "pending" state
5. The request is effectively a black hole

### What Would Be More Realistic

For future reference (no changes now), typical real-world patterns:

- **Rejection event**: Return an event with `event_type="rejected::..."` targeted at the *caller* (requires knowing the caller, possibly via `event.context["reply_to"]`)
- **Backpressure signal**: Return an event to the Source to slow down arrival rate
- **Retry-After**: Include metadata like `context={"retry_after_seconds": 1.0}` so the caller can schedule a retry
- **Queue with overflow**: Like Leaky Bucket but with a rejection event when full instead of silent drop

---

## No Changes Proposed

This is an audit-only document. The current implementations are internally consistent and well-tested — they just model a "drop on overload" pattern rather than "reject with feedback."
