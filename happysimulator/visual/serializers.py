"""Entity state serialization to JSON-safe dicts.

Type-aware registry with safe fallback for custom entities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from happysimulator.core.event import Event


def serialize_entity(entity: object) -> dict[str, Any]:
    """Serialize an entity's observable state to a JSON-safe dict."""
    from happysimulator.components.common import Counter, Sink
    from happysimulator.components.queued_resource import QueuedResource
    from happysimulator.load.source import Source

    try:
        from happysimulator.components.rate_limiter.rate_limited_entity import RateLimitedEntity
    except ImportError:
        RateLimitedEntity = type(None)

    try:
        from happysimulator.components.rate_limiter.inductor import Inductor
    except ImportError:
        Inductor = type(None)

    try:
        from happysimulator.instrumentation.collectors import LatencyTracker, ThroughputTracker
    except ImportError:
        LatencyTracker = type(None)
        ThroughputTracker = type(None)

    try:
        from happysimulator.components.resource import Resource
    except ImportError:
        Resource = type(None)

    if isinstance(entity, QueuedResource):
        result = {
            "depth": entity.depth,
            "accepted": entity.stats_accepted,
            "dropped": entity.stats_dropped,
        }
        # Merge subclass-specific attributes (e.g. _in_flight, service_time)
        for key, val in getattr(entity, "__dict__", {}).items():
            if key.startswith("__") or key in ("name", "_queue", "_driver", "_worker"):
                continue
            label = key.lstrip("_")
            if label not in result and isinstance(val, (int, float, str, bool)):
                result[label] = val
        return result
    if isinstance(entity, Sink):
        stats = entity.latency_stats()
        return {
            "events_received": entity.events_received,
            "avg_latency": round(stats["avg"], 6),
            "p99_latency": round(stats["p99"], 6),
        }
    if isinstance(entity, LatencyTracker):
        return {
            "count": entity.count,
            "mean_latency": round(entity.mean_latency(), 6),
            "p50": round(entity.p50(), 6),
            "p99": round(entity.p99(), 6),
        }
    if isinstance(entity, ThroughputTracker):
        return {"count": entity.count}
    if isinstance(entity, Counter):
        return {"total": entity.total, "by_type": dict(entity.by_type)}
    if isinstance(entity, RateLimitedEntity):
        return {
            "queue_depth": entity.queue_depth,
            "received": entity.stats.received,
            "forwarded": entity.stats.forwarded,
            "dropped": entity.stats.dropped,
        }
    if isinstance(entity, Inductor):
        return {
            "queue_depth": entity.queue_depth,
            "estimated_rate": round(entity.estimated_rate, 4),
            "received": entity.stats.received,
            "forwarded": entity.stats.forwarded,
            "dropped": entity.stats.dropped,
        }
    if isinstance(entity, Resource):
        return {
            "capacity": entity.capacity,
            "available": entity.available,
            "utilization": round(entity.utilization, 4),
            "waiters": entity.waiters,
        }
    try:
        from happysimulator.instrumentation.probe import Probe
    except ImportError:
        Probe = type(None)

    if isinstance(entity, Probe):
        count = len(entity.data_sink.values)
        last_val = entity.data_sink.values[-1][1] if count > 0 else None
        return {
            "metric": entity.metric,
            "target": entity.target.name,
            "samples": count,
            "latest": last_val,
        }
    if isinstance(entity, Source):
        ep = getattr(entity, "_event_provider", None)
        generated = getattr(ep, "_generated", None)
        return {"generated": generated or 0}

    # Fallback: inspect public primitive attributes
    return _fallback_serialize(entity)


def _fallback_serialize(entity: object) -> dict[str, Any]:
    """Inspect __dict__ for primitive-valued public attributes."""
    result: dict[str, Any] = {}
    for key, val in getattr(entity, "__dict__", {}).items():
        if key.startswith("_"):
            continue
        if isinstance(val, (int, float, str, bool)) or (
            isinstance(val, dict)
            and all(isinstance(k, str) for k in val)
            and all(isinstance(v, (int, float, str, bool)) for v in val.values())
        ):
            result[key] = val
    return result


def serialize_event(event: Event) -> dict[str, Any]:
    """Serialize an Event to a JSON-safe dict."""
    target_name = getattr(event.target, "name", type(event.target).__name__)
    return {
        "time_s": event.time.to_seconds(),
        "event_type": event.event_type,
        "target": target_name,
        "id": str(event._id),
        "daemon": event.daemon,
    }


_INTERNAL_EVENT_TYPES = frozenset(
    {
        "source_event",
        "SourceEvent",
        "QUEUE_POLL",
        "QUEUE_NOTIFY",
        "QUEUE_DELIVER",
        "probe_event",
    }
)


def is_internal_event(event_type: str) -> bool:
    """Check if an event type is an internal simulation mechanism."""
    if event_type in _INTERNAL_EVENT_TYPES:
        return True
    return bool(event_type.startswith(("inductor_poll::", "rate_limit_poll::")))
