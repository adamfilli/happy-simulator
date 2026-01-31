# Simulation Definition Schema

This document defines the JSON schema for declaratively specifying simulations. The visual editor generates this format, and the Python runtime interprets it.

## Design Principles

- **Declarative structure**: Entities, connections, and sources are data
- **Python escape hatches**: Custom behavior via embedded Python or class references
- **Visual-friendly**: Includes `position` fields for editor state
- **Round-trippable**: 1:1 mapping between visual representation and JSON

---

## JSON Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "HappySimulator Simulation Definition",
  "type": "object",
  "properties": {

    "simulation": {
      "type": "object",
      "properties": {
        "name": { "type": "string" },
        "end_time": { "type": "number", "description": "Seconds" },
        "drain_time": { "type": "number", "description": "Extra time after sources stop" }
      },
      "required": ["name"]
    },

    "entities": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": { "type": "string", "description": "Unique identifier, used in connections" },
          "type": { "type": "string", "description": "Built-in type or custom class path" },
          "label": { "type": "string", "description": "Display name in visual editor" },
          "count": { "type": "integer", "minimum": 1, "default": 1, "description": "Number of instances (fleet)" },
          "params": { "type": "object", "description": "Type-specific parameters" },
          "position": { "$ref": "#/$defs/position" }
        },
        "required": ["id", "type"]
      }
    },

    "sources": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": { "type": "string" },
          "label": { "type": "string" },
          "target": { "type": "string", "description": "Entity ID to send events to" },
          "arrival": { "$ref": "#/$defs/arrival" },
          "profile": { "$ref": "#/$defs/profile" },
          "duration": { "type": "number", "description": "How long to generate (seconds)" },
          "event_type": { "type": "string", "default": "Request" },
          "position": { "$ref": "#/$defs/position" }
        },
        "required": ["id", "target", "arrival"]
      }
    },

    "connections": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "from": { "type": "string", "description": "Entity ID (with optional port: 'router.out')" },
          "to": {
            "oneOf": [
              { "type": "string" },
              { "type": "array", "items": { "type": "string" } }
            ]
          },
          "routing": {
            "type": "string",
            "enum": ["random", "round_robin", "least_loaded", "hash"],
            "default": "round_robin",
            "description": "How to distribute traffic when 'to' is a fleet (count > 1)"
          },
          "hash_key": {
            "type": "string",
            "description": "Context field to hash on when routing=hash (e.g., 'customer_id')"
          },
          "label": { "type": "string" }
        },
        "required": ["from", "to"]
      }
    },

    "probes": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": { "type": "string" },
          "target": { "type": "string", "description": "Entity ID to observe" },
          "metric": { "type": "string", "description": "Attribute/property name" },
          "interval": { "type": "number", "description": "Sample interval (seconds)" },
          "aggregation": {
            "type": "string",
            "enum": ["sum", "avg", "min", "max", "each"],
            "default": "sum",
            "description": "How to aggregate when target is a fleet. 'each' records per-instance."
          }
        },
        "required": ["id", "target", "metric", "interval"]
      }
    }
  },

  "$defs": {
    "position": {
      "type": "object",
      "properties": {
        "x": { "type": "number" },
        "y": { "type": "number" }
      },
      "description": "Visual editor coordinates"
    },

    "distribution": {
      "oneOf": [
        {
          "type": "object",
          "properties": {
            "type": { "const": "constant" },
            "value": { "type": "number" }
          },
          "required": ["type", "value"]
        },
        {
          "type": "object",
          "properties": {
            "type": { "const": "exponential" },
            "mean": { "type": "number" }
          },
          "required": ["type", "mean"]
        },
        {
          "type": "object",
          "properties": {
            "type": { "const": "uniform" },
            "min": { "type": "number" },
            "max": { "type": "number" }
          },
          "required": ["type", "min", "max"]
        },
        {
          "type": "object",
          "properties": {
            "type": { "const": "normal" },
            "mean": { "type": "number" },
            "std": { "type": "number" }
          },
          "required": ["type", "mean", "std"]
        }
      ]
    },

    "arrival": {
      "oneOf": [
        {
          "type": "object",
          "properties": {
            "type": { "const": "poisson" },
            "rate": { "type": "number", "description": "Events per second" }
          },
          "required": ["type", "rate"]
        },
        {
          "type": "object",
          "properties": {
            "type": { "const": "constant" },
            "interval": { "type": "number", "description": "Seconds between arrivals" }
          },
          "required": ["type", "interval"]
        }
      ]
    },

    "profile": {
      "oneOf": [
        {
          "type": "object",
          "properties": {
            "type": { "const": "constant" },
            "rate": { "type": "number" }
          },
          "required": ["type", "rate"]
        },
        {
          "type": "object",
          "properties": {
            "type": { "const": "ramp" },
            "start_rate": { "type": "number" },
            "end_rate": { "type": "number" },
            "duration": { "type": "number" }
          },
          "required": ["type", "start_rate", "end_rate", "duration"]
        },
        {
          "type": "object",
          "properties": {
            "type": { "const": "spike" },
            "baseline_rate": { "type": "number" },
            "spike_rate": { "type": "number" },
            "warmup": { "type": "number" },
            "spike_duration": { "type": "number" }
          },
          "required": ["type", "baseline_rate", "spike_rate", "warmup", "spike_duration"]
        }
      ]
    }
  }
}
```

---

## Built-in Entity Types

| Type | Description | Key Params |
|------|-------------|------------|
| `QueuedServer` | Queue + worker with service time | `service_time`, `concurrency`, `queue_capacity` |
| `RandomRouter` | Round-robin or random distribution | `strategy: "random" \| "round_robin"` |
| `Delay` | Adds latency to passing events | `delay` (distribution) |
| `Sink` | Terminal node, collects metrics | `track_latency` |
| `Filter` | Drops events based on condition | `drop_rate` or `condition` (Python) |
| `Fork` | Duplicates events to multiple outputs | - |
| `Join` | Waits for N events before emitting | `count` |

---

## Entity Fleets

When `count > 1`, an entity represents a **fleet** of identical instances. The visual editor displays a single node, but the runtime creates N instances.

### Instance Naming

At runtime, instances are created with indexed IDs:
- `servers` with `count: 1000` → `servers[0]`, `servers[1]`, ..., `servers[999]`

### Connections To Fleets

When connecting **to** a fleet, specify how traffic is distributed:

| Routing | Behavior |
|---------|----------|
| `round_robin` | Cycle through instances in order (default) |
| `random` | Uniform random selection |
| `least_loaded` | Pick instance with lowest queue depth / load |
| `hash` | Consistent hash on a context field (sticky sessions) |

```json
{
  "from": "load_balancer",
  "to": "servers",
  "routing": "least_loaded"
}
```

For hash-based routing (e.g., all requests from same customer go to same server):

```json
{
  "from": "load_balancer",
  "to": "servers",
  "routing": "hash",
  "hash_key": "customer_id"
}
```

### Connections From Fleets

When connecting **from** a fleet, all instances share the same downstream:

```json
{
  "from": "servers",
  "to": "sink"
}
```

This creates connections: `servers[0] → sink`, `servers[1] → sink`, etc.

### Probes on Fleets

When probing a fleet, specify how to aggregate across instances:

| Aggregation | Result |
|-------------|--------|
| `sum` | Total across all instances (default) |
| `avg` | Average across instances |
| `min` | Minimum value |
| `max` | Maximum value |
| `each` | Record each instance separately (creates N time series) |

```json
{
  "id": "total_queue_depth",
  "target": "servers",
  "metric": "depth",
  "interval": 0.1,
  "aggregation": "sum"
}
```

### Visual Representation

In the editor, fleets appear as a single node with a badge showing the count:

```
┌─────────────────┐
│  Servers        │
│  ×1000          │  ← Badge indicates fleet size
│  QueuedServer   │
└─────────────────┘
```

---

## Examples

### M/M/1 Queue

A single server with Poisson arrivals and exponential service time:

```json
{
  "simulation": {
    "name": "M/M/1 Queue Example",
    "end_time": 120,
    "drain_time": 10
  },

  "entities": [
    {
      "id": "server",
      "type": "QueuedServer",
      "label": "Server",
      "params": {
        "service_time": { "type": "exponential", "mean": 0.05 },
        "concurrency": 1
      },
      "position": { "x": 400, "y": 200 }
    },
    {
      "id": "sink",
      "type": "Sink",
      "label": "Completed",
      "params": {
        "track_latency": true
      },
      "position": { "x": 600, "y": 200 }
    }
  ],

  "sources": [
    {
      "id": "traffic",
      "label": "Traffic",
      "target": "server",
      "arrival": { "type": "poisson", "rate": 15 },
      "duration": 100,
      "position": { "x": 200, "y": 200 }
    }
  ],

  "connections": [
    { "from": "server", "to": "sink" }
  ],

  "probes": [
    {
      "id": "queue_depth",
      "target": "server",
      "metric": "depth",
      "interval": 0.1
    }
  ]
}
```

### Load-Balanced Servers

Traffic distributed across multiple servers:

```json
{
  "simulation": {
    "name": "Load Balanced Cluster",
    "end_time": 60
  },

  "entities": [
    {
      "id": "lb",
      "type": "RandomRouter",
      "label": "Load Balancer",
      "position": { "x": 300, "y": 200 }
    },
    {
      "id": "server1",
      "type": "QueuedServer",
      "label": "Server 1",
      "params": {
        "service_time": { "type": "exponential", "mean": 0.02 },
        "concurrency": 4
      },
      "position": { "x": 500, "y": 100 }
    },
    {
      "id": "server2",
      "type": "QueuedServer",
      "label": "Server 2",
      "params": {
        "service_time": { "type": "exponential", "mean": 0.02 },
        "concurrency": 4
      },
      "position": { "x": 500, "y": 300 }
    },
    {
      "id": "sink",
      "type": "Sink",
      "label": "Done",
      "position": { "x": 700, "y": 200 }
    }
  ],

  "sources": [
    {
      "id": "requests",
      "target": "lb",
      "arrival": { "type": "poisson", "rate": 100 },
      "profile": {
        "type": "spike",
        "baseline_rate": 50,
        "spike_rate": 200,
        "warmup": 10,
        "spike_duration": 5
      },
      "duration": 30,
      "position": { "x": 100, "y": 200 }
    }
  ],

  "connections": [
    { "from": "lb", "to": ["server1", "server2"] },
    { "from": "server1", "to": "sink" },
    { "from": "server2", "to": "sink" }
  ]
}
```

### Server Fleet (1000 Servers)

A large cluster represented as a single visual element:

```json
{
  "simulation": {
    "name": "Large Server Fleet",
    "end_time": 300
  },

  "entities": [
    {
      "id": "servers",
      "type": "QueuedServer",
      "label": "Server Fleet",
      "count": 1000,
      "params": {
        "service_time": { "type": "exponential", "mean": 0.01 },
        "concurrency": 8
      },
      "position": { "x": 400, "y": 200 }
    },
    {
      "id": "sink",
      "type": "Sink",
      "label": "Completed",
      "position": { "x": 600, "y": 200 }
    }
  ],

  "sources": [
    {
      "id": "traffic",
      "target": "servers",
      "arrival": { "type": "poisson", "rate": 50000 },
      "duration": 300,
      "position": { "x": 200, "y": 200 }
    }
  ],

  "connections": [
    {
      "from": "traffic",
      "to": "servers",
      "routing": "least_loaded"
    },
    { "from": "servers", "to": "sink" }
  ],

  "probes": [
    {
      "id": "total_depth",
      "target": "servers",
      "metric": "depth",
      "interval": 1.0,
      "aggregation": "sum"
    },
    {
      "id": "max_depth",
      "target": "servers",
      "metric": "depth",
      "interval": 1.0,
      "aggregation": "max"
    }
  ]
}
```

This creates 1000 server instances at runtime, but the visual editor shows just one node with a "×1000" badge.

---

## Custom Entity Types

### Embedded Python Handler

For one-off custom behavior, embed Python directly:

```json
{
  "id": "custom_processor",
  "type": "Custom",
  "params": {
    "handler": "def handle_event(self, event):\n    if event.context.get('priority') == 'high':\n        yield 0.01\n    else:\n        yield 0.05\n    return Event(target=self.downstream, context=event.context)"
  }
}
```

### External Python Class

Reference a Python class by module path:

```json
{
  "id": "my_server",
  "type": "mymodule.MyCustomServer",
  "params": { "custom_param": 42 }
}
```

The runtime will import and instantiate the class, passing `params` to the constructor.

---

## Visual Editor Mapping

| Schema Concept | Visual Element |
|----------------|----------------|
| Entity | Draggable node (box) |
| Entity with `count > 1` | Node with count badge (e.g., "×1000") |
| Source | Node with distinct style (e.g., green, play icon) |
| Sink | Node with distinct style (e.g., red, stop icon) |
| Connection | Arrow/edge between nodes |
| Connection with `routing` | Arrow with routing indicator (e.g., "LB" icon) |
| Probe | Badge/indicator attached to entity |
| `position` | Canvas coordinates of node |
| `params` | Properties panel when node is selected |
| `label` | Text displayed on node |
