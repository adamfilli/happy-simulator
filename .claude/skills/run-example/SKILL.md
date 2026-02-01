---
name: run-example
description: Run a simulation example and analyze the output
disable-model-invocation: true
---

# Run Example

Run one of the simulation examples from the `examples/` directory and analyze the results.

## Available Examples

- `dual_path_queue_latency.py` - Dual-path queue latency analysis
- `increasing_queue_depth.py` - Queue depth behavior under increasing load
- `load_aware_routing.py` - Load-aware routing strategies
- `m_m_1_queue.py` - Classic M/M/1 queue simulation
- `metastable_state.py` - Metastable state demonstration
- `retrying_client.py` - Client retry behavior simulation

## Instructions

1. If no example is specified, list the available examples and ask the user to choose
2. Execute with: `.venv/Scripts/python.exe examples/<name>.py`
3. Capture and analyze the output
4. If plots are generated, describe what they show
5. Report on simulation behavior and any interesting observations
