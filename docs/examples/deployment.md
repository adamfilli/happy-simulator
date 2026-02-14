# Deployment Examples

7 examples covering deployment strategies and microservice patterns.

| Example | Description |
|---------|-------------|
| [canary_deployment.py](https://github.com/adamfilli/happy-simulator/blob/main/examples/deployment/canary_deployment.py) | Progressive traffic shift with monitoring |
| [rolling_deployment.py](https://github.com/adamfilli/happy-simulator/blob/main/examples/deployment/rolling_deployment.py) | Replacing backends with health checking |
| [saga_failure_cascade.py](https://github.com/adamfilli/happy-simulator/blob/main/examples/deployment/saga_failure_cascade.py) | Saga pattern failure cascade and compensation |
| [outbox_relay_lag.py](https://github.com/adamfilli/happy-simulator/blob/main/examples/deployment/outbox_relay_lag.py) | Outbox relay lag under load |
| [service_mesh_sidecar.py](https://github.com/adamfilli/happy-simulator/blob/main/examples/deployment/service_mesh_sidecar.py) | Service mesh sidecar proxy overhead |
| [idempotency_under_retries.py](https://github.com/adamfilli/happy-simulator/blob/main/examples/deployment/idempotency_under_retries.py) | Idempotency store under retry storms |
| [gc_pause_cascade.py](https://github.com/adamfilli/happy-simulator/blob/main/examples/deployment/gc_pause_cascade.py) | GC pause cascade: how GC strategy affects tail latency |

## Running

```bash
python examples/deployment/canary_deployment.py
```
