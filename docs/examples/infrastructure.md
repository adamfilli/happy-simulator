# Infrastructure Examples

7 examples modeling low-level system infrastructure.

| Example | Description |
|---------|-------------|
| [cpu_scheduling.py](https://github.com/adamfilli/happy-simulator/blob/main/examples/infrastructure/cpu_scheduling.py) | FairShare vs PriorityPreemptive under mixed workloads |
| [disk_io_contention.py](https://github.com/adamfilli/happy-simulator/blob/main/examples/infrastructure/disk_io_contention.py) | HDD vs SSD vs NVMe under concurrent load |
| [page_cache_eviction.py](https://github.com/adamfilli/happy-simulator/blob/main/examples/infrastructure/page_cache_eviction.py) | Page cache hit rate under different workload patterns |
| [event_log.py](https://github.com/adamfilli/happy-simulator/blob/main/examples/infrastructure/event_log.py) | Event log with retention and consumer groups |
| [consumer_group.py](https://github.com/adamfilli/happy-simulator/blob/main/examples/infrastructure/consumer_group.py) | Consumer group partition assignment strategies |
| [stream_processor.py](https://github.com/adamfilli/happy-simulator/blob/main/examples/infrastructure/stream_processor.py) | Windowed aggregation with late event handling |
| [job_scheduler_dag.py](https://github.com/adamfilli/happy-simulator/blob/main/examples/infrastructure/job_scheduler_dag.py) | DAG-based job scheduler: Extract → Transform → Load |

## Running

```bash
python examples/infrastructure/cpu_scheduling.py
```
