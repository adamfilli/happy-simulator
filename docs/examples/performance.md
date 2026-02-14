# Performance Examples

7 examples exploring performance patterns and auto-scaling.

| Example | Description |
|---------|-------------|
| [auto_scaler.py](https://github.com/adamfilli/happy-simulator/blob/main/examples/performance/auto_scaler.py) | Traffic spike → scaling lag → stabilization |
| [api_gateway_bottleneck.py](https://github.com/adamfilli/happy-simulator/blob/main/examples/performance/api_gateway_bottleneck.py) | API gateway bottleneck analysis |
| [cold_start.py](https://github.com/adamfilli/happy-simulator/blob/main/examples/performance/cold_start.py) | Cache behavior during warmup and reset |
| [work_stealing_pool.py](https://github.com/adamfilli/happy-simulator/blob/main/examples/performance/work_stealing_pool.py) | Work-stealing pool tail latency under skewed workload |
| [zipf_cache_cohorts.py](https://github.com/adamfilli/happy-simulator/blob/main/examples/performance/zipf_cache_cohorts.py) | Zipf-distributed traffic and cache hit rates by cohort |
| [metric_collection_pipeline.py](https://github.com/adamfilli/happy-simulator/blob/main/examples/performance/metric_collection_pipeline.py) | Metric collection pipeline analysis |
| [ai_analysis.py](https://github.com/adamfilli/happy-simulator/blob/main/examples/performance/ai_analysis.py) | AI-powered simulation analysis workflow |

## Running

```bash
python examples/performance/auto_scaler.py
```
