## DEV

1. Improve type checker support for `@simulatable` decorator
   - Currently, static type checkers (Pylance, mypy) cannot see dynamically added attributes (`now`, `set_clock`)
   - Options to explore: `@dataclass_transform` (PEP 681), type checker plugins, or improved generic typing
   - See `happysimulator/core/decorators.py`

2. Event Lifecy cles (i.e. a source that creates an event with a predefined lifecycle)
3. Rate Limiter
    - Distributed convergence example
    - Burst suppression 
4. Queueing network recreations: https://www.grotto-networking.com/DiscreteEventPython.html
5. Model client server ping pong communications
6. Publish to PyPi
7. Remove scipy dependency
   - Only used in `arrival_time_provider.py` for `scipy.integrate`
   - Consider implementing the integration manually or using numpy alternatives

* Add Java ExecutorService pipeline simulation components and example
* Load balancer entity
* Retry explosion example 


## Content
0. Unintuitive - Unstable Queue when Arrival Rate = Service Rate
1. LIFO vs. FIFO queuing strategies
2. Active Queue Managemetn
3. Priority Queues
4. Simulation Assisted AI reasoning
5. Rate Limiting
    - Accuracy: temporal and volumetric

