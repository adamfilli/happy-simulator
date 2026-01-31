## BUGS

1. **Simulation processes events past end_time**
   - Location: `happysimulator/core/simulation.py:150`
   - The loop condition `while self._event_heap.has_events() and self._end_time >= current_time` checks `current_time` BEFORE popping the next event
   - This means if current_time is 0.011s and end_time is 1.0s, the loop continues and pops an event at 60.0s
   - The event at 60.0s is processed even though it's past end_time
   - **Fix**: Check the next event's time before processing: peek at the heap and break if event.time > end_time

## SIMULATIONS TO BUILD

1. **Cold Start Simulation**
   - Model application cold start behavior (JVM warmup, cache population, connection pool initialization)
   - Show how systems behave during startup vs steady-state
   - Demonstrate thundering herd problems when multiple instances restart
   - Visualize cache hit rate progression over time

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
1. LIFO vs. FIFO queuing strategies
2. Active Queue Managemetn
3. Priority Queues
4. Simulation Assisted AI reasoning
5. Rate Limiting
    - Accuracy: temporal and volumetric


## Other Sim work

SimpPy
https://github.com/MaineKuehn/usim
https://github.com/djordon/queueing-tool
https://www.simio.com/
https://github.com/salabim/salabim
https://www.simio.com/
https://github.com/CiwPython/Ciw, https://ciw.readthedocs.io/en/latest/
https://ki-oss.github.io/hades/
