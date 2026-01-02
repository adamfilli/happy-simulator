import logging
import random
from typing import Callable, Generator

from archive.data import Data
from archive.constant_latency import ConstantLatency
from happysimulator.entities.entity import Entity
from happysimulator.events.event import Event
from archive.client_server_request_event import Request
from archive.measurement_event import MeasurementEvent
from archive.latency_distribution import LatencyDistribution
from happysimulator.load.profile import Profile
from happysimulator.load import ConstantProfile
from happysimulator.utils.instant import Instant
from archive.response_status import ResponseStatus

logger = logging.getLogger(__name__)

class Server(Entity):
    def __init__(self,
                 name: str,
                 server_latency: LatencyDistribution = ConstantLatency(Instant.from_seconds(0.1)),
                 failure_rate: Profile = ConstantProfile(rate=0.0),
                 concurrency_penalty_func: Callable[[int], float] = lambda x: 0.0):
        super().__init__(name)

        # config
        self._failure_rate = failure_rate
        self._concurrency_penalty_func = concurrency_penalty_func

        # stats
        self._latency = server_latency
        self._requests_count = Data()
        self._successful_requests_count = Data()
        self._failed_requests_count = Data()
        self._requests_finished_count = Data()
        self._responses_count = Data()
        self._server_side_latency = Data()
        self._concurrent_requests = 0
    
    def handle_event(self, event: Event) -> Generator[float, None, list[Event]]:
        """
        Entry point. Now delegates to the generator logic.
        """
        if isinstance(event, Request):
            # We return the generator object created by calling the method
            return self._process_request_flow(event)
        return []
    
    def _process_request_flow(self, request: Request):
        """
        A single linear flow that spans simulation time.
        """
        logger.info(f"[{request.time.to_seconds()}][{self.name}] Server received request")
        
        # 1. Update Start Stats
        self._requests_count.add_stat(1, request.time)
        self._concurrent_requests += 1
        request.server_receive_request_time = request.time

        # 2. Calculate Latency (The "Blocking" Part)
        # We calculate how long this operation takes
        processing_latency = self._latency.get_latency(request.time) + \
                             self._concurrency_penalty_func(self._concurrent_requests)
        
        # 3. YIELD CONTROL
        # We yield the latency (Time or float). The simulator will suspend us here.
        yield processing_latency

        # 4. Determine Success/Fail
        request.response_status = (ResponseStatus.FAIL
                                   if random.random() < self._failure_rate.get_rate(request.time)
                                   else ResponseStatus.SUCCESS)

        # 5. Update End Stats
        self._requests_finished_count.add_stat(1, request.time)
        self._concurrent_requests -= 1
        
        logger.info(f"[{request.time.to_seconds()}][{self.name}] Completed status {request.response_status.value}")

        request.server_send_response_time = request.time
        self._server_side_latency.add_stat(
            (request.time - request.server_receive_request_time).to_seconds(),
            request.time
        )

        # 6. Send Response (Return the next event to occur)
        # We calculate network latency and send it back to the client
        request.callback = request.client.receive_response
        request.time = request.time + request.network_latency.get_latency(request.time)
            
        return [request]


    def concurrency_stats(self, event: Event) -> list[Event]:
        logger.debug(f"[{event.time.to_seconds()}][{self.name}][{event.name}] Server measurement event for concurrency_stats.")
        self.sink_data(self._concurrent_requests, event)
        return []

    def requests_latency(self, event: Event) -> list[Event]:
        logger.debug(f"[{event.time.to_seconds()}][{self.name}][{event.name}] Received measurement event for request latency")

        self.sink_data(self._server_side_latency, event)

    def requests_count(self, event: Event) -> list[Event]:
        logger.debug(f"[{event.time.to_seconds()}][{self.name}][{event.name}] Received measurement event for request count")

        self.sink_data(self._requests_count, event)

    def successful_requests_count(self, event: Event) -> list[Event]:
        logger.debug(f"[{event.time.to_seconds()}][{self.name}][{event.name}] Received measurement event for successful request count")

        self.sink_data(self._successful_requests_count, event)

    def failed_requests_count(self, event: Event) -> list[Event]:
        logger.debug(f"[{event.time.to_seconds()}][{self.name}][{event.name}] Received measurement event for failed request count")

        self.sink_data(self._failed_requests_count, event)