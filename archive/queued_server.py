
"""
A server with a queue. There is an extra step before requests are started: the work must go into a queue
"""
import logging

from archive.data import Data
from happysimulator.entities.queue import Queue
from archive.server import Server
from happysimulator.events.event import Event
from archive.queue_event import QueueEvent
from happysimulator.math.latency_distribution import LatencyDistribution
from happysimulator.load.profile import Profile
from happysimulator.load import ConstantProfile
from archive.response_status import ResponseStatus

logger = logging.getLogger(__name__)

class QueuedServer(Server):
    def __init__(self,
                 name: str,
                 queue: Queue,
                 server_latency: LatencyDistribution,
                 failure_rate: Profile = ConstantProfile(rate=0.0),
                 threads: int = 1):
        super().__init__(name, server_latency, failure_rate)

        # config
        self._baseline_latency = server_latency
        self._queue = queue
        self._threads = threads

        # stats
        self._rejected_requests_count = Data()

    def start_request(self, request: Event) -> list[Event]:
        # instead of processing this request, put it in the queue.
        # If I currently have capacity, I will also schedule an instant queue Pop.
        logger.info(f"[{request.time.to_seconds()}][{self.name}][{request.name}] Queued Server received request. Server concurrency: {self._concurrent_requests}")

        # TODO: ideally, the queue itself would reject the request, not the queued server
        # this is a requirement to implement for sophisticated queuing policies, since the logic needs to exist
        # in the queue itself, and not in this class
        if not self._queue.has_capacity():
            self._rejected_requests_count.add_stat(1, request.time)
            logger.info(f"[{request.time.to_seconds()}][{self.name}][{request.name}] Server queue is full, will not queue request.")

            request.response_status = ResponseStatus.FAIL
            request.server_send_response_time = request.time
            request.callback = request.client.receive_response
            request.time = request.time + request.network_latency.get_latency(request.time)

            return [request]

        queued_event = request
        queued_event.callback = super().start_request # so that when we get this event back, we skip the queueing

        return [
            QueueEvent(name="QueuePut",
                       time=request.time,
                       callback=self._queue.put,
                       queued_event=queued_event,
                       immediate_pop=self._concurrent_requests < self._threads)
        ]


    def done_request(self, request: Event) -> list[Event]:
        logger.info(f"[{request.time.to_seconds()}][{self.name}][{request.name}] Queued Server completed request. Calling server.")

        events = super().done_request(request)

        if self._concurrent_requests < self._threads:
            # the server has capacity to start the next task
            events.append(Event(name="QueuePop", time=request.time, callback=self._queue.pop))

        return events