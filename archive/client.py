import logging
from typing import Optional

from archive.data import Data
from happysimulator.entities.entity import Entity
from happysimulator.events.event import Event
from archive.measurement_event import MeasurementEvent
from happysimulator.utils.instant import Instant
from happysimulator.utils.response_status import ResponseStatus

logger = logging.getLogger(__name__)

class Client(Entity):
    def __init__(self, name: str, timeout: Optional[Instant] = None, retries: int = 0, retry_delay: Instant = Instant.from_seconds(0)):
        super().__init__(name)

        # config
        self._timeout = timeout
        self._retries = retries
        self._retry_delay = retry_delay

        # stats
        self._requests_count = Data()
        self._successful_requests_count = Data()
        self._failed_requests_count = Data()
        self._timeout_requests_count = Data()
        self._requests_latency = Data()
        self._failed_requests_latency = Data()
        self._successful_requests_latency = Data()
        self._responses = Data()

    def send_request(self, request: Event) -> list[Event]:
        logger.info(f"[{request.time.to_seconds()}][{self.name}][{request.name}] Client sending request")
        self._requests_count.add_stat(1, request.time)

        request.client_send_request_time = request.time
        request.callback = request.server.start_request
        request.time = request.time + request.network_latency.get_latency(request.time)

        return [request]

    def receive_response(self, request: Event) -> list[Event]:
        latency = (request.time - request.client_send_request_time).to_seconds()

        logger.info(f"[{request.time.to_seconds()}][{self.name}][{request.name}] Client received response after {latency} seconds")
        self._responses.add_stat(1, request.time)

        self._requests_latency.add_stat(latency, request.time)

        request_failed = request.response_status == ResponseStatus.FAIL

        if self._timeout is not None:
            timed_out = latency > self._timeout.to_seconds()
            if timed_out:
                self._timeout_requests_count.add_stat(1, request.time)

            request_failed |= timed_out

        if request_failed:
            self._failed_requests_count.add_stat(1, request.time)
            self._failed_requests_latency.add_stat(latency, request.time)
        else:
            self._successful_requests_count.add_stat(1, request.time)
            self._successful_requests_latency.add_stat(latency, request.time)
            return []

        # request_failed == True
        if self._retries > 0 and request.attempt <= self._retries:
            logger.info(f"[{request.time.to_seconds()}][{self.name}][{request.name}] Client will retry request. Attempt:{request.attempt}. Retries: {self._retries}")
            request.callback = self.send_request
            request.time = request.time + self._retry_delay.to_seconds()
            request.attempt += 1
            return [request]

        return [] # no further action needed for this request

    def requests_stats(self, event: MeasurementEvent)  -> list[Event]:
        logger.info(f"[{event.time.to_seconds()}][{self.name}][{event.name}] Received measurement event for requests_stats.")

        self.sink_data(self._requests_count, event) # TODO implement all of the above and below logic

        return [] # measurement events don't generate additional events

    def requests_latency(self, event: MeasurementEvent) -> list[Event]:
        logger.info(f"[{event.time.to_seconds()}][{self.name}][{event.name}] Received measurement event for request latency")

        self.sink_data(self._requests_latency, event)

    def successful_requests_latency(self, event: MeasurementEvent) -> list[Event]:
        logger.info(f"[{event.time.to_seconds()}][{self.name}][{event.name}] Received measurement event for successful request latency")

        self.sink_data(self._successful_requests_latency, event)

    def failed_requests_latency(self, event: MeasurementEvent) -> list[Event]:
        logger.info(f"[{event.time.to_seconds()}][{self.name}][{event.name}] Received measurement event for failed request latency")

        self.sink_data(self._failed_requests_latency, event)

    def requests_count(self, event: MeasurementEvent) -> list[Event]:
        logger.info(f"[{event.time.to_seconds()}][{self.name}][{event.name}] Received measurement event for request count")

        self.sink_data(self._requests_count, event)

    def successful_requests_count(self, event: MeasurementEvent) -> list[Event]:
        logger.info(f"[{event.time.to_seconds()}][{self.name}][{event.name}] Received measurement event for successful request count")

        self.sink_data(self._successful_requests_count, event)

    def failed_requests_count(self, event: MeasurementEvent) -> list[Event]:
        logger.info(f"[{event.time.to_seconds()}][{self.name}][{event.name}] Received measurement event for failed request count")

        self.sink_data(self._failed_requests_count, event)
        
    def timeout_requests_count(self, event: MeasurementEvent) -> list[Event]:
        logger.info(f"[{event.time.to_seconds()}][{self.name}][{event.name}] Received measurement event for timeout request count")

        self.sink_data(self._timeout_requests_count, event)
