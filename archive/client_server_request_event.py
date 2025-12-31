from happysimulator.math.constant_latency import ConstantLatency
from happysimulator.entities.client import Client
from happysimulator.entities.server import Server
from happysimulator.events.event import Event
from happysimulator.math.latency_distribution import LatencyDistribution
from happysimulator.utils.instant import Instant
from happysimulator.utils.ids import get_id

class Request(Event):
    def __init__(self, time: Instant, client: Client, server: Server, callback, name: str = None, network_latency: LatencyDistribution = ConstantLatency(Instant.from_seconds(0.1))):
        if name is None:
            name = f"Request-{get_id()}"

        super().__init__(time, name, callback)
        self.client = client
        self.server = server
        self.client_send_request_time = None
        self.server_receive_request_time = None
        self.server_send_response_time = None
        self.client_receive_response_time = None
        self.network_latency = network_latency
        self.response = None
        self.attempt = 1
        self.response_status = None