"""Request event modeling client-server communication with latency tracking.

The Request event flows through the system accumulating timing information
at each stage, enabling detailed latency breakdown analysis after simulation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from happysimulator.events.event import Event
from happysimulator.utils.instant import Instant

from happysimulator.math.latency_distribution import LatencyDistribution


class ResponseStatus(Enum):
    """Outcome of a completed request.

    Used by clients to distinguish successful responses from failures,
    enabling retry logic and success rate metrics.
    """
    SUCCESS = "SUCCESS"
    FAIL = "FAIL"
    REJECTED = "REJECTED"


@dataclass
class Request(Event):
    """A request flowing between client and server with full timing instrumentation.

    Tracks four timestamps that together enable latency breakdown:
    - client_send_time: When the client initiated the request
    - server_receive_time: When the server began processing
    - server_send_time: When the server sent the response
    - client_receive_time: When the client received the response

    From these, you can compute:
    - Network latency (one-way): server_receive_time - client_send_time
    - Server processing time: server_send_time - server_receive_time
    - Round-trip time: client_receive_time - client_send_time

    The request carries references to both endpoints so it can route itself
    without the simulation needing explicit knowledge of the topology.

    Attributes:
        client: The SimpleClient that originated this request.
        server: The SimpleServer processing this request.
        network_latency: Distribution applied to both send and response legs.
        attempt: Current retry attempt number (starts at 1).
    """
    # References to endpoints
    client: "SimpleClient" = field(default=None, repr=False)
    server: "SimpleServer" = field(default=None, repr=False)

    # Network latency distribution (applied on send and response)
    network_latency: "LatencyDistribution" = field(default=None, repr=False)

    # Timing fields (populated during flow)
    client_send_time: Optional[Instant] = field(default=None, repr=False)
    server_receive_time: Optional[Instant] = field(default=None, repr=False)
    server_send_time: Optional[Instant] = field(default=None, repr=False)
    client_receive_time: Optional[Instant] = field(default=None, repr=False)

    # Response state
    response_status: Optional[ResponseStatus] = field(default=None, repr=False)

    # Retry tracking
    attempt: int = field(default=1, repr=False)
