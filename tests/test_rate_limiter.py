from happysimulator.entities.rate_limiter import RateLimiter
from happysimulator.events.client_server_request_event import Request
from happysimulator.entities.client import Client
from happysimulator.entities.server import Server
from happysimulator.time import Time
from happysimulator.distribution.constant_latency import ConstantLatency


def test_immediate_forwarding():
    rl = RateLimiter("rl", capacity=1, refill_rate=1, initial_tokens=1)
    client = Client("c")
    server = Server("s", server_latency=ConstantLatency(Time.from_seconds(0)))

    req = Request(time=Time.from_seconds(0), client=client, server=server, callback=rl.start_request)

    evs = rl.start_request(req)
    assert len(evs) == 1
    ev = evs[0]
    # Should be forwarded immediately to the server
    assert ev.callback == server.start_request
    assert ev.time == Time.from_seconds(0)


def test_delayed_forwarding():
    rl = RateLimiter("rl", capacity=1, refill_rate=1, initial_tokens=0)
    client = Client("c")
    server = Server("s", server_latency=ConstantLatency(Time.from_seconds(0)))

    req = Request(time=Time.from_seconds(0), client=client, server=server, callback=rl.start_request)

    evs = rl.start_request(req)
    assert len(evs) == 1
    ev = evs[0]

    # No tokens initially -> should be delayed by 1 second
    assert ev.callback == rl.forward_request
    assert ev.time == Time.from_seconds(1)

    # When forwarding happens at t=1, the request should be passed to the server
    evs2 = rl.forward_request(ev)
    assert len(evs2) == 1
    ev2 = evs2[0]
    assert ev2.callback == server.start_request
    assert ev2.time == Time.from_seconds(1)


def test_refill_and_multiple_requests():
    rl = RateLimiter("rl", capacity=1, refill_rate=1, initial_tokens=1)
    client = Client("c")
    server = Server("s", server_latency=ConstantLatency(Time.from_seconds(0)))

    # First request at t=0 consumes the only token
    req1 = Request(time=Time.from_seconds(0), client=client, server=server, callback=rl.start_request)
    evs1 = rl.start_request(req1)
    assert evs1[0].callback == server.start_request

    # Second request at t=0.5 should be delayed until t=1.0
    req2 = Request(time=Time.from_seconds(0.5), client=client, server=server, callback=rl.start_request)
    evs2 = rl.start_request(req2)
    ev2 = evs2[0]
    assert ev2.callback == rl.forward_request
    assert ev2.time == Time.from_seconds(1)

    # Forward the delayed one at t=1.0
    evs_forwarded = rl.forward_request(ev2)
    assert evs_forwarded[0].callback == server.start_request

    # Third request at t=2.0 should be immediate (tokens refilled)
    req3 = Request(time=Time.from_seconds(2.0), client=client, server=server, callback=rl.start_request)
    evs3 = rl.start_request(req3)
    assert evs3[0].callback == server.start_request
    assert evs3[0].time == Time.from_seconds(2.0)
