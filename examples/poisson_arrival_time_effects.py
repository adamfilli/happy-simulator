from happysimulator import (
    ArrivalDistribution,
    ConstantLatency,
    Client,
    Queue,
    QueuedServer,
    Request,
    Generator,
    Measurement,
    ConstantProfile,
    Simulation,
    Stat,
    Time,
)


"""
This example demonstrates the counter intuitive result that arises from random arrival processes. 

Even though the server can handle 10 requests per second, if the arrival process is Poisson, the queue will grow indefinitely.

This is because in the business of load management, you can go into "debt", but you can never accumulate capital.

At constant rate and capacity, it's a D/D/1 stable queue.

At poisson rate and capacity, it's a M/D/1 queue (Poisson arrivals, deterministic service, one server), it's unstable and queue -> infinity.
"""

server_threads = 10
server_latency = ConstantLatency(Time.from_seconds(1))
# thus, server throughput is 10 requests per second

arrival_rate = 10
arrival_distribution = ArrivalDistribution.POISSON
# 10 = 10... so we should be able to support this load right?

client = Client(name="BasicClient")

queue = Queue(name="MyQueue") # unbounded FIFO queue
server = QueuedServer(name="MyQueuedServer",
                      server_latency=server_latency,
                      threads=server_threads,
                      queue=queue)

network_latency = ConstantLatency(Time.from_seconds(1))

profile = ConstantProfile(rate=arrival_rate)

request_generator = Generator(func=lambda time: [Request(time=time, client=client, server=server, callback=client.send_request, network_latency=network_latency)],
                              profile=profile,
                              distribution=arrival_distribution)

simulation_run_result = Simulation(
    end_time=Time.from_seconds(240),
    entities=[client, server],
    generators=[request_generator],
    measurements=[
        Measurement(name="Client Request Volume",
                    func=client.requests_count,
                    stats=[Stat.SUM],
                    interval=Time.from_seconds(1)),
        Measurement(name="Client Latency",
                    func=client.requests_latency,
                    stats=[Stat.AVG, Stat.P99, Stat.P0],
                    interval=Time.from_seconds(1)),
        Measurement(name="Server Latency",
                    func=server.requests_latency,
                    stats=[Stat.AVG, Stat.P99, Stat.P0],
                    interval=Time.from_seconds(1)),
        Measurement(name="Queue Depth",
                    func=queue.depth,
                    stats=[Stat.AVG, Stat.P99, Stat.P0],
                    interval=Time.from_seconds(1)),
        Measurement(name="Queue Time",
                    func=queue.queue_time,
                    stats=[Stat.AVG, Stat.P99, Stat.P0],
                    interval=Time.from_seconds(1))
    ]
).run()

simulation_run_result.print_csv()
simulation_run_result.display_graphs()
