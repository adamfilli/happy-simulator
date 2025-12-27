from happysimulator import (
    ArrivalDistribution,
    ConstantLatency,
    Client,
    Queue,
    QueuedServer,
    Request,
    Generator,
    Measurement,
    RampupProfile,
    Simulation,
    Stat,
    Time,
)

"""
This is a basic load shedding example.

The Queue size on the server is limited, so it will load shed requests when the queue is full. 

If Queue_size is set to unbounded (0), all requests get served eventually, but the client timeout request volume increases.

Change the queue size and the client timeout parameters to see how the system behaves

The traffic profile is a TPS rampup.
"""

queue_size = 10
# queue_size = 0 # unbounded

client_timeout = Time.from_seconds(5)

client = Client(name="BasicClient", timeout=client_timeout)

queue = Queue(name="MyQueue", size=queue_size)
server = QueuedServer(name="MyQueuedServer",
                      server_latency=ConstantLatency(Time.from_seconds(1)),
                      threads=10,
                      queue=queue)

network_latency = ConstantLatency(Time.from_seconds(1))

profile = RampupProfile(starting_rate=1, rampup_factor=0.25)

request_generator = Generator(func=lambda time: [Request(time=time, client=client, server=server, callback=client.send_request, network_latency=network_latency)],
                              profile=profile,
                              distribution=ArrivalDistribution.CONSTANT)

simulation_run_result = Simulation(
    end_time=Time.from_seconds(120),
    entities=[client, server],
    generators=[request_generator],
    measurements=[
        Measurement(name="Client Request Volume",
                    func=client.requests_count,
                    stats=[Stat.SUM],
                    interval=Time.from_seconds(2)),
        Measurement(name="Client Successful Request Volume",
                    func=client.successful_requests_count,
                    stats=[Stat.SUM],
                    interval=Time.from_seconds(2)),
        Measurement(name="Client Failed Request Volume",
                    func=client.failed_requests_count,
                    stats=[Stat.SUM],
                    interval=Time.from_seconds(2)),
        Measurement(name="Client Timed Out Request Volume",
                    func=client.timeout_requests_count,
                    stats=[Stat.SUM],
                    interval=Time.from_seconds(2)),
        Measurement(name="Client Latency",
                    func=client.requests_latency,
                    stats=[Stat.AVG],
                    interval=Time.from_seconds(2)),
        Measurement(name="Client Successful Request Latency",
                    func=client.successful_requests_latency,
                    stats=[Stat.AVG],
                    interval=Time.from_seconds(2)),
        Measurement(name="Client Failed Request Latency",
                    func=client.failed_requests_latency,
                    stats=[Stat.AVG],
                    interval=Time.from_seconds(2)),
        Measurement(name="Queue Depth",
                    func=queue.depth,
                    stats=[Stat.AVG, Stat.P99, Stat.P0],
                    interval=Time.from_seconds(2)),
        Measurement(name="Server concurrency",
                    func=server.concurrency_stats,
                    stats=[Stat.AVG],
                    interval=Time.from_seconds(2)),
    ]
).run()

simulation_run_result.print_csv()
simulation_run_result.display_graphs()
