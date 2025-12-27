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
    SpikeProfile,
    Simulation,
    Stat,
    Time,
)

SIMULATION_DURATION_SECONDS = 60
MEASUREMENT_PERIOD_SECONDS = 1

client = Client(name="Basic", retries=3, retry_delay=Time.from_seconds(0), timeout=Time.from_seconds(1.0))

queue = Queue(name="MyQueue", size=0) # adjust size to see how it impacts congestive collapse

failure_profile = SpikeProfile(rampup_start=Time.from_seconds(10),
                               rampdown_start=Time.from_seconds(15),
                               starting_rate=0.0,
                               rampup_factor=10,
                               rampdown_factor=10)

# Create a server with exponentially distributed service time
server = QueuedServer(name="Expo",
                server_latency=ConstantLatency(Time.from_seconds(0.1)),
                failure_rate=failure_profile,
                queue=queue)

network_latency = ConstantLatency(Time.from_seconds(0.0))

request_generator = Generator(func=lambda time: [Request(time=time, client=client, server=server, callback=client.send_request, network_latency=network_latency)],
                              profile=ConstantProfile(rate=10),
                              distribution=ArrivalDistribution.CONSTANT)

measurements = [
        Measurement(name="Client Request Count",
                    func=client.requests_count,
                    stats=[Stat.SUM],
                    interval=Time.from_seconds(MEASUREMENT_PERIOD_SECONDS)),
        Measurement(name="Client Failed Request Count",
                    func=client.failed_requests_count,
                    stats=[Stat.SUM],
                    interval=Time.from_seconds(MEASUREMENT_PERIOD_SECONDS)),
        Measurement(name="Client Successful Request Count",
                    func=client.successful_requests_count,
                    stats=[Stat.SUM],
                    interval=Time.from_seconds(MEASUREMENT_PERIOD_SECONDS)),
        Measurement(name="Client Latency",
                    func=client.requests_latency,
                    stats=[Stat.AVG],
                    interval=Time.from_seconds(MEASUREMENT_PERIOD_SECONDS)),
        Measurement(name="Queue Depth",
                    func=queue.depth,
                    stats=[Stat.AVG, Stat.P99, Stat.P0],
                    interval=Time.from_seconds(1))
    ]


result = Simulation(
    end_time=Time.from_seconds(SIMULATION_DURATION_SECONDS),
    generators=[request_generator],
    measurements=measurements
).run()

result.display_graphs()
result.print_csv()
