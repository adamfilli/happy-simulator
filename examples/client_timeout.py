from happysimulator import (
    ArrivalDistribution,
    ExponentialLatency,
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

SIMULATION_DURATION_SECONDS = 120
MEASUREMENT_PERIOD_SECONDS = 1

# client with a 5 second timeout
client = Client(name="Basic", timeout=Time.from_seconds(5))

# Create a server with exponentially distributed service time
queue = Queue(name="MyQueue") # unbounded FIFO queue
server = QueuedServer(name="Expo", server_latency=ExponentialLatency(Time.from_seconds(0.5)), queue=queue, threads=10)

# define our network latency, in this case equal to our server latency, and also exponentially distributed
network_latency = ExponentialLatency(Time.from_seconds(0.5))

profile = RampupProfile(starting_rate=10, rampup_factor=0.5)

# create a generator profile which brings the simulation to life by telling the client to make requests to the server
# in this case, with a varying rated defined by a sinusoid, and exponentially distributed requests
request_generator = Generator(func=lambda time: [Request(time=time, client=client, server=server, callback=client.send_request, network_latency=network_latency)],
                              profile=profile,
                              distribution=ArrivalDistribution.POISSON)

# define what we want to measure in our simulation, and at what time-resolution
measurements = [
        Measurement(name="Client Request Count",
                    func=client.requests_count,
                    stats=[Stat.SUM],
                    interval=Time.from_seconds(MEASUREMENT_PERIOD_SECONDS)),
        Measurement(name="Client Request Latency",
                    func=client.requests_latency,
                    stats=[Stat.AVG, Stat.P99, Stat.P0],
                    interval=Time.from_seconds(MEASUREMENT_PERIOD_SECONDS)),
        Measurement(name="Client Successful Request Count",
                    func=client.successful_requests_count,
                    stats=[Stat.SUM],
                    interval=Time.from_seconds(MEASUREMENT_PERIOD_SECONDS)),
        Measurement(name="Client Failed Request Count",
                    func=client.failed_requests_count,
                    stats=[Stat.SUM],
                    interval=Time.from_seconds(MEASUREMENT_PERIOD_SECONDS))
    ]


result = Simulation(
    end_time=Time.from_seconds(SIMULATION_DURATION_SECONDS),
    generators=[request_generator],
    measurements=measurements
).run()

result.display_graphs()
result.print_csv()
#result.save_csvs(directory="/tmp/")