from happysimulator import (
    ArrivalDistribution,
    ExponentialLatency,
    Client,
    Server,
    Request,
    Generator,
    Measurement,
    SinusoidProfile,
    Simulation,
    Stat,
    Time,
)
from happysimulator.entities.rate_limiter import RateLimiter

SIMULATION_DURATION_SECONDS = 120
MEASUREMENT_PERIOD_SECONDS = 1

# Create a basic client
client = Client(name="Basic")

# Create a backend server with exponentially distributed service time
backend = Server(name="Backend", server_latency=ExponentialLatency(Time.from_seconds(0.5)))

# Create a rate limiter between client and server
# capacity=1 token, refill_rate=1 token/sec -> at most 1 req/sec sustained
rate_limiter = RateLimiter(name="RL", capacity=1, refill_rate=1, initial_tokens=1)

# define our network latency
network_latency = ExponentialLatency(Time.from_seconds(0.5))

# create a generator profile which brings the simulation to life by telling the client to make requests to the server
# requests will be sent to the client which then sends them to the rate limiter; the rate limiter will forward to backend
request_generator = Generator(func=lambda time: [Request(time=time, client=client, server=backend, callback=client.send_request, network_latency=network_latency)],
                              profile=SinusoidProfile(shift=10, amplitude=5, period=Time.from_seconds(30)),
                              distribution=ArrivalDistribution.POISSON)

# define what we want to measure in our simulation, and at what time-resolution
measurements = [
        Measurement(name="Client Request Count", func=client.requests_count, stats=[Stat.SUM], interval=Time.from_seconds(MEASUREMENT_PERIOD_SECONDS)),
        Measurement(name="Backend Request Count", func=backend.requests_count, stats=[Stat.SUM], interval=Time.from_seconds(MEASUREMENT_PERIOD_SECONDS)),
        Measurement(name="Client Request Latency", func=client.requests_latency, stats=[Stat.AVG, Stat.P99, Stat.P0], interval=Time.from_seconds(MEASUREMENT_PERIOD_SECONDS)),
        Measurement(name="RateLimiter Queued Requests", func=rate_limiter.queued_requests_count, stats=[Stat.SUM], interval=Time.from_seconds(MEASUREMENT_PERIOD_SECONDS)),
        Measurement(name="RateLimiter Available Tokens", func=rate_limiter.available_tokens, stats=[Stat.AVG, Stat.P0], interval=Time.from_seconds(MEASUREMENT_PERIOD_SECONDS)),
    ]

result = Simulation(
    end_time=Time.from_seconds(SIMULATION_DURATION_SECONDS),
    generators=[request_generator],
    measurements=measurements
).run()

result.display_graphs()
result.print_csv()
