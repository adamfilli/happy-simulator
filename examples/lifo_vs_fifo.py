from happysimulator.arrival_distribution import ArrivalDistribution
from happysimulator.distribution.constant_latency import ConstantLatency
from happysimulator.entities.client import Client
from happysimulator.entities.lifo_queue import LifoQueue
from happysimulator.entities.queue import Queue
from happysimulator.entities.queued_server import QueuedServer
from happysimulator.events.client_server_request_event import Request
from happysimulator.generator import Generator
from happysimulator.measurement import Measurement
from happysimulator.profiles.constant_profile import ConstantProfile
from happysimulator.simulation import Simulation
from happysimulator.stat import Stat
from happysimulator.time import Time

"""
Demonstrates a well known property of FIFO vs LIFO queues, FIFO queues distribute the queue penalty to all requests equally,
wheras LIFO queues concentrate the penalty on a few requests.

https://dzone.com/articles/fifo-vs-lifo-queueing-improving-service-availabili
"""

server_threads = 10
server_latency = ConstantLatency(Time.from_seconds(1))
# thus, server throughput is 10 requests per second

arrival_rate = 10
arrival_distribution = ArrivalDistribution.POISSON
# 10 = 10... so we should be able to support this load right?

network_latency = ConstantLatency(Time.from_seconds(1))
profile = ConstantProfile(rate=arrival_rate)

fifo_client = Client(name="FIFOClient")
fifo_queue = Queue(name="FIFOQueue") # unbounded FIFO queue
fifo_server = QueuedServer(name="FifoServer",
                      server_latency=server_latency,
                      threads=server_threads,
                      queue=fifo_queue)
fifo_request_generator = Generator(func=lambda time: [Request(time=time, client=fifo_client, server=fifo_server, callback=fifo_client.send_request, network_latency=network_latency)],
                              profile=profile,
                              distribution=arrival_distribution)


lifo_client = Client(name="LIFOClient")
lifo_queue = LifoQueue(name="LIFOQueue") # unbounded LIFO queue
lifo_server = QueuedServer(name="LifoServer",
                      server_latency=server_latency,
                      threads=server_threads,
                      queue=lifo_queue)
lifo_request_generator = Generator(func=lambda time: [Request(time=time, client=lifo_client, server=lifo_server, callback=lifo_client.send_request, network_latency=network_latency)],
                              profile=profile,
                              distribution=arrival_distribution)

simulation_run_result = Simulation(
    end_time=Time.from_seconds(60),
    entities=[fifo_client, fifo_server, lifo_client, lifo_server],
    generators=[fifo_request_generator, lifo_request_generator],
    measurements=[
        Measurement(name="Fifo Client Latency",
                    func=fifo_client.requests_latency,
                    stats=[Stat.AVG, Stat.P99, Stat.P0],
                    interval=Time.from_seconds(2)),
        Measurement(name="Fifo Queue Depth",
                    func=fifo_queue.depth,
                    stats=[Stat.AVG, Stat.P99, Stat.P0],
                    interval=Time.from_seconds(2)),
        Measurement(name="Lifo Client Latency",
                    func=lifo_client.requests_latency,
                    stats=[Stat.AVG, Stat.P99, Stat.P0],
                    interval=Time.from_seconds(2)),
        Measurement(name="Lifo Queue Depth",
                    func=lifo_queue.depth,
                    stats=[Stat.AVG, Stat.P99, Stat.P0],
                    interval=Time.from_seconds(2))
    ]
).run()

simulation_run_result.print_csv()
simulation_run_result.display_graphs()
