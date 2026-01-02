import unittest
import random

from archive.exponential_latency import ExponentialLatency
from happysimulator.utils.instant import Instant


class TestExponentialLatency(unittest.TestCase):
    def setUp(self):
        random.seed(0)

    def test_latency_non_negative(self):
        mean_latency = Instant.from_seconds(1)
        exp_latency = ExponentialLatency(mean_latency)
        for _ in range(1000):
            latency = exp_latency.get_latency(Instant.Epoch)
            self.assertGreaterEqual(latency.to_seconds(), 0, "Latency should be non-negative")

    def test_approximate_mean_latency(self):
        mean_latency_value = 1
        mean_latency = Instant.from_seconds(mean_latency_value)
        exp_latency = ExponentialLatency(mean_latency)
        latencies = [exp_latency.get_latency(Instant.Epoch).to_seconds() for _ in range(10000)]
        observed_mean = sum(latencies) / len(latencies)
        self.assertAlmostEqual(observed_mean, mean_latency_value, delta=0.1, msg="Observed mean does not approximate expected mean")

if __name__ == '__main__':
    unittest.main()