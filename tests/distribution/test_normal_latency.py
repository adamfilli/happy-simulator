import unittest
from unittest.mock import patch

from happysimulator.math.normal_latency import NormalLatency
from happysimulator.utils.instant import Instant


class TestNormalLatency(unittest.TestCase):
    def setUp(self):
        self.mean_latency = Instant.from_seconds(10)  # 10 seconds
        self.std_dev = Instant.from_seconds(2)  # 2 seconds variance
        self.normal_latency = NormalLatency(self.mean_latency, self.std_dev)

    def test_initialization(self):
        self.assertEqual(self.normal_latency._mean_latency, 10)
        self.assertEqual(self.normal_latency._std_dev, 2)

    @patch('random.gauss')
    def test_get_latency_returns_time_object(self, mock_random_gauss):
        mock_random_gauss.return_value = 10
        latency = self.normal_latency.get_latency(Instant.from_seconds(0))
        self.assertIsInstance(latency, Instant)

    @patch('random.gauss')
    def test_get_latency_non_negative(self, mock_random_gauss):
        mock_random_gauss.return_value = -5  # A value that would normally be invalid
        latency = self.normal_latency.get_latency(Instant.from_seconds(0))
        self.assertTrue(latency.to_seconds() >= 0)

    def test_latency_distribution(self):
        sample_size = 1000
        generated_latencies = [self.normal_latency.get_latency(Instant.from_seconds(0)).to_seconds() for _ in
                               range(sample_size)]

        mean = sum(generated_latencies) / sample_size
        within_one_std_dev = sum(1 for x in generated_latencies if
                                 self.mean_latency.to_seconds() - self.std_dev.to_seconds() <= x <= self.mean_latency.to_seconds() + self.std_dev.to_seconds()) / sample_size

        self.assertTrue(abs(mean - self.mean_latency.to_seconds()) < 1, "Mean latency is not within expected range")
        self.assertTrue(within_one_std_dev > 0.5, # 0.68 is the expected
                        "Not enough values within one standard deviation")


if __name__ == '__main__':
    unittest.main()