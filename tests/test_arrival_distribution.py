import unittest
from unittest.mock import patch

from archive.arrival_distribution import ArrivalDistribution
from happysimulator.utils.instant import Instant


class TestArrivalDistribution(unittest.TestCase):
    def test_constant_distribution(self):
        rate_per_second = 1  # 1 event per second
        current_time = Instant.from_seconds(0)
        expected_time = Instant.from_seconds(1)  # Expecting the next event in 1 second

        result = ArrivalDistribution.CONSTANT.get_next_arrival_time(current_time, rate_per_second)
        self.assertEqual(result, expected_time)

    @patch('random.expovariate', return_value=0.5)  # Mock random to return a fixed value
    def test_poisson_distribution(self, mock_expovariate):
        current_time = Instant.from_seconds(1)
        rate_per_second = 2
        expected_time = Instant.from_seconds(1.5)
        next_time = ArrivalDistribution.POISSON.get_next_arrival_time(current_time, rate_per_second)
        self.assertEqual(next_time, expected_time, "POISSON distribution did not produce expected next arrival time")


if __name__ == '__main__':
    unittest.main()