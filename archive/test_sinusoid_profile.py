import unittest

from happysimulator.load import SinusoidProfile
from happysimulator.utils.instant import Instant


class TestSinusoidProfile(unittest.TestCase):
    def test_initialization(self):
        profile = SinusoidProfile(1.0, 5.0, Instant.from_seconds(10))
        self.assertEqual(profile._shift, 1.0)
        self.assertEqual(profile._amplitude, 5.0)
        self.assertEqual(profile._period.to_seconds(), 10)

    def test_get_rate_basic(self):
        profile = SinusoidProfile(1.0, 5.0, Instant.from_seconds(10))
        # At time 0, sin(0) = 0, so rate should be equal to shift
        self.assertEqual(profile.get_rate(Instant.from_seconds(0)), 1.0)
        # At time 2.5 seconds, sin(pi/2) = 1, so rate should be shift + amplitude
        self.assertAlmostEqual(profile.get_rate(Instant.from_seconds(2.5)), 6.0)


if __name__ == '__main__':
    unittest.main()