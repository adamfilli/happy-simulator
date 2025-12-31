import unittest

from happysimulator.utils.duration import Duration
from happysimulator.utils.instant import Instant


class TestDuration(unittest.TestCase):
    def test_from_seconds_int(self):
        d = Duration.from_seconds(1)
        self.assertEqual(d.to_seconds(), 1.0)

    def test_from_seconds_float(self):
        d = Duration.from_seconds(0.5)
        self.assertEqual(d.to_seconds(), 0.5)

    def test_add_duration(self):
        a = Duration.from_seconds(1)
        b = Duration.from_seconds(2)
        self.assertEqual((a + b).to_seconds(), 3.0)

    def test_sub_duration(self):
        a = Duration.from_seconds(3)
        b = Duration.from_seconds(1)
        self.assertEqual((a - b).to_seconds(), 2.0)

    def test_compare(self):
        self.assertTrue(Duration.from_seconds(1) < Duration.from_seconds(2))
        self.assertTrue(Duration.from_seconds(2) >= Duration.from_seconds(1))

    def test_instant_plus_duration(self):
        t = Instant.from_seconds(1)
        d = Duration.from_seconds(2)
        self.assertEqual((t + d).to_seconds(), 3.0)
        # also Duration + Instant should work via __radd__
        self.assertEqual((d + t).to_seconds(), 3.0)

    def test_instant_minus_duration(self):
        t = Instant.from_seconds(3)
        d = Duration.from_seconds(1)
        self.assertEqual((t - d).to_seconds(), 2.0)


if __name__ == "__main__":
    unittest.main()