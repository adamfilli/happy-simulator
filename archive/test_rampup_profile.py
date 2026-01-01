import unittest

from happysimulator.load import RampupProfile
from happysimulator.utils.instant import Instant


class TestRampupProfile(unittest.TestCase):
    def test_initialization(self):
        """Tests that the profile is initialized with the correct starting rate and ramp-up factor."""
        profile = RampupProfile(5, 0.5)
        self.assertEqual(profile._rate, 5)
        self.assertEqual(profile._factor, 0.5)

    def test_get_rate(self):
        """Tests that get_rate correctly calculates the rate at various times."""
        profile = RampupProfile(5, 0.5)
        time_zero = Instant(0)
        self.assertEqual(profile.get_rate(time_zero), 5)

        time_ten = Instant.from_seconds(10)
        self.assertEqual(profile.get_rate(time_ten), 10)  # Starting rate 5 + (10 seconds * 0.5 factor) = 10

    def test_negative_starting_rate(self):
        """Tests handling of negative starting rates."""
        # Assuming we want to check for an error or a specific behavior
        with self.assertRaises(ValueError):
            RampupProfile(-1, 0.5)

    def test_negative_rampup_factor(self):
        """Tests handling of negative ramp-up factors."""
        profile = RampupProfile(5, -0.5)
        time_ten = Instant.from_seconds(10)
        self.assertEqual(profile.get_rate(time_ten), 0)  # Example expected behavior


if __name__ == '__main__':
    unittest.main()
