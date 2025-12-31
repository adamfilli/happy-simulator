import unittest

from happysimulator.load import SpikeProfile
from happysimulator.utils.instant import Instant


class TestSpikeProfile(unittest.TestCase):
    def setUp(self):
        # Setup common to all tests
        self.rampup_start = Instant.from_seconds(10)
        self.rampdown_start = Instant.from_seconds(20)
        self.starting_rate = 1.0
        self.rampup_factor = 0.5
        self.rampdown_factor = 0.25

    def test_rate_before_rampup(self):
        spike_profile = SpikeProfile(self.rampup_start, self.rampdown_start, self.starting_rate, self.rampup_factor, self.rampdown_factor)
        time_before_rampup = Instant.from_seconds(5)
        self.assertEqual(spike_profile.get_rate(time_before_rampup), self.starting_rate)

    def test_rate_during_rampup(self):
        spike_profile = SpikeProfile(self.rampup_start, self.rampdown_start, self.starting_rate, self.rampup_factor, self.rampdown_factor)
        time_during_rampup = Instant.from_seconds(15)  # 5 seconds into rampup
        expected_rate = self.starting_rate + 5 * self.rampup_factor
        self.assertEqual(spike_profile.get_rate(time_during_rampup), expected_rate)

    def test_rate_at_peak(self):
        spike_profile = SpikeProfile(self.rampup_start, self.rampdown_start, self.starting_rate, self.rampup_factor, self.rampdown_factor)
        self.assertEqual(spike_profile.get_rate(self.rampdown_start), spike_profile.peak_rate)

    def test_rate_during_rampdown(self):
        spike_profile = SpikeProfile(self.rampup_start, self.rampdown_start, self.starting_rate, self.rampup_factor, self.rampdown_factor)
        time_during_rampdown = Instant.from_seconds(25)  # 5 seconds into rampdown
        expected_rate = spike_profile.peak_rate - 5 * self.rampdown_factor
        self.assertEqual(spike_profile.get_rate(time_during_rampdown), expected_rate)

    def test_rate_after_rampdown(self):
        spike_profile = SpikeProfile(self.rampup_start, self.rampdown_start, self.starting_rate, self.rampup_factor, self.rampdown_factor)
        time_after_rampdown = Instant.from_seconds(30)
        # Ensure rate does not go below starting_rate after rampdown
        self.assertEqual(spike_profile.get_rate(time_after_rampdown), max(self.starting_rate, spike_profile.peak_rate - 10 * self.rampdown_factor))

if __name__ == '__main__':
    unittest.main()
