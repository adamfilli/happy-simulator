import unittest
from unittest.mock import MagicMock

from happysimulator.entities.entity import Entity


class TestEntity(unittest.TestCase):
    def setUp(self):
        self.entity = Entity("Test Entity")
        self.mock_data = MagicMock()
        self.mock_event = MagicMock()
        self.mock_sink = MagicMock()
        self.mock_event.sink = [self.mock_sink]
        self.mock_stat = MagicMock(name="stat_name")
        self.mock_event.stat = self.mock_stat
        self.mock_event.time = 100
        self.mock_event.interval = 10

    def test_sink_data_with_float_single_stat(self):
        self.entity.sink_data(10.0, self.mock_event)
        self.mock_sink.add_stat.assert_called_once_with(100, 10.0)

    def test_sink_data_with_int_single_stat(self):
        self.entity.sink_data(10, self.mock_event)
        self.mock_sink.add_stat.assert_called_once_with(100, 10.0)

    def test_sink_data_with_float_multiple_stats_raises_error(self):
        self.mock_event.stat = [self.mock_stat, MagicMock(name="another_stat")]
        with self.assertRaises(RuntimeError):
            self.entity.sink_data(10.0, self.mock_event)

    def test_sink_data_with_unsupported_type_raises_error(self):
        self.mock_event.stat = [self.mock_stat]
        with self.assertRaises(RuntimeError):
            self.entity.sink_data("unsupported", self.mock_event)

if __name__ == '__main__':
    unittest.main()