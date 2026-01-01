import unittest
from unittest.mock import MagicMock

from happysimulator.events.event import Event
from happysimulator.utils.instant import Instant


class TestableEvent(Event):
    pass

class TestEvent(unittest.TestCase):
    def setUp(self):
        self.time = MagicMock(spec=Instant)
        self.callback = MagicMock()
        self.name = "TestEvent"
        self.event1 = TestableEvent(self.time, self.name, self.callback)
        self.event2 = TestableEvent(self.time, self.name, self.callback)

    def test_hash_uniqueness(self):
        # Since uuid.uuid4() is used in hash generation, each instance should have a unique hash
        self.assertNotEqual(hash(self.event1), hash(self.event2))

    def test_equality(self):
        # Two events with the same hash should be equal, but due to uuid, this should not happen
        event_copy = TestableEvent(self.time, self.name, self.callback)
        event_copy._hash = self.event1._hash  # Forcing hash equality
        self.assertEqual(self.event1, event_copy)

    def test_inequality(self):
        self.assertNotEqual(self.event1, self.event2)

    def test_ordering(self):
        self.event1._hash = 1
        self.event2._hash = 2
        self.assertTrue(self.event1 < self.event2)
        self.assertTrue(self.event1 <= self.event2)
        self.assertTrue(self.event2 > self.event1)
        self.assertTrue(self.event2 >= self.event1)

    def test_comparison_with_non_event(self):
        self.assertEqual(self.event1.__eq__(42), NotImplemented)
        self.assertEqual(self.event1.__lt__(42), NotImplemented)


if __name__ == '__main__':
    unittest.main()
