import unittest

from happysimulator.events.event import Event
from happysimulator.event_heap import EventHeap
from happysimulator.utils.instant import Instant


class MockEvent(Event):
    def invoke(self) -> ['Event']:
        return []

    def __init__(self, time: Instant):
        super().__init__(time, "anything", self.invoke)


class TestEventHeap(unittest.TestCase):

    def test_heap_operations(self):
        # Initialize the EventHeap with a list of events
        initial_events = [MockEvent(Instant.from_seconds(s)) for s in [5, 1, 3]]
        heap = EventHeap(initial_events)

        # (1 -> 3 -> 5)
        self.assertTrue(heap.has_events())

        # Test peek
        self.assertEqual(heap.peek().time.to_seconds(), 1)

        # Test pop
        self.assertEqual(heap.pop().time.to_seconds(), 1)
        self.assertEqual(heap.pop().time.to_seconds(), 3)

        # (5)
        heap.push(MockEvent(Instant.from_seconds(2)))
        # (2 -> 5)
        self.assertEqual(heap.peek().time.to_seconds(), 2)

        heap.push([MockEvent(Instant.from_seconds(s)) for s in [1, 3, 5, 6]])
        # (1 -> 2 -> 3 -> 5 -> 5 -> 6)
        self.assertEqual(heap.pop().time.to_seconds(), 1)
        self.assertEqual(heap.pop().time.to_seconds(), 2)
        self.assertEqual(heap.pop().time.to_seconds(), 3)
        self.assertEqual(heap.pop().time.to_seconds(), 5)
        self.assertEqual(heap.pop().time.to_seconds(), 5)
        self.assertEqual(heap.pop().time.to_seconds(), 6)

        # ()
        self.assertFalse(heap.has_events())

if __name__ == '__main__':
    unittest.main()
