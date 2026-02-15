"""Tests for BalkingQueue component."""

from __future__ import annotations

import random

from happysimulator.components.industrial.balking import BalkingQueue
from happysimulator.components.queue_policy import FIFOQueue, LIFOQueue


class TestBalkingQueuePolicy:
    def test_creates_with_defaults(self):
        inner = FIFOQueue()
        bq = BalkingQueue(inner)
        assert bq.balk_threshold == 5
        assert bq.balk_probability == 1.0
        assert bq.balked == 0

    def test_accepts_below_threshold(self):
        inner = FIFOQueue()
        bq = BalkingQueue(inner, balk_threshold=3)
        assert bq.push("a") is True
        assert bq.push("b") is True
        assert bq.balked == 0

    def test_rejects_at_threshold_with_probability_1(self):
        inner = FIFOQueue()
        bq = BalkingQueue(inner, balk_threshold=2, balk_probability=1.0)
        bq.push("a")
        bq.push("b")
        assert bq.push("c") is False
        assert bq.balked == 1

    def test_probabilistic_balking(self):
        random.seed(42)
        inner = FIFOQueue()
        bq = BalkingQueue(inner, balk_threshold=0, balk_probability=0.5)

        accepted = 0
        for _ in range(100):
            if bq.push("x"):
                accepted += 1
                bq.pop()  # Keep queue at 0 to test threshold=0

        # With p=0.5 and 100 trials, we expect ~50 balks
        assert 30 < bq.balked < 70
        assert 30 < accepted < 70

    def test_zero_balk_probability_never_balks(self):
        inner = FIFOQueue()
        bq = BalkingQueue(inner, balk_threshold=0, balk_probability=0.0)
        for _ in range(10):
            bq.push("x")
        assert bq.balked == 0
        assert len(bq) == 10

    def test_delegates_pop_and_peek(self):
        inner = FIFOQueue()
        bq = BalkingQueue(inner, balk_threshold=10)
        bq.push("a")
        bq.push("b")
        assert bq.peek() == "a"
        assert bq.pop() == "a"
        assert bq.pop() == "b"
        assert bq.is_empty() is True

    def test_capacity_from_inner(self):
        inner = FIFOQueue(capacity=5)
        bq = BalkingQueue(inner, balk_threshold=10)
        assert bq.capacity == 5

    def test_wraps_lifo(self):
        inner = LIFOQueue()
        bq = BalkingQueue(inner, balk_threshold=5)
        bq.push("a")
        bq.push("b")
        assert bq.pop() == "b"  # LIFO order
