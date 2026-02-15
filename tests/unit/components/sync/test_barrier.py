"""Tests for Barrier."""

import pytest

from happysimulator.components.sync import Barrier


class TestBarrierCreation:
    """Tests for Barrier creation."""

    def test_creates_with_parties(self):
        """Barrier is created with specified parties."""
        barrier = Barrier(name="test", parties=3)

        assert barrier.parties == 3
        assert barrier.waiting == 0
        assert barrier.broken is False
        assert barrier.generation == 0

    def test_has_name(self):
        """Barrier has a name."""
        barrier = Barrier(name="phase_sync", parties=2)

        assert barrier.name == "phase_sync"

    def test_rejects_zero_parties(self):
        """Barrier rejects parties < 1."""
        with pytest.raises(ValueError):
            Barrier(name="test", parties=0)

    def test_rejects_negative_parties(self):
        """Barrier rejects negative parties."""
        with pytest.raises(ValueError):
            Barrier(name="test", parties=-1)

    def test_accepts_single_party(self):
        """Barrier accepts single party (trivial barrier)."""
        barrier = Barrier(name="test", parties=1)

        assert barrier.parties == 1


class TestBarrierSingleParty:
    """Tests for single-party barrier (trivial case)."""

    def test_single_party_releases_immediately(self):
        """Single party breaks barrier immediately."""
        barrier = Barrier(name="test", parties=1)

        gen = barrier.wait()
        list(gen)

        # Should return immediately with index 0 (leader)
        assert barrier.waiting == 0
        assert barrier.generation == 1
        assert barrier.stats.barrier_breaks == 1

    def test_single_party_returns_zero(self):
        """Single party gets arrival index 0."""
        barrier = Barrier(name="test", parties=1)

        gen = barrier.wait()
        try:
            while True:
                next(gen)
        except StopIteration as e:
            arrival_index = e.value

        assert arrival_index == 0


class TestBarrierTwoParties:
    """Tests for two-party barrier."""

    def test_first_party_waits(self):
        """First party waits for second."""
        barrier = Barrier(name="test", parties=2)

        gen1 = barrier.wait()
        next(gen1)  # First yield - party is waiting

        assert barrier.waiting == 1
        assert barrier.generation == 0

    def test_second_party_breaks_barrier(self):
        """Second party breaks the barrier."""
        barrier = Barrier(name="test", parties=2)

        # First party waits
        gen1 = barrier.wait()
        next(gen1)

        # Second party arrives - breaks barrier
        gen2 = barrier.wait()
        try:
            while True:
                next(gen2)
        except StopIteration as e:
            index2 = e.value

        assert index2 == 0  # Last to arrive gets 0
        assert barrier.generation == 1
        assert barrier.stats.barrier_breaks == 1

    def test_first_party_released(self):
        """First party is released when barrier breaks."""
        barrier = Barrier(name="test", parties=2)

        # First party waits
        gen1 = barrier.wait()
        next(gen1)

        # Second party breaks barrier
        gen2 = barrier.wait()
        list(gen2)

        # First party should now complete
        try:
            while True:
                next(gen1)
        except StopIteration as e:
            index1 = e.value

        assert index1 == 1  # First to arrive gets higher index

    def test_arrival_indices(self):
        """Parties get correct arrival indices."""
        barrier = Barrier(name="test", parties=2)

        # First party
        gen1 = barrier.wait()
        next(gen1)

        # Second party - breaks barrier
        gen2 = barrier.wait()
        try:
            while True:
                next(gen2)
        except StopIteration as e:
            index2 = e.value

        # First party continues
        try:
            while True:
                next(gen1)
        except StopIteration as e:
            index1 = e.value

        # Last to arrive gets 0 (leader), first gets 1
        assert index2 == 0
        assert index1 == 1


class TestBarrierMultipleParties:
    """Tests for barriers with multiple parties."""

    def test_three_parties(self):
        """Three parties synchronize correctly."""
        barrier = Barrier(name="test", parties=3)

        # Party 1 waits
        gen1 = barrier.wait()
        next(gen1)
        assert barrier.waiting == 1

        # Party 2 waits
        gen2 = barrier.wait()
        next(gen2)
        assert barrier.waiting == 2

        # Party 3 breaks barrier
        gen3 = barrier.wait()
        try:
            while True:
                next(gen3)
        except StopIteration as e:
            index3 = e.value

        assert index3 == 0  # Last gets leader index
        assert barrier.waiting == 0
        assert barrier.generation == 1

    def test_reuse_after_break(self):
        """Barrier can be reused after breaking."""
        barrier = Barrier(name="test", parties=2)

        # First round
        gen1a = barrier.wait()
        next(gen1a)
        gen2a = barrier.wait()
        list(gen2a)
        list(gen1a)  # Complete first party

        assert barrier.generation == 1

        # Second round
        gen1b = barrier.wait()
        next(gen1b)
        assert barrier.waiting == 1

        gen2b = barrier.wait()
        list(gen2b)

        assert barrier.generation == 2
        assert barrier.stats.barrier_breaks == 2


class TestBarrierReset:
    """Tests for Barrier.reset()."""

    def test_reset_clears_waiters(self):
        """reset() releases waiting parties."""
        barrier = Barrier(name="test", parties=3)

        gen1 = barrier.wait()
        next(gen1)

        gen2 = barrier.wait()
        next(gen2)

        barrier.reset()

        assert barrier.waiting == 0
        assert barrier.broken is False
        assert barrier.stats.resets == 1

    def test_reset_advances_generation(self):
        """reset() advances generation."""
        barrier = Barrier(name="test", parties=2)
        initial_gen = barrier.generation

        barrier.reset()

        assert barrier.generation == initial_gen + 1


class TestBarrierAbort:
    """Tests for Barrier.abort()."""

    def test_abort_breaks_barrier(self):
        """abort() puts barrier in broken state."""
        barrier = Barrier(name="test", parties=2)

        barrier.abort()

        assert barrier.broken is True

    def test_wait_raises_when_broken(self):
        """wait() raises when barrier is broken."""
        barrier = Barrier(name="test", parties=2)
        barrier.abort()

        with pytest.raises(RuntimeError):
            gen = barrier.wait()
            next(gen)

    def test_abort_releases_waiters(self):
        """abort() releases waiting parties."""
        barrier = Barrier(name="test", parties=3)

        gen1 = barrier.wait()
        next(gen1)

        barrier.abort()

        assert barrier.waiting == 0


class TestBarrierStatistics:
    """Tests for Barrier statistics."""

    def test_tracks_wait_calls(self):
        """Statistics track wait() calls."""
        barrier = Barrier(name="test", parties=2)

        gen1 = barrier.wait()
        next(gen1)

        gen2 = barrier.wait()
        list(gen2)

        assert barrier.stats.wait_calls == 2

    def test_tracks_barrier_breaks(self):
        """Statistics track barrier breaks."""
        barrier = Barrier(name="test", parties=1)

        list(barrier.wait())
        list(barrier.wait())
        list(barrier.wait())

        assert barrier.stats.barrier_breaks == 3

    def test_tracks_resets(self):
        """Statistics track resets."""
        barrier = Barrier(name="test", parties=2)

        barrier.reset()
        barrier.reset()

        assert barrier.stats.resets == 2


class TestBarrierHandleEvent:
    """Tests for Barrier.handle_event()."""

    def test_handle_event_does_nothing(self):
        """handle_event is a no-op."""
        barrier = Barrier(name="test", parties=2)

        # Should not raise
        barrier.handle_event(None)
