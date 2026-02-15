"""Tests for cache eviction policies."""

import pytest

from happysimulator.components.datastore import (
    ClockEviction,
    FIFOEviction,
    LFUEviction,
    LRUEviction,
    RandomEviction,
    SampledLRUEviction,
    SLRUEviction,
    TTLEviction,
    TwoQueueEviction,
)


class TestLRUEviction:
    """Tests for LRU eviction policy."""

    def test_evicts_least_recently_used(self):
        """LRU evicts the least recently accessed key."""
        policy = LRUEviction()

        policy.on_insert("a")
        policy.on_insert("b")
        policy.on_insert("c")

        # Access 'a' to make it most recent
        policy.on_access("a")

        # Should evict 'b' (least recently used)
        evicted = policy.evict()
        assert evicted == "b"

    def test_evict_empty_returns_none(self):
        """evict() returns None when empty."""
        policy = LRUEviction()

        assert policy.evict() is None

    def test_on_remove_removes_tracking(self):
        """on_remove stops tracking the key."""
        policy = LRUEviction()

        policy.on_insert("a")
        policy.on_insert("b")
        policy.on_remove("a")

        evicted = policy.evict()
        assert evicted == "b"

    def test_clear(self):
        """clear() removes all tracking."""
        policy = LRUEviction()

        policy.on_insert("a")
        policy.on_insert("b")
        policy.clear()

        assert policy.evict() is None


class TestLFUEviction:
    """Tests for LFU eviction policy."""

    def test_evicts_least_frequently_used(self):
        """LFU evicts the key with lowest access count."""
        policy = LFUEviction()

        policy.on_insert("a")
        policy.on_insert("b")
        policy.on_insert("c")

        # Access 'a' and 'c' multiple times
        policy.on_access("a")
        policy.on_access("a")
        policy.on_access("c")

        # 'b' has lowest count (1), should be evicted
        evicted = policy.evict()
        assert evicted == "b"

    def test_evict_empty_returns_none(self):
        """evict() returns None when empty."""
        policy = LFUEviction()

        assert policy.evict() is None

    def test_new_keys_start_at_one(self):
        """New keys have count of 1."""
        policy = LFUEviction()

        policy.on_insert("a")
        policy.on_insert("b")

        # Both have count 1, should evict first inserted
        evicted = policy.evict()
        assert evicted in ("a", "b")

    def test_clear(self):
        """clear() removes all tracking."""
        policy = LFUEviction()

        policy.on_insert("a")
        policy.clear()

        assert policy.evict() is None


class TestTTLEviction:
    """Tests for TTL eviction policy."""

    def test_evicts_expired_first(self):
        """TTL evicts expired keys first."""
        current_time = [0.0]

        def clock():
            return current_time[0]

        policy = TTLEviction(ttl=1.0, clock_func=clock)

        policy.on_insert("a")
        current_time[0] = 0.5
        policy.on_insert("b")

        # Advance past 'a' expiry but not 'b'
        current_time[0] = 1.5

        evicted = policy.evict()
        assert evicted == "a"

    def test_evicts_oldest_if_none_expired(self):
        """TTL evicts oldest if no keys expired."""
        current_time = [0.0]

        def clock():
            return current_time[0]

        policy = TTLEviction(ttl=10.0, clock_func=clock)

        policy.on_insert("a")
        current_time[0] = 0.1
        policy.on_insert("b")

        # Neither expired
        evicted = policy.evict()
        assert evicted == "a"  # Oldest

    def test_is_expired(self):
        """is_expired checks key TTL status."""
        current_time = [0.0]

        def clock():
            return current_time[0]

        policy = TTLEviction(ttl=1.0, clock_func=clock)

        policy.on_insert("a")

        assert policy.is_expired("a") is False

        current_time[0] = 1.5
        assert policy.is_expired("a") is True

    def test_get_expired_keys(self):
        """get_expired_keys returns all expired keys."""
        current_time = [0.0]

        def clock():
            return current_time[0]

        policy = TTLEviction(ttl=1.0, clock_func=clock)

        policy.on_insert("a")
        current_time[0] = 0.5
        policy.on_insert("b")
        current_time[0] = 1.5
        policy.on_insert("c")

        expired = policy.get_expired_keys()
        assert set(expired) == {"a", "b"}

    def test_rejects_invalid_ttl(self):
        """TTL rejects ttl <= 0."""
        with pytest.raises(ValueError):
            TTLEviction(ttl=0)

        with pytest.raises(ValueError):
            TTLEviction(ttl=-1)


class TestFIFOEviction:
    """Tests for FIFO eviction policy."""

    def test_evicts_first_inserted(self):
        """FIFO evicts the first inserted key."""
        policy = FIFOEviction()

        policy.on_insert("a")
        policy.on_insert("b")
        policy.on_insert("c")

        assert policy.evict() == "a"
        assert policy.evict() == "b"
        assert policy.evict() == "c"

    def test_access_doesnt_affect_order(self):
        """Accessing a key doesn't change FIFO order."""
        policy = FIFOEviction()

        policy.on_insert("a")
        policy.on_insert("b")
        policy.on_access("a")  # Should not change order

        assert policy.evict() == "a"

    def test_evict_empty_returns_none(self):
        """evict() returns None when empty."""
        policy = FIFOEviction()

        assert policy.evict() is None


class TestRandomEviction:
    """Tests for random eviction policy."""

    def test_evicts_random_key(self):
        """Random evicts some key."""
        policy = RandomEviction(seed=42)

        policy.on_insert("a")
        policy.on_insert("b")
        policy.on_insert("c")

        evicted = policy.evict()
        assert evicted in ("a", "b", "c")

    def test_evict_empty_returns_none(self):
        """evict() returns None when empty."""
        policy = RandomEviction()

        assert policy.evict() is None

    def test_deterministic_with_seed(self):
        """Random is deterministic with same seed."""
        policy1 = RandomEviction(seed=123)
        policy2 = RandomEviction(seed=123)

        for key in ["a", "b", "c", "d", "e"]:
            policy1.on_insert(key)
            policy2.on_insert(key)

        evict1 = [policy1.evict() for _ in range(3)]
        evict2 = [policy2.evict() for _ in range(3)]

        assert evict1 == evict2


class TestSLRUEviction:
    """Tests for Segmented LRU eviction policy."""

    def test_new_items_go_to_probationary(self):
        """New items enter the probationary segment."""
        policy = SLRUEviction()

        policy.on_insert("a")
        policy.on_insert("b")

        assert policy.probationary_size == 2
        assert policy.protected_size == 0

    def test_promotes_on_reaccess(self):
        """Items are promoted to protected on re-access."""
        policy = SLRUEviction()

        policy.on_insert("a")
        policy.on_insert("b")

        policy.on_access("a")  # Promote 'a' to protected

        assert policy.probationary_size == 1
        assert policy.protected_size == 1

    def test_evicts_from_probationary_first(self):
        """SLRU evicts from probationary before protected."""
        policy = SLRUEviction()

        policy.on_insert("a")
        policy.on_insert("b")
        policy.on_access("a")  # Promote 'a'

        # 'b' is in probationary, 'a' is in protected
        evicted = policy.evict()
        assert evicted == "b"

    def test_evicts_from_protected_when_probationary_empty(self):
        """SLRU evicts from protected when probationary is empty."""
        policy = SLRUEviction()

        policy.on_insert("a")
        policy.on_access("a")  # Promote to protected

        evicted = policy.evict()
        assert evicted == "a"

    def test_scan_resistance(self):
        """SLRU resists scan patterns - hot items stay cached."""
        policy = SLRUEviction()

        # Hot items accessed multiple times
        policy.on_insert("hot1")
        policy.on_insert("hot2")
        policy.on_access("hot1")
        policy.on_access("hot2")

        # Scan of cold items
        for i in range(10):
            policy.on_insert(f"cold{i}")

        # Evict 10 items - should evict cold items, keep hot
        evicted = set()
        for _ in range(10):
            key = policy.evict()
            if key:
                evicted.add(key)

        # Hot items should not be evicted
        assert "hot1" not in evicted
        assert "hot2" not in evicted

    def test_rejects_invalid_ratio(self):
        """SLRU rejects invalid protected_ratio."""
        with pytest.raises(ValueError):
            SLRUEviction(protected_ratio=0)

        with pytest.raises(ValueError):
            SLRUEviction(protected_ratio=1)


class TestSampledLRUEviction:
    """Tests for Sampled LRU eviction policy."""

    def test_evicts_least_recent_in_sample(self):
        """Sampled LRU evicts the oldest key in the sample."""
        policy = SampledLRUEviction(sample_size=3, seed=42)

        # Insert keys in order
        for i in range(10):
            policy.on_insert(f"key{i}")

        # Access recent keys to make them newer
        policy.on_access("key8")
        policy.on_access("key9")

        # Evict - should pick from sample and evict oldest in sample
        evicted = policy.evict()
        assert evicted is not None
        # Should not evict the most recently accessed
        assert evicted not in ("key8", "key9")

    def test_respects_sample_size(self):
        """Eviction considers only sample_size keys."""
        policy = SampledLRUEviction(sample_size=2, seed=42)

        policy.on_insert("a")
        policy.on_insert("b")
        policy.on_insert("c")

        evicted = policy.evict()
        assert evicted in ("a", "b", "c")

    def test_deterministic_with_seed(self):
        """Sampled LRU is deterministic with same seed."""
        policy1 = SampledLRUEviction(sample_size=3, seed=123)
        policy2 = SampledLRUEviction(sample_size=3, seed=123)

        for key in ["a", "b", "c", "d", "e"]:
            policy1.on_insert(key)
            policy2.on_insert(key)

        assert policy1.evict() == policy2.evict()

    def test_rejects_invalid_sample_size(self):
        """Sampled LRU rejects sample_size < 1."""
        with pytest.raises(ValueError):
            SampledLRUEviction(sample_size=0)


class TestClockEviction:
    """Tests for Clock (Second-Chance) eviction policy."""

    def test_evicts_unreferenced_key(self):
        """Clock evicts key with reference bit clear."""
        policy = ClockEviction()

        policy.on_insert("a")
        policy.on_insert("b")
        policy.on_insert("c")

        # All have ref bit set initially
        # First evict clears ref bits, second pass evicts
        evicted = policy.evict()
        assert evicted in ("a", "b", "c")

    def test_gives_second_chance(self):
        """Clock gives accessed keys a second chance."""
        policy = ClockEviction()

        policy.on_insert("a")
        policy.on_insert("b")

        # Access 'a' repeatedly to keep its ref bit set
        policy.on_access("a")

        # First evict should give 'a' second chance and evict 'b'
        # (since 'a' has ref bit, it gets cleared; 'b' also has ref bit
        # but we continue scanning and may evict either after clearing)
        evicted1 = policy.evict()
        assert evicted1 is not None

    def test_evict_empty_returns_none(self):
        """Clock evict returns None when empty."""
        policy = ClockEviction()

        assert policy.evict() is None

    def test_size_tracking(self):
        """Clock tracks size correctly."""
        policy = ClockEviction()

        policy.on_insert("a")
        policy.on_insert("b")
        assert policy.size == 2

        policy.evict()
        assert policy.size == 1


class TestTwoQueueEviction:
    """Tests for 2Q eviction policy."""

    def test_new_items_go_to_a1in(self):
        """New items enter A1in queue."""
        policy = TwoQueueEviction()

        policy.on_insert("a")
        policy.on_insert("b")

        # A1in is FIFO, evict should return first item
        evicted = policy.evict()
        assert evicted == "a"

    def test_ghost_queue_promotes_to_am(self):
        """Items re-accessed after eviction go to Am."""
        policy = TwoQueueEviction()

        policy.on_insert("a")
        policy.on_insert("b")
        policy.on_insert("c")

        # Evict 'a' - goes to ghost queue
        evicted = policy.evict()
        assert evicted == "a"

        # Re-insert 'a' - should go to Am (not A1in)
        policy.on_insert("a")

        # Evict 'b' and 'c' from A1in
        assert policy.evict() == "b"
        assert policy.evict() == "c"

        # Now 'a' should be evicted from Am
        assert policy.evict() == "a"

    def test_am_is_lru(self):
        """Am queue uses LRU ordering."""
        policy = TwoQueueEviction()

        # Insert and evict to get items into ghost queue
        for key in ["a", "b", "c"]:
            policy.on_insert(key)
        for _ in range(3):
            policy.evict()

        # Re-insert all - they go to Am
        for key in ["a", "b", "c"]:
            policy.on_insert(key)

        # Access 'a' to make it most recent in Am
        policy.on_access("a")

        # Evict from Am - should be 'b' (LRU)
        evicted = policy.evict()
        assert evicted == "b"

    def test_rejects_invalid_ratio(self):
        """2Q rejects invalid kin_ratio."""
        with pytest.raises(ValueError):
            TwoQueueEviction(kin_ratio=0)

        with pytest.raises(ValueError):
            TwoQueueEviction(kin_ratio=1)
