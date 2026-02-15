"""Integration tests for SoftTTLCache: client -> cache -> datastore.

Verifies the three cache zones (fresh, stale, expired) work correctly
when a client reads through a SoftTTLCache backed by a KVStore.
"""

from typing import Generator

from happysimulator import Entity, Event, Instant, Simulation
from happysimulator.components.datastore import KVStore, SoftTTLCache


class Client(Entity):
    """Simple client that reads keys from a SoftTTLCache."""

    def __init__(self, name: str, cache: SoftTTLCache):
        super().__init__(name)
        self.cache = cache
        self.results: list[tuple[str, object]] = []

    def handle_event(self, event: Event) -> Generator[float, None, None]:
        key = event.context["key"]
        value = yield from self.cache.get(key)
        self.results.append((key, value))
        return None


def _build(
    *,
    soft_ttl: float = 5.0,
    hard_ttl: float = 10.0,
    db_read_latency: float = 0.01,
    cache_capacity: int | None = None,
) -> tuple[KVStore, SoftTTLCache, Client]:
    """Create a datastore -> cache -> client stack."""
    datastore = KVStore(name="db", read_latency=db_read_latency, write_latency=0.005)
    cache = SoftTTLCache(
        name="cache",
        backing_store=datastore,
        soft_ttl=soft_ttl,
        hard_ttl=hard_ttl,
        cache_capacity=cache_capacity,
    )
    client = Client(name="client", cache=cache)
    return datastore, cache, client


def _request(target: Client, time_s: float, key: str) -> Event:
    return Event(
        time=Instant.from_seconds(time_s),
        event_type="Read",
        target=target,
        context={"key": key},
    )


class TestClientCacheDatastoreFlow:
    """End-to-end: client reads through SoftTTLCache into KVStore."""

    def test_cold_miss_fetches_from_datastore(self):
        """First read of a key goes to the datastore and populates cache."""
        datastore, cache, client = _build()
        datastore.put_sync("user:1", {"name": "Alice"})

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=1.0,
            entities=[client, cache],
        )
        sim.schedule(_request(client, 0.0, "user:1"))
        sim.run()

        assert client.results == [("user:1", {"name": "Alice"})]
        assert cache.stats.hard_misses == 1
        assert cache.stats.fresh_hits == 0
        assert cache.contains_cached("user:1")

    def test_fresh_hit_avoids_datastore(self):
        """Second read within soft_ttl is a fresh hit — no extra datastore read."""
        datastore, cache, client = _build(soft_ttl=5.0, hard_ttl=10.0)
        datastore.put_sync("user:1", "v1")

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=2.0,
            entities=[client, cache],
        )
        sim.schedule(_request(client, 0.0, "user:1"))   # cold miss
        sim.schedule(_request(client, 1.0, "user:1"))   # fresh hit (1s < 5s soft_ttl)
        sim.run()

        assert cache.stats.hard_misses == 1
        assert cache.stats.fresh_hits == 1
        # Datastore should have been read only once (the cold miss)
        assert datastore.stats.reads == 1

    def test_stale_hit_triggers_background_refresh(self):
        """Read in the stale zone returns cached value and starts a refresh."""
        datastore, cache, client = _build(soft_ttl=2.0, hard_ttl=10.0)
        datastore.put_sync("user:1", "v1")

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=5.0,
            entities=[client, cache],
        )
        sim.schedule(_request(client, 0.0, "user:1"))   # cold miss
        sim.schedule(_request(client, 3.0, "user:1"))   # stale hit (3s > 2s soft_ttl)
        sim.run()

        assert cache.stats.hard_misses == 1
        assert cache.stats.stale_hits == 1
        assert cache.stats.background_refreshes == 1
        assert cache.stats.refresh_successes == 1
        # Client still got the value both times
        assert all(v == "v1" for _, v in client.results)

    def test_expired_entry_blocks_for_fresh_fetch(self):
        """Read after hard_ttl forces a blocking fetch from the datastore."""
        datastore, cache, client = _build(soft_ttl=2.0, hard_ttl=5.0)
        datastore.put_sync("user:1", "v1")

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=8.0,
            entities=[client, cache],
        )
        sim.schedule(_request(client, 0.0, "user:1"))    # cold miss
        sim.schedule(_request(client, 6.0, "user:1"))    # expired (6s > 5s hard_ttl)
        sim.run()

        assert cache.stats.hard_misses == 2
        assert datastore.stats.reads == 2
        assert client.results == [("user:1", "v1"), ("user:1", "v1")]

    def test_missing_key_returns_none(self):
        """Reading a key that doesn't exist in the datastore returns None."""
        _, cache, client = _build()

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=1.0,
            entities=[client, cache],
        )
        sim.schedule(_request(client, 0.0, "nonexistent"))
        sim.run()

        assert client.results == [("nonexistent", None)]
        assert cache.stats.hard_misses == 1

    def test_multiple_keys(self):
        """Multiple distinct keys each go through cold miss then fresh hit."""
        datastore, cache, client = _build(soft_ttl=5.0, hard_ttl=10.0)
        for i in range(3):
            datastore.put_sync(f"k{i}", f"v{i}")

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=3.0,
            entities=[client, cache],
        )
        # First pass: all cold misses
        for i in range(3):
            sim.schedule(_request(client, 0.1 * i, f"k{i}"))
        # Second pass: all fresh hits
        for i in range(3):
            sim.schedule(_request(client, 1.0 + 0.1 * i, f"k{i}"))
        sim.run()

        assert cache.stats.hard_misses == 3
        assert cache.stats.fresh_hits == 3
        assert cache.cache_size == 3

    def test_lru_eviction_under_capacity(self):
        """When cache is at capacity, least-recently-used key is evicted."""
        datastore, cache, client = _build(
            soft_ttl=30.0, hard_ttl=60.0, cache_capacity=2
        )
        for i in range(3):
            datastore.put_sync(f"k{i}", f"v{i}")

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=5.0,
            entities=[client, cache],
        )
        sim.schedule(_request(client, 0.0, "k0"))  # miss, cache: [k0]
        sim.schedule(_request(client, 0.5, "k1"))  # miss, cache: [k0, k1]
        sim.schedule(_request(client, 1.0, "k2"))  # miss, evicts k0 -> cache: [k1, k2]
        sim.schedule(_request(client, 1.5, "k0"))  # miss again (was evicted)
        sim.run()

        assert cache.stats.hard_misses == 4
        assert cache.stats.evictions == 2  # k0 evicted first, then k1
        assert cache.cache_size == 2
