"""Unit tests for DNSResolver."""

import pytest

from happysimulator.components.infrastructure.dns_resolver import (
    DNSResolver,
    DNSRecord,
    DNSStats,
)
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant


class TestDNSRecord:
    def test_creation(self):
        record = DNSRecord("example.com", "10.0.0.1", ttl_s=60.0)
        assert record.hostname == "example.com"
        assert record.ip_address == "10.0.0.1"
        assert record.ttl_s == 60.0

    def test_default_ttl(self):
        record = DNSRecord("example.com", "10.0.0.1")
        assert record.ttl_s == 300.0


class TestDNSResolverCreation:
    def test_defaults(self):
        dns = DNSResolver("dns")
        assert dns.name == "dns"
        assert dns.cache_size == 0

    def test_invalid_capacity(self):
        with pytest.raises(ValueError, match="cache_capacity must be >= 1"):
            DNSResolver("bad", cache_capacity=0)

    def test_stats_initial(self):
        dns = DNSResolver("dns")
        stats = dns.stats
        assert isinstance(stats, DNSStats)
        assert stats.lookups == 0
        assert stats.cache_hits == 0
        assert stats.cache_misses == 0
        assert stats.hit_rate == 0.0
        assert stats.avg_resolution_latency_s == 0.0


class TestDNSResolverBehavior:
    def _make_dns(self, **kwargs) -> tuple[DNSResolver, Simulation]:
        dns = DNSResolver("test_dns", **kwargs)
        sim = Simulation(
            start_time=Instant.from_seconds(0),
            end_time=Instant.from_seconds(1000),
            entities=[dns],
        )
        return dns, sim

    def _exhaust(self, gen):
        """Run a generator to completion."""
        values = []
        try:
            while True:
                values.append(next(gen))
        except StopIteration as e:
            return values, e.value

    def test_resolve_known_record(self):
        records = {"api.example.com": DNSRecord("api.example.com", "10.0.0.1")}
        dns, sim = self._make_dns(records=records)

        values, ip = self._exhaust(dns.resolve("api.example.com"))
        assert ip == "10.0.0.1"
        assert len(values) == 3  # root + tld + auth latency
        assert dns.stats.cache_misses == 1

    def test_resolve_unknown_returns_none(self):
        dns, sim = self._make_dns()
        values, ip = self._exhaust(dns.resolve("nonexistent.com"))
        assert ip is None
        assert len(values) == 3  # still does full lookup

    def test_cache_hit_on_second_lookup(self):
        records = {"api.example.com": DNSRecord("api.example.com", "10.0.0.1")}
        dns, sim = self._make_dns(records=records)

        # First lookup: miss
        self._exhaust(dns.resolve("api.example.com"))
        assert dns.stats.cache_misses == 1

        # Second lookup: hit (no latency)
        values, ip = self._exhaust(dns.resolve("api.example.com"))
        assert ip == "10.0.0.1"
        assert len(values) == 0  # cache hit, no latency
        assert dns.stats.cache_hits == 1

    def test_hierarchical_latency(self):
        records = {"x.com": DNSRecord("x.com", "1.2.3.4")}
        dns, sim = self._make_dns(
            records=records,
            root_latency_s=0.020,
            tld_latency_s=0.015,
            auth_latency_s=0.010,
        )

        values, _ = self._exhaust(dns.resolve("x.com"))
        assert values == [0.020, 0.015, 0.010]

    def test_add_record(self):
        dns, sim = self._make_dns()
        dns.add_record(DNSRecord("new.com", "5.5.5.5"))

        values, ip = self._exhaust(dns.resolve("new.com"))
        assert ip == "5.5.5.5"

    def test_cache_eviction(self):
        records = {
            f"host{i}.com": DNSRecord(f"host{i}.com", f"10.0.0.{i}")
            for i in range(5)
        }
        dns, sim = self._make_dns(records=records, cache_capacity=3)

        # Fill cache
        for i in range(3):
            self._exhaust(dns.resolve(f"host{i}.com"))
        assert dns.cache_size == 3

        # Add a 4th — should evict one
        self._exhaust(dns.resolve("host3.com"))
        assert dns.cache_size == 3
        assert dns.stats.cache_evictions >= 1

    def test_hit_rate(self):
        records = {"x.com": DNSRecord("x.com", "1.1.1.1")}
        dns, sim = self._make_dns(records=records)

        self._exhaust(dns.resolve("x.com"))  # miss
        self._exhaust(dns.resolve("x.com"))  # hit
        self._exhaust(dns.resolve("x.com"))  # hit

        assert dns.stats.hit_rate == pytest.approx(2.0 / 3.0)

    def test_total_resolution_latency(self):
        records = {"x.com": DNSRecord("x.com", "1.1.1.1")}
        dns, sim = self._make_dns(
            records=records,
            root_latency_s=0.020,
            tld_latency_s=0.015,
            auth_latency_s=0.010,
        )

        self._exhaust(dns.resolve("x.com"))
        assert dns.stats.total_resolution_latency_s == pytest.approx(0.045)

    def test_handle_event_is_noop(self):
        dns, sim = self._make_dns()
        from happysimulator.core.event import Event
        event = Event(
            time=Instant.from_seconds(1),
            event_type="Test",
            target=dns,
        )
        result = dns.handle_event(event)
        assert result is None

    def test_repr(self):
        dns, sim = self._make_dns()
        assert "test_dns" in repr(dns)
