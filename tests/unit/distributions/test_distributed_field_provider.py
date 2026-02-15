"""Tests for DistributedFieldProvider."""

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant
from happysimulator.distributions import UniformDistribution, ZipfDistribution
from happysimulator.load import DistributedFieldProvider


class DummyTarget(Entity):
    """Simple target entity for testing."""

    def __init__(self, name: str = "target"):
        super().__init__(name)
        self.received_events: list[Event] = []

    def handle_event(self, event: Event) -> list[Event]:
        self.received_events.append(event)
        return []


class TestDistributedFieldProviderCreation:
    """Tests for DistributedFieldProvider creation."""

    def test_creates_with_single_distribution(self):
        """Creates with a single field distribution."""
        target = DummyTarget()
        dist = ZipfDistribution(range(100), s=1.0)

        provider = DistributedFieldProvider(
            target=target,
            event_type="Request",
            field_distributions={"customer_id": dist},
        )

        assert provider.target == target
        assert provider.event_type == "Request"
        assert provider.generated == 0

    def test_creates_with_multiple_distributions(self):
        """Creates with multiple field distributions."""
        target = DummyTarget()

        provider = DistributedFieldProvider(
            target=target,
            event_type="Request",
            field_distributions={
                "customer_id": ZipfDistribution(range(100), s=1.0),
                "region": UniformDistribution(["us", "eu", "ap"]),
            },
        )

        assert provider.event_type == "Request"

    def test_creates_with_static_fields(self):
        """Creates with static fields."""
        target = DummyTarget()

        provider = DistributedFieldProvider(
            target=target,
            event_type="Request",
            field_distributions={"customer_id": ZipfDistribution(range(100))},
            static_fields={"source": "api", "version": "v2"},
        )

        assert provider.generated == 0

    def test_creates_with_stop_after(self):
        """Creates with stop_after time."""
        target = DummyTarget()

        provider = DistributedFieldProvider(
            target=target,
            event_type="Request",
            field_distributions={"customer_id": ZipfDistribution(range(100))},
            stop_after=Instant.from_seconds(10.0),
        )

        assert provider.generated == 0


class TestDistributedFieldProviderGetEvents:
    """Tests for get_events method."""

    def test_get_events_returns_single_event(self):
        """get_events returns a single event."""
        target = DummyTarget()
        dist = ZipfDistribution(range(100), s=1.0, seed=42)

        provider = DistributedFieldProvider(
            target=target,
            event_type="Request",
            field_distributions={"customer_id": dist},
        )

        events = provider.get_events(Instant.from_seconds(1.0))

        assert len(events) == 1
        assert events[0].event_type == "Request"
        assert events[0].target == target

    def test_get_events_includes_sampled_field(self):
        """get_events includes the sampled field in context."""
        target = DummyTarget()
        dist = ZipfDistribution(range(100), s=1.0, seed=42)

        provider = DistributedFieldProvider(
            target=target,
            event_type="Request",
            field_distributions={"customer_id": dist},
        )

        events = provider.get_events(Instant.from_seconds(1.0))

        assert "customer_id" in events[0].context
        assert events[0].context["customer_id"] in range(100)

    def test_get_events_includes_multiple_fields(self):
        """get_events includes all distributed fields."""
        target = DummyTarget()

        provider = DistributedFieldProvider(
            target=target,
            event_type="Request",
            field_distributions={
                "customer_id": ZipfDistribution(range(100), seed=42),
                "region": UniformDistribution(["us", "eu", "ap"], seed=42),
            },
        )

        events = provider.get_events(Instant.from_seconds(1.0))

        assert "customer_id" in events[0].context
        assert "region" in events[0].context
        assert events[0].context["region"] in ["us", "eu", "ap"]

    def test_get_events_includes_static_fields(self):
        """get_events includes static fields in context."""
        target = DummyTarget()

        provider = DistributedFieldProvider(
            target=target,
            event_type="Request",
            field_distributions={"customer_id": ZipfDistribution(range(100))},
            static_fields={"source": "api", "version": "v2"},
        )

        events = provider.get_events(Instant.from_seconds(1.0))

        assert events[0].context["source"] == "api"
        assert events[0].context["version"] == "v2"

    def test_get_events_includes_created_at(self):
        """get_events includes created_at timestamp."""
        target = DummyTarget()

        provider = DistributedFieldProvider(
            target=target,
            event_type="Request",
            field_distributions={"customer_id": ZipfDistribution(range(100))},
        )

        time = Instant.from_seconds(5.0)
        events = provider.get_events(time)

        assert events[0].context["created_at"] == time

    def test_get_events_increments_generated(self):
        """get_events increments the generated counter."""
        target = DummyTarget()

        provider = DistributedFieldProvider(
            target=target,
            event_type="Request",
            field_distributions={"customer_id": ZipfDistribution(range(100))},
        )

        assert provider.generated == 0

        provider.get_events(Instant.from_seconds(1.0))
        assert provider.generated == 1

        provider.get_events(Instant.from_seconds(2.0))
        assert provider.generated == 2

    def test_get_events_stops_after_time(self):
        """get_events returns empty list after stop_after time."""
        target = DummyTarget()

        provider = DistributedFieldProvider(
            target=target,
            event_type="Request",
            field_distributions={"customer_id": ZipfDistribution(range(100))},
            stop_after=Instant.from_seconds(5.0),
        )

        # Before stop_after
        events = provider.get_events(Instant.from_seconds(4.0))
        assert len(events) == 1

        # At stop_after (should still work)
        events = provider.get_events(Instant.from_seconds(5.0))
        assert len(events) == 1

        # After stop_after
        events = provider.get_events(Instant.from_seconds(5.1))
        assert len(events) == 0

    def test_samples_independently_each_call(self):
        """Each call samples independently (may return same or different values)."""
        target = DummyTarget()
        dist = ZipfDistribution(range(1000), s=1.0, seed=42)

        provider = DistributedFieldProvider(
            target=target,
            event_type="Request",
            field_distributions={"customer_id": dist},
        )

        # Generate many events and verify we get variety
        customer_ids = set()
        for i in range(100):
            events = provider.get_events(Instant.from_seconds(float(i)))
            customer_ids.add(events[0].context["customer_id"])

        # With Zipf, we should see some repeated values but also variety
        assert len(customer_ids) > 1  # Not all the same


class TestDistributedFieldProviderRepr:
    """Tests for repr."""

    def test_repr_shows_event_type_and_fields(self):
        """repr shows event type and field names."""
        target = DummyTarget()

        provider = DistributedFieldProvider(
            target=target,
            event_type="Request",
            field_distributions={
                "customer_id": ZipfDistribution(range(100)),
                "region": UniformDistribution(["us", "eu"]),
            },
        )

        r = repr(provider)

        assert "Request" in r
        assert "customer_id" in r
        assert "region" in r
