"""Tests for predefined network condition factory functions."""

import pytest

from happysimulator.components.network.conditions import (
    local_network,
    datacenter_network,
    cross_region_network,
    internet_network,
    satellite_network,
    lossy_network,
    slow_network,
    mobile_3g_network,
    mobile_4g_network,
)
from happysimulator.components.network.link import NetworkLink


class TestNetworkConditionFactories:
    """Tests for network condition factory functions."""

    def test_local_network_characteristics(self):
        """local_network() creates link with expected characteristics."""
        link = local_network()
        assert isinstance(link, NetworkLink)
        assert link.name == "local"
        assert link.bandwidth_bps == 1_000_000_000  # 1 Gbps
        assert link.packet_loss_rate == 0.0
        assert link.jitter is None

    def test_local_network_custom_name(self):
        """local_network() accepts custom name."""
        link = local_network(name="my_local")
        assert link.name == "my_local"

    def test_datacenter_network_characteristics(self):
        """datacenter_network() creates link with expected characteristics."""
        link = datacenter_network()
        assert isinstance(link, NetworkLink)
        assert link.name == "datacenter"
        assert link.bandwidth_bps == 10_000_000_000  # 10 Gbps
        assert link.packet_loss_rate == 0.0
        assert link.jitter is not None

    def test_cross_region_network_characteristics(self):
        """cross_region_network() creates link with expected characteristics."""
        link = cross_region_network()
        assert isinstance(link, NetworkLink)
        assert link.name == "cross_region"
        assert link.bandwidth_bps == 1_000_000_000  # 1 Gbps
        assert link.packet_loss_rate == 0.0001  # 0.01%
        assert link.jitter is not None

    def test_internet_network_characteristics(self):
        """internet_network() creates link with expected characteristics."""
        link = internet_network()
        assert isinstance(link, NetworkLink)
        assert link.name == "internet"
        assert link.bandwidth_bps == 100_000_000  # 100 Mbps
        assert link.packet_loss_rate == 0.001  # 0.1%
        assert link.jitter is not None

    def test_satellite_network_characteristics(self):
        """satellite_network() creates link with expected characteristics."""
        link = satellite_network()
        assert isinstance(link, NetworkLink)
        assert link.name == "satellite"
        assert link.bandwidth_bps == 10_000_000  # 10 Mbps
        assert link.packet_loss_rate == 0.005  # 0.5%
        assert link.jitter is not None

    def test_lossy_network_configurable_loss(self):
        """lossy_network() creates link with specified loss rate."""
        link = lossy_network(loss_rate=0.1)
        assert isinstance(link, NetworkLink)
        assert link.name == "lossy"
        assert link.packet_loss_rate == 0.1

    def test_lossy_network_custom_name_and_latency(self):
        """lossy_network() accepts custom name and latency."""
        link = lossy_network(loss_rate=0.05, name="my_lossy", base_latency=0.050)
        assert link.name == "my_lossy"
        assert link.packet_loss_rate == 0.05

    def test_lossy_network_rejects_invalid_loss_rate(self):
        """lossy_network() rejects loss rate outside [0, 1]."""
        with pytest.raises(ValueError, match="loss_rate"):
            lossy_network(loss_rate=1.5)

        with pytest.raises(ValueError, match="loss_rate"):
            lossy_network(loss_rate=-0.1)

    def test_slow_network_configurable_latency(self):
        """slow_network() creates link with specified latency."""
        link = slow_network(latency_seconds=0.500)
        assert isinstance(link, NetworkLink)
        assert link.name == "slow"
        assert link.packet_loss_rate == 0.0

    def test_slow_network_custom_bandwidth(self):
        """slow_network() accepts custom bandwidth."""
        link = slow_network(latency_seconds=0.100, bandwidth_bps=500_000)
        assert link.bandwidth_bps == 500_000

    def test_mobile_3g_network_characteristics(self):
        """mobile_3g_network() creates link with expected characteristics."""
        link = mobile_3g_network()
        assert isinstance(link, NetworkLink)
        assert link.name == "mobile_3g"
        assert link.bandwidth_bps == 2_000_000  # 2 Mbps
        assert link.packet_loss_rate == 0.005  # 0.5%
        assert link.jitter is not None

    def test_mobile_4g_network_characteristics(self):
        """mobile_4g_network() creates link with expected characteristics."""
        link = mobile_4g_network()
        assert isinstance(link, NetworkLink)
        assert link.name == "mobile_4g"
        assert link.bandwidth_bps == 20_000_000  # 20 Mbps
        assert link.packet_loss_rate == 0.001  # 0.1%
        assert link.jitter is not None


class TestNetworkConditionLatencyOrder:
    """Tests verifying relative latency ordering of network conditions."""

    def test_latency_ordering(self):
        """Network conditions have expected relative latency ordering."""
        from happysimulator.core.temporal import Instant

        local = local_network()
        dc = datacenter_network()
        cross = cross_region_network()
        internet = internet_network()
        satellite = satellite_network()

        # Get base latencies (ignoring jitter for comparison)
        local_lat = local.latency.get_latency(Instant.Epoch).to_seconds()
        dc_lat = dc.latency.get_latency(Instant.Epoch).to_seconds()
        cross_lat = cross.latency.get_latency(Instant.Epoch).to_seconds()
        internet_lat = internet.latency.get_latency(Instant.Epoch).to_seconds()
        satellite_lat = satellite.latency.get_latency(Instant.Epoch).to_seconds()

        # Verify ordering: local < datacenter < cross_region < internet < satellite
        assert local_lat < dc_lat < cross_lat
        assert cross_lat < internet_lat < satellite_lat

    def test_bandwidth_ordering(self):
        """Network conditions have expected relative bandwidth ordering."""
        local = local_network()
        dc = datacenter_network()
        internet = internet_network()
        satellite = satellite_network()
        mobile_3g = mobile_3g_network()

        # Datacenter has highest bandwidth
        assert dc.bandwidth_bps > local.bandwidth_bps
        assert local.bandwidth_bps > internet.bandwidth_bps
        assert internet.bandwidth_bps > satellite.bandwidth_bps
        assert satellite.bandwidth_bps > mobile_3g.bandwidth_bps
