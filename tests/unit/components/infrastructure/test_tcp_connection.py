"""Unit tests for TCPConnection."""

import pytest
import random

from happysimulator.components.infrastructure.tcp_connection import (
    TCPConnection,
    TCPStats,
    CongestionControl,
    AIMD,
    Cubic,
    BBR,
)
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant


class TestCongestionControl:
    def test_aimd_slow_start(self):
        cc = AIMD()
        # cwnd < ssthresh -> slow start
        new_cwnd = cc.on_ack(cwnd=4.0, ssthresh=64.0)
        assert new_cwnd == 5.0  # +1 per ack in slow start

    def test_aimd_congestion_avoidance(self):
        cc = AIMD()
        # cwnd >= ssthresh -> linear increase
        new_cwnd = cc.on_ack(cwnd=64.0, ssthresh=64.0)
        assert new_cwnd == pytest.approx(64.0 + 1.0 / 64.0)

    def test_aimd_on_loss(self):
        cc = AIMD()
        new_cwnd, new_ssthresh = cc.on_loss(cwnd=64.0)
        assert new_cwnd == 32.0
        assert new_ssthresh == 32.0

    def test_aimd_loss_floor(self):
        cc = AIMD()
        new_cwnd, _ = cc.on_loss(cwnd=2.0)
        assert new_cwnd >= 2.0

    def test_aimd_name(self):
        assert AIMD().name == "AIMD"

    def test_cubic_slow_start(self):
        cc = Cubic()
        new_cwnd = cc.on_ack(cwnd=4.0, ssthresh=64.0)
        assert new_cwnd == 5.0

    def test_cubic_on_loss(self):
        cc = Cubic(beta=0.7)
        new_cwnd, _ = cc.on_loss(cwnd=100.0)
        assert new_cwnd == pytest.approx(70.0)

    def test_cubic_name(self):
        assert Cubic().name == "Cubic"

    def test_bbr_startup(self):
        cc = BBR()
        new_cwnd = cc.on_ack(cwnd=4.0, ssthresh=64.0)
        assert new_cwnd == 8.0  # doubles in startup

    def test_bbr_on_loss(self):
        cc = BBR()
        new_cwnd, _ = cc.on_loss(cwnd=100.0)
        assert new_cwnd == 90.0

    def test_bbr_name(self):
        assert BBR().name == "BBR"


class TestTCPConnection:
    def _make_tcp(self, **kwargs) -> tuple[TCPConnection, Simulation]:
        tcp = TCPConnection("test_tcp", **kwargs)
        sim = Simulation(
            start_time=Instant.from_seconds(0),
            end_time=Instant.from_seconds(100),
            entities=[tcp],
        )
        return tcp, sim

    def test_creation_defaults(self):
        tcp = TCPConnection("tcp")
        assert tcp.name == "tcp"
        assert tcp.cwnd == 10.0

    def test_stats_initial(self):
        tcp, sim = self._make_tcp()
        stats = tcp.stats
        assert isinstance(stats, TCPStats)
        assert stats.segments_sent == 0
        assert stats.segments_acked == 0
        assert stats.retransmissions == 0
        assert stats.algorithm == "AIMD"

    def test_rtt_positive(self):
        tcp, sim = self._make_tcp(base_rtt_s=0.05)
        assert tcp.rtt_s >= 0.05

    def test_throughput(self):
        tcp, sim = self._make_tcp()
        assert tcp.throughput_segments_per_s > 0

    def test_send_small_data(self):
        random.seed(42)
        tcp, sim = self._make_tcp(loss_rate=0.0)
        gen = tcp.send(1460)  # 1 segment

        values = []
        try:
            while True:
                values.append(next(gen))
        except StopIteration:
            pass

        assert tcp.stats.segments_sent >= 1
        assert tcp.stats.segments_acked >= 1
        assert tcp.stats.total_bytes_sent >= 1460

    def test_send_multi_segment(self):
        random.seed(42)
        tcp, sim = self._make_tcp(loss_rate=0.0)
        gen = tcp.send(14600)  # 10 segments

        values = []
        try:
            while True:
                values.append(next(gen))
        except StopIteration:
            pass

        assert tcp.stats.segments_sent >= 10

    def test_cwnd_grows_without_loss(self):
        random.seed(42)
        tcp, sim = self._make_tcp(loss_rate=0.0)
        initial_cwnd = tcp.cwnd

        gen = tcp.send(146000)  # many segments
        try:
            while True:
                next(gen)
        except StopIteration:
            pass

        assert tcp.cwnd > initial_cwnd

    def test_loss_causes_retransmissions(self):
        random.seed(42)
        tcp, sim = self._make_tcp(loss_rate=0.1)  # 10% loss

        gen = tcp.send(146000)
        try:
            while True:
                next(gen)
        except StopIteration:
            pass

        assert tcp.stats.retransmissions > 0

    def test_cubic_algorithm(self):
        tcp, sim = self._make_tcp(congestion_control=Cubic())
        assert tcp.stats.algorithm == "Cubic"

    def test_bbr_algorithm(self):
        tcp, sim = self._make_tcp(congestion_control=BBR())
        assert tcp.stats.algorithm == "BBR"

    def test_handle_event_is_noop(self):
        tcp, sim = self._make_tcp()
        from happysimulator.core.event import Event
        event = Event(
            time=Instant.from_seconds(1),
            event_type="Test",
            target=tcp,
        )
        result = tcp.handle_event(event)
        assert result is None

    def test_repr(self):
        tcp, sim = self._make_tcp()
        assert "test_tcp" in repr(tcp)
        assert "AIMD" in repr(tcp)
