"""TCP connection model with congestion control.

Models TCP-like transport between entities with congestion control
algorithms that manage throughput based on network conditions. Includes
slow start, congestion avoidance, and retransmission logic.

Congestion control algorithms:
- AIMD: Additive Increase / Multiplicative Decrease (classic Reno).
- Cubic: Cubic function-based window growth (Linux default).
- BBR: Bottleneck Bandwidth and RTT estimation.
"""

from __future__ import annotations

import logging
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generator

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Congestion control algorithms
# ---------------------------------------------------------------------------


class CongestionControl(ABC):
    """Strategy for TCP congestion window management."""

    @abstractmethod
    def on_ack(self, cwnd: float, ssthresh: float) -> float:
        """Update cwnd after receiving an ACK.

        Args:
            cwnd: Current congestion window (segments).
            ssthresh: Slow start threshold.

        Returns:
            New cwnd value.
        """
        ...

    @abstractmethod
    def on_loss(self, cwnd: float) -> tuple[float, float]:
        """Update cwnd and ssthresh after packet loss.

        Returns:
            Tuple of (new_cwnd, new_ssthresh).
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class AIMD(CongestionControl):
    """Additive Increase / Multiplicative Decrease (TCP Reno).

    Args:
        additive_increase: Segments added per RTT in congestion avoidance (default 1).
        multiplicative_decrease: Factor to multiply cwnd on loss (default 0.5).
    """

    def __init__(
        self,
        *,
        additive_increase: float = 1.0,
        multiplicative_decrease: float = 0.5,
    ) -> None:
        self._ai = additive_increase
        self._md = multiplicative_decrease

    def on_ack(self, cwnd: float, ssthresh: float) -> float:
        if cwnd < ssthresh:
            # Slow start: exponential growth
            return cwnd + 1.0
        # Congestion avoidance: linear growth
        return cwnd + self._ai / cwnd

    def on_loss(self, cwnd: float) -> tuple[float, float]:
        new_ssthresh = max(cwnd * self._md, 2.0)
        return new_ssthresh, new_ssthresh

    @property
    def name(self) -> str:
        return "AIMD"


class Cubic(CongestionControl):
    """CUBIC congestion control (Linux default).

    Uses a cubic function for window growth, providing better
    performance on high-bandwidth, high-latency networks.

    Args:
        beta: Multiplicative decrease factor (default 0.7).
        c: Cubic scaling constant (default 0.4).
    """

    def __init__(self, *, beta: float = 0.7, c: float = 0.4) -> None:
        self._beta = beta
        self._c = c
        self._w_max: float = 0.0
        self._t_epoch: float = 0.0
        self._ack_count: int = 0

    def on_ack(self, cwnd: float, ssthresh: float) -> float:
        if cwnd < ssthresh:
            return cwnd + 1.0

        self._ack_count += 1
        t = self._ack_count / max(cwnd, 1.0)  # approximate time in RTTs

        # CUBIC function: W(t) = C * (t - K)^3 + W_max
        k = ((self._w_max * (1.0 - self._beta)) / self._c) ** (1.0 / 3.0)
        w_cubic = self._c * (t - k) ** 3 + self._w_max

        # TCP-friendly region
        w_tcp = self._w_max * self._beta + (3.0 * (1.0 - self._beta) / (1.0 + self._beta)) * t

        return max(cwnd + 1.0 / cwnd, max(w_cubic, w_tcp))

    def on_loss(self, cwnd: float) -> tuple[float, float]:
        self._w_max = cwnd
        self._ack_count = 0
        new_cwnd = max(cwnd * self._beta, 2.0)
        return new_cwnd, new_cwnd

    @property
    def name(self) -> str:
        return "Cubic"


class BBR(CongestionControl):
    """Bottleneck Bandwidth and Round-trip propagation time.

    Estimates bottleneck bandwidth and min RTT to set cwnd optimally.
    Simplified model: paces sending rate based on estimated BDP.

    Args:
        gain: Pacing gain during steady state (default 1.0).
        drain_gain: Gain during drain phase (default 0.75).
    """

    def __init__(self, *, gain: float = 1.0, drain_gain: float = 0.75) -> None:
        self._gain = gain
        self._drain_gain = drain_gain
        self._btl_bw: float = 1.0
        self._min_rtt: float = float("inf")
        self._phase: str = "startup"
        self._ack_count: int = 0

    def on_ack(self, cwnd: float, ssthresh: float) -> float:
        self._ack_count += 1

        if self._phase == "startup":
            new_cwnd = cwnd * 2.0
            if new_cwnd > ssthresh and ssthresh > 0:
                self._phase = "drain"
            return new_cwnd

        if self._phase == "drain":
            new_cwnd = cwnd * self._drain_gain
            if new_cwnd <= cwnd * self._gain:
                self._phase = "probe_bw"
            return max(new_cwnd, 2.0)

        # probe_bw: stable operation
        return cwnd + self._gain / cwnd

    def on_loss(self, cwnd: float) -> tuple[float, float]:
        # BBR doesn't react to loss the same way
        # Slight reduction but maintains BDP estimate
        new_cwnd = max(cwnd * 0.9, 2.0)
        return new_cwnd, new_cwnd

    @property
    def name(self) -> str:
        return "BBR"


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TCPStats:
    """Frozen snapshot of TCP connection statistics.

    Attributes:
        segments_sent: Total segments transmitted.
        segments_acked: Total acknowledgements received.
        retransmissions: Total segments retransmitted.
        cwnd: Current congestion window (segments).
        ssthresh: Current slow start threshold.
        rtt_s: Current estimated round-trip time.
        throughput_segments_per_s: Current throughput estimate.
        total_bytes_sent: Total bytes transmitted.
        algorithm: Name of the congestion control algorithm.
    """

    segments_sent: int = 0
    segments_acked: int = 0
    retransmissions: int = 0
    cwnd: float = 0.0
    ssthresh: float = 0.0
    rtt_s: float = 0.0
    throughput_segments_per_s: float = 0.0
    total_bytes_sent: int = 0
    algorithm: str = ""


# ---------------------------------------------------------------------------
# TCPConnection entity
# ---------------------------------------------------------------------------


class TCPConnection(Entity):
    """TCP-like transport with congestion control.

    Models a TCP connection between two endpoints with configurable
    congestion control. Provides ``send()`` to transmit data, which
    yields appropriate delays based on cwnd and RTT.

    Args:
        name: Entity name.
        congestion_control: Congestion control algorithm. Defaults to AIMD.
        base_rtt_s: Base round-trip time (default 50ms).
        loss_rate: Probability of packet loss per segment (default 0.001).
        mss_bytes: Maximum segment size in bytes (default 1460).
        initial_cwnd: Initial congestion window in segments (default 10).
        initial_ssthresh: Initial slow-start threshold (default 64).
        retransmit_timeout_s: Retransmission timeout (default 1.0s).

    Example::

        tcp = TCPConnection("conn", congestion_control=Cubic(), base_rtt_s=0.02)
        sim = Simulation(entities=[tcp, ...], ...)

        # In another entity's handle_event:
        yield from tcp.send(65536)  # send 64KB
    """

    def __init__(
        self,
        name: str,
        *,
        congestion_control: CongestionControl | None = None,
        base_rtt_s: float = 0.05,
        loss_rate: float = 0.001,
        mss_bytes: int = 1460,
        initial_cwnd: float = 10.0,
        initial_ssthresh: float = 64.0,
        retransmit_timeout_s: float = 1.0,
    ) -> None:
        super().__init__(name)
        self._cc = congestion_control or AIMD()
        self._base_rtt_s = base_rtt_s
        self._loss_rate = loss_rate
        self._mss = mss_bytes
        self._cwnd = initial_cwnd
        self._ssthresh = initial_ssthresh
        self._rto_s = retransmit_timeout_s

        # Stats
        self._segments_sent: int = 0
        self._segments_acked: int = 0
        self._retransmissions: int = 0
        self._total_bytes_sent: int = 0

    @property
    def cwnd(self) -> float:
        """Current congestion window in segments."""
        return self._cwnd

    @property
    def rtt_s(self) -> float:
        """Current estimated RTT in seconds."""
        # RTT increases under congestion (queuing delay)
        queuing = 0.001 * self._cwnd / max(self._ssthresh, 1.0)
        return self._base_rtt_s + queuing

    @property
    def throughput_segments_per_s(self) -> float:
        """Estimated throughput in segments per second."""
        rtt = self.rtt_s
        return self._cwnd / rtt if rtt > 0 else 0.0

    @property
    def stats(self) -> TCPStats:
        """Frozen snapshot of TCP connection statistics."""
        return TCPStats(
            segments_sent=self._segments_sent,
            segments_acked=self._segments_acked,
            retransmissions=self._retransmissions,
            cwnd=self._cwnd,
            ssthresh=self._ssthresh,
            rtt_s=self.rtt_s,
            throughput_segments_per_s=self.throughput_segments_per_s,
            total_bytes_sent=self._total_bytes_sent,
            algorithm=self._cc.name,
        )

    def send(self, size_bytes: int) -> Generator[float, None, None]:
        """Send data over the TCP connection.

        Segments data into MSS-sized chunks, applies congestion control,
        and yields appropriate delays. Simulates loss and retransmission.

        Args:
            size_bytes: Total bytes to send.
        """
        segments = math.ceil(size_bytes / self._mss)
        sent = 0

        while sent < segments:
            # Send a window's worth of segments
            window = min(int(self._cwnd), segments - sent)

            for _ in range(window):
                self._segments_sent += 1
                self._total_bytes_sent += self._mss

                if random.random() < self._loss_rate:
                    # Packet loss — retransmit
                    self._retransmissions += 1
                    self._cwnd, self._ssthresh = self._cc.on_loss(self._cwnd)
                    yield self._rto_s  # retransmission timeout
                    self._segments_sent += 1
                    self._total_bytes_sent += self._mss
                else:
                    # Successful ACK
                    self._segments_acked += 1
                    self._cwnd = self._cc.on_ack(self._cwnd, self._ssthresh)

                sent += 1
                if sent >= segments:
                    break

            # Yield RTT for this window
            yield self.rtt_s

    def handle_event(self, event: Event) -> None:
        """TCPConnection does not process events directly."""
        pass

    def __repr__(self) -> str:
        return (
            f"TCPConnection('{self.name}', cc={self._cc.name}, "
            f"cwnd={self._cwnd:.1f}, rtt={self.rtt_s:.4f}s)"
        )
