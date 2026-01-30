"""Exponential distribution fitted to percentile targets.

PercentileFittedLatency constructs an exponential distribution by fitting
to user-provided percentile values (p50, p90, p99, p999, p9999). This allows
modeling latency from observed percentile data.
"""

import logging
import math
import random
from dataclasses import dataclass

from happysimulator.core.instant import Instant
from happysimulator.distributions.latency_distribution import LatencyDistribution

logger = logging.getLogger(__name__)


@dataclass
class PercentileTarget:
    """A target percentile value for curve fitting.

    Attributes:
        percentile: The percentile as a fraction (e.g., 0.50 for p50).
        value: The target latency value in seconds at this percentile.
    """
    percentile: float
    value: float


class PercentileFittedLatency(LatencyDistribution):
    """Exponential latency distribution fitted to percentile targets.

    Fits an exponential distribution to match provided percentile values
    using least-squares optimization. At least one percentile must be
    provided.

    For an exponential distribution with rate λ:
    - CDF: F(x) = 1 - e^(-λx)
    - Quantile: Q(p) = -ln(1-p) / λ
    - Mean: 1/λ

    The fitting minimizes the squared error between the target percentile
    values and the exponential quantile function.

    Args:
        p50: Optional target value for 50th percentile (median).
        p90: Optional target value for 90th percentile.
        p99: Optional target value for 99th percentile.
        p999: Optional target value for 99.9th percentile.
        p9999: Optional target value for 99.99th percentile.

    Raises:
        ValueError: If no percentiles are provided.
    """

    def __init__(
        self,
        p50: Instant | None = None,
        p90: Instant | None = None,
        p99: Instant | None = None,
        p999: Instant | None = None,
        p9999: Instant | None = None,
    ):
        """Initialize by fitting an exponential to the provided percentiles."""
        targets = self._build_targets(p50, p90, p99, p999, p9999)
        if not targets:
            raise ValueError("At least one percentile must be provided")

        self._lambda = self._fit_exponential(targets)
        mean_latency = 1.0 / self._lambda

        super().__init__(Instant.from_seconds(mean_latency))

        logger.debug(
            "PercentileFittedLatency created: lambda=%.6f mean=%.6fs from %d percentile(s)",
            self._lambda,
            self._mean_latency,
            len(targets),
        )

    def _build_targets(
        self,
        p50: Instant | None,
        p90: Instant | None,
        p99: Instant | None,
        p999: Instant | None,
        p9999: Instant | None,
    ) -> list[PercentileTarget]:
        """Convert provided percentile Instants to PercentileTarget objects."""
        targets = []
        if p50 is not None:
            targets.append(PercentileTarget(0.50, p50.to_seconds()))
        if p90 is not None:
            targets.append(PercentileTarget(0.90, p90.to_seconds()))
        if p99 is not None:
            targets.append(PercentileTarget(0.99, p99.to_seconds()))
        if p999 is not None:
            targets.append(PercentileTarget(0.999, p999.to_seconds()))
        if p9999 is not None:
            targets.append(PercentileTarget(0.9999, p9999.to_seconds()))
        return targets

    def _fit_exponential(self, targets: list[PercentileTarget]) -> float:
        """Fit exponential rate parameter λ to minimize percentile error.

        For an exponential distribution, the quantile function is:
            Q(p) = -ln(1-p) / λ

        Rearranging for λ given a target value v at percentile p:
            λ = -ln(1-p) / v

        With multiple targets, we use weighted least squares. Each target
        implies a λ value; we find the λ that minimizes total squared error.

        The optimal λ in least squares sense is:
            λ = Σ[(-ln(1-p_i))²] / Σ[(-ln(1-p_i)) * v_i]

        Args:
            targets: List of percentile targets to fit.

        Returns:
            Fitted rate parameter λ.
        """
        # For each target, compute the coefficient c_i = -ln(1 - p_i)
        # The quantile equation is: v_i = c_i / λ, or c_i = λ * v_i
        # We want to minimize: Σ (c_i - λ * v_i)²
        # Taking derivative and setting to zero:
        # λ = Σ(c_i * v_i) / Σ(v_i²)

        sum_cv = 0.0
        sum_vv = 0.0

        for target in targets:
            c = -math.log(1.0 - target.percentile)
            v = target.value
            sum_cv += c * v
            sum_vv += v * v

        fitted_lambda = sum_cv / sum_vv

        if logger.isEnabledFor(logging.DEBUG):
            for target in targets:
                c = -math.log(1.0 - target.percentile)
                implied_lambda = c / target.value
                fitted_value = c / fitted_lambda
                logger.debug(
                    "  p%.2f: target=%.6fs implied_λ=%.6f fitted_value=%.6fs error=%.2f%%",
                    target.percentile * 100,
                    target.value,
                    implied_lambda,
                    fitted_value,
                    100 * (fitted_value - target.value) / target.value,
                )

        return fitted_lambda

    def get_latency(self, current_time: Instant) -> Instant:
        """Sample a random latency from the fitted exponential distribution."""
        sample = random.expovariate(self._lambda)
        logger.debug(
            "PercentileFittedLatency sampled: %.6fs (mean=%.6fs)",
            sample,
            self._mean_latency,
        )
        return Instant.from_seconds(sample)

    def get_percentile(self, p: float) -> Instant:
        """Get the latency value at a given percentile.

        Args:
            p: Percentile as a fraction (0 < p < 1).

        Returns:
            Latency value at the given percentile.
        """
        value = -math.log(1.0 - p) / self._lambda
        return Instant.from_seconds(value)
