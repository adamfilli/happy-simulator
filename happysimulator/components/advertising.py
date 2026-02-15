"""Advertising economics simulation components.

Models the dynamics of digital advertising platforms, including audience
targeting tiers, advertiser profitability analysis, and the Adverse
Advertising Amplification (AAA) effect.

The AAA effect describes how small changes in consumer sentiment cause
disproportionately large revenue losses for advertising platforms. This
happens because outer-ring (broad, high-CPA) campaigns become unprofitable
first and get shut off, removing the platform's most lucrative revenue streams.

Typical usage::

    platform = AdPlatform("Meta")
    tiers = [
        AudienceTier("Niche", base_monthly_sales=100, base_cpa=10.0),
        AudienceTier("Broad", base_monthly_sales=1000, base_cpa=40.0),
    ]
    advertiser = Advertiser(
        "PosterShop",
        product_price=100.0,
        production_cost=50.0,
        tiers=tiers,
        platform=platform,
    )
    sim = Simulation(entities=[platform, advertiser], end_time=...)
    for e in advertiser.start_events():
        sim.schedule(e)
    sim.run()
"""

from __future__ import annotations

from dataclasses import dataclass

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant
from happysimulator.instrumentation.data import Data


@dataclass(frozen=True)
class AudienceTier:
    """An audience segment with specific advertising economics.

    Represents one concentric ring of advertising reach. Inner rings
    (niche audiences) have high conversion rates and low cost per
    acquisition (CPA). Outer rings (broad audiences) have low conversion
    rates and high CPA.

    The ad spend for a tier is constant regardless of consumer sentiment
    (you reach the same audience at the same cost). When sentiment drops,
    fewer people convert, so CPA rises. When CPA exceeds the profit
    margin, a rational advertiser shuts off the tier entirely.

    Args:
        name: Human-readable tier label.
        base_monthly_sales: Expected sales per period at sentiment=1.0.
        base_cpa: Cost per acquisition at sentiment=1.0.
    """

    name: str
    base_monthly_sales: int
    base_cpa: float

    @property
    def monthly_ad_spend(self) -> float:
        """Fixed monthly ad spend (reach cost is constant)."""
        return self.base_monthly_sales * self.base_cpa

    def effective_cpa(self, sentiment: float) -> float:
        """CPA adjusted for current consumer sentiment."""
        if sentiment <= 0:
            return float("inf")
        return self.base_cpa / sentiment

    def monthly_sales(self, sentiment: float) -> float:
        """Expected sales given current sentiment."""
        return self.base_monthly_sales * sentiment

    def breakeven_sentiment(self, margin: float) -> float:
        """Minimum sentiment at which this tier is profitable.

        Below this value, CPA exceeds margin and a rational advertiser
        would shut off the campaign.
        """
        if margin <= 0:
            return float("inf")
        return self.base_cpa / margin

    def is_profitable(self, sentiment: float, margin: float) -> bool:
        """Whether this tier generates positive profit at given sentiment."""
        return self.effective_cpa(sentiment) < margin

    def tier_profit(self, sentiment: float, margin: float) -> float:
        """Advertiser profit from this tier for one period."""
        if not self.is_profitable(sentiment, margin):
            return 0.0
        sales = self.monthly_sales(sentiment)
        return sales * (margin - self.effective_cpa(sentiment))

    def tier_platform_revenue(self, sentiment: float, margin: float) -> float:
        """Platform revenue from this tier for one period.

        Returns zero if the tier is shut off (unprofitable). When active,
        the platform collects the full fixed ad spend regardless of
        conversion count.
        """
        if not self.is_profitable(sentiment, margin):
            return 0.0
        return self.monthly_ad_spend


@dataclass(frozen=True)
class AdvertiserStats:
    """Aggregate statistics for an advertiser."""

    periods_evaluated: int = 0
    total_profit: float = 0.0
    total_platform_revenue: float = 0.0
    tier_shutoff_events: int = 0


class Advertiser(Entity):
    """A business that sells products through advertising on a platform.

    Manages multiple audience tiers, periodically evaluating profitability
    and shutting off tiers that don't generate positive profit. Reports
    ad revenue to the associated AdPlatform.

    The margin (product_price - production_cost) determines the maximum
    CPA tolerable. As consumer sentiment drops, CPA rises and outer tiers
    become unprofitable first.

    Args:
        name: Entity name.
        product_price: Selling price per unit.
        production_cost: Non-advertising cost per unit.
        tiers: List of AudienceTier objects, from niche to broad.
        platform: The AdPlatform entity to report revenue to.
        evaluation_interval: Simulated seconds between evaluations.
    """

    def __init__(
        self,
        name: str,
        *,
        product_price: float,
        production_cost: float,
        tiers: list[AudienceTier],
        platform: AdPlatform,
        evaluation_interval: float = 1.0,
    ):
        super().__init__(name)
        self.product_price = product_price
        self.production_cost = production_cost
        self.tiers = list(tiers)
        self.platform = platform
        self.evaluation_interval = evaluation_interval
        self.margin = product_price - production_cost

        self._sentiment = 1.0
        self.active_tiers: list[AudienceTier] = list(tiers)
        self._periods_evaluated = 0
        self._total_profit = 0.0
        self._total_platform_revenue = 0.0
        self._tier_shutoff_events = 0

        # Time-series data
        self.profit_data = Data()
        self.platform_revenue_data = Data()
        self.active_tier_data = Data()
        self.sentiment_data = Data()

    @property
    def sentiment(self) -> float:
        return self._sentiment

    @sentiment.setter
    def sentiment(self, value: float):
        self._sentiment = max(0.0, min(1.0, value))

    @property
    def stats(self) -> AdvertiserStats:
        """Return a frozen snapshot of advertiser statistics."""
        return AdvertiserStats(
            periods_evaluated=self._periods_evaluated,
            total_profit=self._total_profit,
            total_platform_revenue=self._total_platform_revenue,
            tier_shutoff_events=self._tier_shutoff_events,
        )

    def start_events(self) -> list[Event]:
        """Generate the initial evaluation event."""
        return [
            Event(
                time=Instant.from_seconds(self.evaluation_interval),
                event_type="EvaluateCampaigns",
                target=self,
            )
        ]

    def handle_event(self, event: Event) -> list[Event] | None:
        if event.event_type == "EvaluateCampaigns":
            return self._evaluate()
        if event.event_type == "SentimentChange":
            meta = event.context.get("metadata", {})
            self.sentiment = meta.get("sentiment", self._sentiment)
            return None
        return None

    def _evaluate(self) -> list[Event]:
        time_s = self.now.to_seconds()

        prev_count = len(self.active_tiers)
        self.active_tiers = [t for t in self.tiers if t.is_profitable(self._sentiment, self.margin)]
        new_count = len(self.active_tiers)

        if new_count < prev_count:
            self._tier_shutoff_events += prev_count - new_count

        monthly_profit = sum(t.tier_profit(self._sentiment, self.margin) for t in self.active_tiers)
        monthly_platform_rev = sum(
            t.tier_platform_revenue(self._sentiment, self.margin) for t in self.active_tiers
        )

        self._periods_evaluated += 1
        self._total_profit += monthly_profit
        self._total_platform_revenue += monthly_platform_rev

        self.profit_data.add_stat(monthly_profit, self.now)
        self.platform_revenue_data.add_stat(monthly_platform_rev, self.now)
        self.active_tier_data.add_stat(new_count, self.now)
        self.sentiment_data.add_stat(self._sentiment, self.now)

        return [
            Event(
                time=self.now,
                event_type="AdRevenue",
                target=self.platform,
                context={
                    "metadata": {
                        "revenue": monthly_platform_rev,
                        "advertiser": self.name,
                        "active_tiers": new_count,
                        "sentiment": self._sentiment,
                    }
                },
            ),
            Event(
                time=Instant.from_seconds(time_s + self.evaluation_interval),
                event_type="EvaluateCampaigns",
                target=self,
            ),
        ]

    def sensitivity_analysis(
        self,
        sentiment_range: tuple[float, float] = (0.0, 1.0),
        steps: int = 100,
    ) -> list[dict]:
        """Static analysis of profit/revenue across sentiment values.

        Returns a list of dicts with keys: sentiment, advertiser_profit,
        platform_revenue, active_tiers, tier_names.
        """
        results = []
        lo, hi = sentiment_range
        for i in range(steps + 1):
            s = lo + (hi - lo) * i / steps
            active = [t for t in self.tiers if t.is_profitable(s, self.margin)]
            profit = sum(t.tier_profit(s, self.margin) for t in active)
            rev = sum(t.tier_platform_revenue(s, self.margin) for t in active)
            results.append(
                {
                    "sentiment": s,
                    "advertiser_profit": profit,
                    "platform_revenue": rev,
                    "active_tiers": len(active),
                    "tier_names": [t.name for t in active],
                }
            )
        return results


@dataclass(frozen=True)
class AdPlatformStats:
    """Aggregate statistics for an ad platform."""

    revenue_events: int = 0
    total_revenue: float = 0.0


class AdPlatform(Entity):
    """An advertising platform that collects revenue from advertisers.

    Passively receives AdRevenue events from Advertiser entities and
    tracks cumulative and per-period revenue.

    Args:
        name: Entity name (e.g., "Meta", "Google").
    """

    def __init__(self, name: str):
        super().__init__(name)
        self._revenue_events = 0
        self._total_revenue = 0.0
        self.revenue_data = Data()

    @property
    def stats(self) -> AdPlatformStats:
        """Return a frozen snapshot of platform statistics."""
        return AdPlatformStats(
            revenue_events=self._revenue_events,
            total_revenue=self._total_revenue,
        )

    def handle_event(self, event: Event) -> None:
        if event.event_type == "AdRevenue":
            meta = event.context.get("metadata", {})
            revenue = meta.get("revenue", 0.0)
            self._revenue_events += 1
            self._total_revenue += revenue
            self.revenue_data.add_stat(revenue, self.now)
