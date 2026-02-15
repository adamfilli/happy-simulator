"""Tests for advertising economics components."""

from __future__ import annotations

import pytest

from happysimulator import Event, Instant, Simulation
from happysimulator.components.advertising import (
    AdPlatform,
    Advertiser,
    AudienceTier,
)

# =============================================================================
# AudienceTier tests
# =============================================================================


class TestAudienceTier:
    """Tests for AudienceTier dataclass."""

    def test_monthly_ad_spend(self):
        tier = AudienceTier("test", base_monthly_sales=100, base_cpa=10.0)
        assert tier.monthly_ad_spend == 1000.0

    def test_effective_cpa_at_baseline(self):
        tier = AudienceTier("test", base_monthly_sales=100, base_cpa=10.0)
        assert tier.effective_cpa(1.0) == 10.0

    def test_effective_cpa_with_reduced_sentiment(self):
        tier = AudienceTier("test", base_monthly_sales=100, base_cpa=10.0)
        assert tier.effective_cpa(0.5) == pytest.approx(20.0)

    def test_effective_cpa_zero_sentiment(self):
        tier = AudienceTier("test", base_monthly_sales=100, base_cpa=10.0)
        assert tier.effective_cpa(0.0) == float("inf")

    def test_monthly_sales_at_baseline(self):
        tier = AudienceTier("test", base_monthly_sales=100, base_cpa=10.0)
        assert tier.monthly_sales(1.0) == 100.0

    def test_monthly_sales_with_reduced_sentiment(self):
        tier = AudienceTier("test", base_monthly_sales=100, base_cpa=10.0)
        assert tier.monthly_sales(0.75) == pytest.approx(75.0)

    def test_breakeven_sentiment(self):
        tier = AudienceTier("test", base_monthly_sales=100, base_cpa=10.0)
        # margin = 50, breakeven = 10/50 = 0.20
        assert tier.breakeven_sentiment(50.0) == pytest.approx(0.20)

    def test_breakeven_sentiment_zero_margin(self):
        tier = AudienceTier("test", base_monthly_sales=100, base_cpa=10.0)
        assert tier.breakeven_sentiment(0.0) == float("inf")

    def test_is_profitable_above_breakeven(self):
        tier = AudienceTier("test", base_monthly_sales=100, base_cpa=10.0)
        assert tier.is_profitable(0.5, 50.0) is True  # breakeven is 0.20

    def test_is_profitable_below_breakeven(self):
        tier = AudienceTier("test", base_monthly_sales=100, base_cpa=10.0)
        assert tier.is_profitable(0.15, 50.0) is False

    def test_is_profitable_at_breakeven(self):
        # At exact breakeven, CPA == margin, so not profitable (strict <)
        tier = AudienceTier("test", base_monthly_sales=100, base_cpa=40.0)
        assert tier.is_profitable(0.80, 50.0) is False

    def test_tier_profit_when_profitable(self):
        tier = AudienceTier("test", base_monthly_sales=100, base_cpa=10.0)
        # sentiment=1.0, margin=50: profit = 100 * (50 - 10) = 4000
        assert tier.tier_profit(1.0, 50.0) == pytest.approx(4000.0)

    def test_tier_profit_when_unprofitable(self):
        tier = AudienceTier("test", base_monthly_sales=100, base_cpa=40.0)
        # sentiment=0.5, CPA=80, margin=50 -> unprofitable
        assert tier.tier_profit(0.5, 50.0) == 0.0

    def test_tier_platform_revenue_when_active(self):
        tier = AudienceTier("test", base_monthly_sales=100, base_cpa=10.0)
        # Ad spend is always 100 * 10 = 1000 when active
        assert tier.tier_platform_revenue(1.0, 50.0) == 1000.0
        assert tier.tier_platform_revenue(0.5, 50.0) == 1000.0

    def test_tier_platform_revenue_when_shutoff(self):
        tier = AudienceTier("test", base_monthly_sales=100, base_cpa=40.0)
        # At sentiment=0.5, CPA=80 > margin=50 -> shut off -> 0
        assert tier.tier_platform_revenue(0.5, 50.0) == 0.0

    def test_frozen_dataclass(self):
        tier = AudienceTier("test", base_monthly_sales=100, base_cpa=10.0)
        with pytest.raises(AttributeError):
            tier.base_cpa = 20.0  # type: ignore[misc]


# =============================================================================
# Essay scenario: verifying exact AAA numbers
# =============================================================================


class TestEssayScenario:
    """Verify the exact numbers from the AAA essay."""

    @pytest.fixture
    def essay_tiers(self):
        return [
            AudienceTier("Niche", base_monthly_sales=100, base_cpa=10.0),
            AudienceTier("Broad", base_monthly_sales=1000, base_cpa=40.0),
        ]

    def test_normal_advertiser_profit(self, essay_tiers):
        margin = 50.0
        total = sum(t.tier_profit(1.0, margin) for t in essay_tiers)
        # Niche: 100*(50-10) = 4000, Broad: 1000*(50-40) = 10000
        assert total == pytest.approx(14000.0)

    def test_normal_platform_revenue(self, essay_tiers):
        margin = 50.0
        total = sum(t.tier_platform_revenue(1.0, margin) for t in essay_tiers)
        # Niche: 1000, Broad: 40000
        assert total == pytest.approx(41000.0)

    def test_recession_tier_shutoff(self, essay_tiers):
        margin = 50.0
        # At sentiment=0.80, Broad CPA = 40/0.80 = 50 >= margin -> shut off
        assert essay_tiers[0].is_profitable(0.80, margin) is True
        assert essay_tiers[1].is_profitable(0.80, margin) is False

    def test_recession_advertiser_profit(self, essay_tiers):
        margin = 50.0
        active = [t for t in essay_tiers if t.is_profitable(0.80, margin)]
        total = sum(t.tier_profit(0.80, margin) for t in active)
        # Only Niche: 80 * (50 - 12.50) = 80 * 37.50 = 3000
        assert total == pytest.approx(3000.0)

    def test_recession_platform_revenue(self, essay_tiers):
        margin = 50.0
        active = [t for t in essay_tiers if t.is_profitable(0.80, margin)]
        total = sum(t.tier_platform_revenue(0.80, margin) for t in active)
        # Only Niche: ad spend = 100 * 10 = 1000
        assert total == pytest.approx(1000.0)

    def test_amplification_effect(self, essay_tiers):
        margin = 50.0

        profit_before = sum(t.tier_profit(1.0, margin) for t in essay_tiers)
        rev_before = sum(t.tier_platform_revenue(1.0, margin) for t in essay_tiers)

        active_after = [t for t in essay_tiers if t.is_profitable(0.80, margin)]
        profit_after = sum(t.tier_profit(0.80, margin) for t in active_after)
        rev_after = sum(t.tier_platform_revenue(0.80, margin) for t in active_after)

        profit_loss_pct = (profit_before - profit_after) / profit_before * 100
        rev_loss_pct = (rev_before - rev_after) / rev_before * 100

        # 20% sentiment drop causes ~78.6% profit loss and ~97.6% revenue loss
        assert profit_loss_pct == pytest.approx(78.57, abs=0.1)
        assert rev_loss_pct == pytest.approx(97.56, abs=0.1)

        # Platform loss is much worse than advertiser loss
        assert rev_loss_pct > profit_loss_pct * 1.2


# =============================================================================
# Advertiser entity tests
# =============================================================================


class TestAdvertiser:
    """Tests for the Advertiser entity."""

    @pytest.fixture
    def setup(self):
        platform = AdPlatform("TestPlatform")
        tiers = [
            AudienceTier("Niche", base_monthly_sales=100, base_cpa=10.0),
            AudienceTier("Broad", base_monthly_sales=1000, base_cpa=40.0),
        ]
        advertiser = Advertiser(
            "TestAdv",
            product_price=100.0,
            production_cost=50.0,
            tiers=tiers,
            platform=platform,
            evaluation_interval=1.0,
        )
        return advertiser, platform

    def test_initial_state(self, setup):
        adv, _ = setup
        assert adv.sentiment == 1.0
        assert len(adv.active_tiers) == 2
        assert adv.margin == 50.0

    def test_sentiment_clamped(self, setup):
        adv, _ = setup
        adv.sentiment = 1.5
        assert adv.sentiment == 1.0
        adv.sentiment = -0.5
        assert adv.sentiment == 0.0

    def test_start_events(self, setup):
        adv, _ = setup
        events = adv.start_events()
        assert len(events) == 1
        assert events[0].event_type == "EvaluateCampaigns"
        assert events[0].target is adv

    def test_simulation_normal(self, setup):
        adv, platform = setup
        sim = Simulation(
            entities=[platform, adv],
            end_time=Instant.from_seconds(4),
        )
        for e in adv.start_events():
            sim.schedule(e)
        sim.run()

        # All evaluations produce the same per-period profit/revenue
        assert adv.stats.periods_evaluated >= 3
        per_period = adv.stats.total_profit / adv.stats.periods_evaluated
        assert per_period == pytest.approx(14000.0)
        assert platform.stats.total_revenue > 0
        per_rev = platform.stats.total_revenue / platform.stats.revenue_events
        assert per_rev == pytest.approx(41000.0)

    def test_simulation_with_recession(self, setup):
        adv, platform = setup
        sim = Simulation(
            entities=[platform, adv],
            end_time=Instant.from_seconds(5),
        )
        for e in adv.start_events():
            sim.schedule(e)

        # Recession at t=2.5 (between month 2 and month 3 evaluations)
        sim.schedule(
            Event(
                time=Instant.from_seconds(2.5),
                event_type="SentimentChange",
                target=adv,
                context={"metadata": {"sentiment": 0.80}},
            )
        )
        sim.run()

        # Verify per-period values via Data objects
        profits = adv.profit_data.raw_values()
        # First 2 evaluations at sentiment=1.0, rest at 0.80
        normal_profits = [p for p in profits if p == pytest.approx(14000.0)]
        recession_profits = [p for p in profits if p == pytest.approx(3000.0)]
        assert len(normal_profits) == 2
        assert len(recession_profits) >= 2

    def test_tier_shutoff_counted(self, setup):
        adv, platform = setup
        sim = Simulation(
            entities=[platform, adv],
            end_time=Instant.from_seconds(3),
        )
        for e in adv.start_events():
            sim.schedule(e)

        sim.schedule(
            Event(
                time=Instant.from_seconds(1.5),
                event_type="SentimentChange",
                target=adv,
                context={"metadata": {"sentiment": 0.80}},
            )
        )
        sim.run()

        # Month 1: 2 tiers active, Month 2: 1 tier (broad shut off)
        assert adv.stats.tier_shutoff_events == 1

    def test_sensitivity_analysis(self, setup):
        adv, _ = setup
        results = adv.sensitivity_analysis(steps=10)
        assert len(results) == 11

        # At sentiment=1.0 (last entry), all tiers active
        assert results[-1]["active_tiers"] == 2
        assert results[-1]["advertiser_profit"] == pytest.approx(14000.0)
        assert results[-1]["platform_revenue"] == pytest.approx(41000.0)

        # At sentiment=0.0 (first entry), no tiers active
        assert results[0]["active_tiers"] == 0
        assert results[0]["advertiser_profit"] == 0.0
        assert results[0]["platform_revenue"] == 0.0

    def test_data_recording(self, setup):
        adv, platform = setup
        sim = Simulation(
            entities=[platform, adv],
            end_time=Instant.from_seconds(3),
        )
        for e in adv.start_events():
            sim.schedule(e)
        sim.run()

        # Advertiser records data each evaluation period
        assert len(adv.profit_data.times()) >= 2
        assert len(adv.platform_revenue_data.times()) >= 2
        assert len(adv.active_tier_data.times()) >= 2
        assert len(adv.sentiment_data.times()) >= 2
        # Platform receives revenue events
        assert len(platform.revenue_data.times()) >= 2


# =============================================================================
# AdPlatform entity tests
# =============================================================================


class TestAdPlatform:
    """Tests for the AdPlatform entity."""

    def test_initial_state(self):
        platform = AdPlatform("Test")
        assert platform.stats.total_revenue == 0.0
        assert platform.stats.revenue_events == 0

    def test_receives_revenue(self):
        platform = AdPlatform("Test")
        adv = Advertiser(
            "Adv",
            product_price=100.0,
            production_cost=50.0,
            tiers=[AudienceTier("T", base_monthly_sales=100, base_cpa=10.0)],
            platform=platform,
        )
        sim = Simulation(
            entities=[platform, adv],
            end_time=Instant.from_seconds(2),
        )
        for e in adv.start_events():
            sim.schedule(e)
        sim.run()

        assert platform.stats.revenue_events >= 1
        # Each revenue event delivers 100 * 10 = 1000 in ad spend
        per_event = platform.stats.total_revenue / platform.stats.revenue_events
        assert per_event == pytest.approx(1000.0)

    def test_ignores_other_events(self):

        platform = AdPlatform("Test")
        sim = Simulation(
            entities=[platform],
            end_time=Instant.from_seconds(2),
        )
        sim.schedule(
            Event(
                time=Instant.from_seconds(1.0),
                event_type="SomeOtherEvent",
                target=platform,
            )
        )
        sim.run()
        assert platform.stats.revenue_events == 0


# =============================================================================
# Multi-advertiser tests
# =============================================================================


class TestMultiAdvertiser:
    """Test that multiple advertisers can share a platform."""

    def test_two_advertisers(self):
        platform = AdPlatform("SharedPlatform")
        tiers1 = [AudienceTier("T1", base_monthly_sales=100, base_cpa=10.0)]
        tiers2 = [AudienceTier("T2", base_monthly_sales=200, base_cpa=20.0)]

        adv1 = Advertiser(
            "Adv1",
            product_price=100.0,
            production_cost=50.0,
            tiers=tiers1,
            platform=platform,
        )
        adv2 = Advertiser(
            "Adv2",
            product_price=100.0,
            production_cost=50.0,
            tiers=tiers2,
            platform=platform,
        )

        sim = Simulation(
            entities=[platform, adv1, adv2],
            end_time=Instant.from_seconds(2),
        )
        for e in adv1.start_events():
            sim.schedule(e)
        for e in adv2.start_events():
            sim.schedule(e)
        sim.run()

        # Platform receives revenue from both advertisers
        assert platform.stats.revenue_events >= 2
        # Adv1: 100*10=1000 per period, Adv2: 200*20=4000 per period
        # Total per period = 5000
        assert platform.stats.total_revenue > 0
        # Verify both advertisers contributed
        assert adv1.stats.periods_evaluated >= 1
        assert adv2.stats.periods_evaluated >= 1
