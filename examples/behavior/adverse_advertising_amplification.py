"""Adverse Advertising Amplification (AAA) demonstration.

Simulates the asymmetric impact of consumer sentiment changes on
advertisers vs. advertising platforms. A small drop in consumer
spending causes a disproportionately large revenue loss for the platform.

Based on the AAA essay: .dev/aaa.md

## The Concentric Rings Model

```
                     +-----------------------------------------------+
                     |           Broad Audience (Men 25-34)           |
                     |    CPA: $40  |  1000 sales/mo  |  s_min: 0.80 |
                     |                                               |
                     |       +-------------------------------+       |
                     |       |    Niche (Stock Art Lovers)    |       |
                     |       | CPA: $10 | 100 sales | s: 0.20|       |
                     |       +-------------------------------+       |
                     |                                               |
                     +-----------------------------------------------+

Product: $100 price, $50 production cost, $50 margin
Ad spend is FIXED per tier (you reach the same audience regardless of sentiment).
When sentiment drops, conversions drop, CPA rises, outer rings shut off first.
```

## The Amplification Effect

When consumer sentiment drops by 20% (1.0 -> 0.80):
  - Tier 1 (Niche):  CPA $10 -> $12.50, still profitable, stays active
  - Tier 2 (Broad):  CPA $40 -> $50.00, hits breakeven, SHUT OFF

Result:
  - Advertiser profit:  $14,000 -> $3,000  (-78.6%)
  - Platform revenue:   $41,000 -> $1,000  (-97.6%)

A 20% consumer sentiment drop causes a 97.6% platform revenue drop.
That's the AAA effect.

## Extended Model (4 Tiers)

Adds intermediate tiers to show the cascading shutoff effect as sentiment
gradually decreases. Each tier has a different breakeven sentiment:

  Tier 1 "Core Niche"     :  100 sales,  CPA $10  -> breakeven s=0.20
  Tier 2 "Adjacent Interest": 300 sales,  CPA $25  -> breakeven s=0.50
  Tier 3 "Demographic Match": 1000 sales, CPA $40  -> breakeven s=0.80
  Tier 4 "Broad Audience"  : 2000 sales,  CPA $48  -> breakeven s=0.96
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from happysimulator import (
    Data,
    Event,
    Instant,
    Simulation,
    SimulationSummary,
)
from happysimulator.components.advertising import (
    AdPlatform,
    Advertiser,
    AudienceTier,
)


# =============================================================================
# Scenario definitions
# =============================================================================


ESSAY_TIERS = [
    AudienceTier("Niche (Stock Art)", base_monthly_sales=100, base_cpa=10.0),
    AudienceTier("Broad (Men 25-34)", base_monthly_sales=1000, base_cpa=40.0),
]

EXTENDED_TIERS = [
    AudienceTier("Core Niche", base_monthly_sales=100, base_cpa=10.0),
    AudienceTier("Adjacent Interest", base_monthly_sales=300, base_cpa=25.0),
    AudienceTier("Demographic Match", base_monthly_sales=1000, base_cpa=40.0),
    AudienceTier("Broad Audience", base_monthly_sales=2000, base_cpa=48.0),
]

PRODUCT_PRICE = 100.0
PRODUCTION_COST = 50.0
MARGIN = PRODUCT_PRICE - PRODUCTION_COST


# =============================================================================
# Static analysis (no simulation needed)
# =============================================================================


def print_tier_table(tiers: list[AudienceTier], margin: float) -> None:
    """Print a summary table of tier economics."""
    print(f"  {'Tier':<28s} {'Sales/mo':>10s} {'CPA':>8s} {'Ad Spend':>10s} "
          f"{'Profit':>10s} {'Plat Rev':>10s} {'Breakeven':>10s}")
    print(f"  {'-' * 28} {'-' * 10} {'-' * 8} {'-' * 10} "
          f"{'-' * 10} {'-' * 10} {'-' * 10}")
    for t in tiers:
        print(f"  {t.name:<28s} {t.base_monthly_sales:>10,d} "
              f"${t.base_cpa:>6.0f} ${t.monthly_ad_spend:>8,.0f} "
              f"${t.tier_profit(1.0, margin):>8,.0f} "
              f"${t.tier_platform_revenue(1.0, margin):>8,.0f} "
              f"  s={t.breakeven_sentiment(margin):.2f}")


def analyze_sentiment_shift(
    tiers: list[AudienceTier],
    margin: float,
    before: float,
    after: float,
) -> None:
    """Analyze the impact of a sentiment shift on all parties."""
    # Before
    active_before = [t for t in tiers if t.is_profitable(before, margin)]
    profit_before = sum(t.tier_profit(before, margin) for t in active_before)
    rev_before = sum(t.tier_platform_revenue(before, margin) for t in active_before)

    # After
    active_after = [t for t in tiers if t.is_profitable(after, margin)]
    profit_after = sum(t.tier_profit(after, margin) for t in active_after)
    rev_after = sum(t.tier_platform_revenue(after, margin) for t in active_after)

    sentiment_change = (after - before) / before * 100

    print(f"\n  BEFORE (sentiment = {before:.2f}):")
    for t in tiers:
        if t.is_profitable(before, margin):
            sales = t.monthly_sales(before)
            cpa = t.effective_cpa(before)
            profit = t.tier_profit(before, margin)
            print(f"    {t.name:<28s} {sales:>6.0f} sales, "
                  f"CPA ${cpa:>6.2f}, profit ${profit:>8,.0f}")
        else:
            print(f"    {t.name:<28s}   SHUT OFF")
    print(f"    {'-' * 70}")
    print(f"    Advertiser profit:  ${profit_before:>10,.0f}/mo")
    print(f"    Platform revenue:   ${rev_before:>10,.0f}/mo")

    print(f"\n  AFTER (sentiment = {after:.2f}):")
    for t in tiers:
        if t.is_profitable(after, margin):
            sales = t.monthly_sales(after)
            cpa = t.effective_cpa(after)
            profit = t.tier_profit(after, margin)
            status = ""
            if not t.is_profitable(before, margin):
                status = " <- REACTIVATED"
            print(f"    {t.name:<28s} {sales:>6.0f} sales, "
                  f"CPA ${cpa:>6.2f}, profit ${profit:>8,.0f}{status}")
        else:
            print(f"    {t.name:<28s}   SHUT OFF  "
                  f"(CPA ${t.effective_cpa(after):.2f} >= ${margin:.0f} margin)")
    print(f"    {'-' * 70}")
    print(f"    Advertiser profit:  ${profit_after:>10,.0f}/mo", end="")
    if profit_before > 0:
        pct = (profit_after - profit_before) / profit_before * 100
        print(f"  ({pct:+.1f}%)")
    else:
        print()
    print(f"    Platform revenue:   ${rev_after:>10,.0f}/mo", end="")
    if rev_before > 0:
        pct = (rev_after - rev_before) / rev_before * 100
        print(f"  ({pct:+.1f}%)")
    else:
        print()

    print(f"\n  THE AMPLIFICATION:")
    print(f"    Consumer sentiment:  {sentiment_change:+.1f}%")
    if profit_before > 0:
        print(f"    Advertiser profit:   "
              f"{(profit_after - profit_before) / profit_before * 100:+.1f}%")
    if rev_before > 0:
        print(f"    Platform revenue:    "
              f"{(rev_after - rev_before) / rev_before * 100:+.1f}%")
        if profit_before > 0 and profit_after != profit_before:
            advertiser_pct = abs((profit_after - profit_before) / profit_before * 100)
            platform_pct = abs((rev_after - rev_before) / rev_before * 100)
            if advertiser_pct > 0:
                print(f"    Amplification factor: {platform_pct / advertiser_pct:.1f}x "
                      f"(platform loss is {platform_pct / advertiser_pct:.1f}x "
                      f"worse than advertiser)")


# =============================================================================
# Dynamic simulation
# =============================================================================


@dataclass
class AAAResult:
    """Results from an AAA simulation run."""

    advertiser: Advertiser
    platform: AdPlatform
    summary: SimulationSummary


def run_essay_scenario(duration_months: int = 24) -> AAAResult:
    """Run the 2-tier scenario from the essay.

    Months 1-12:  Normal sentiment (1.0)
    Month 12.5:   Recession hits (sentiment -> 0.80)
    Months 13-24: Reduced sentiment
    """
    platform = AdPlatform("Meta")
    advertiser = Advertiser(
        "PosterShop",
        product_price=PRODUCT_PRICE,
        production_cost=PRODUCTION_COST,
        tiers=ESSAY_TIERS,
        platform=platform,
        evaluation_interval=1.0,  # 1 second = 1 month
    )

    sim = Simulation(
        entities=[platform, advertiser],
        duration=duration_months + 1,
    )

    for e in advertiser.start_events():
        sim.schedule(e)

    # Recession at month 12.5
    sim.schedule(
        Event(
            time=Instant.from_seconds(12.5),
            event_type="SentimentChange",
            target=advertiser,
            context={"metadata": {"sentiment": 0.80}},
        )
    )

    summary = sim.run()
    return AAAResult(advertiser=advertiser, platform=platform, summary=summary)


def run_extended_scenario(duration_months: int = 36) -> AAAResult:
    """Run the 4-tier extended scenario with gradual sentiment changes.

    Months 1-6:   Normal sentiment (1.0)
    Month 6.5:    Mild slowdown (sentiment -> 0.90)
    Month 12.5:   Recession deepens (sentiment -> 0.70)
    Month 18.5:   Trough (sentiment -> 0.40)
    Month 24.5:   Recovery begins (sentiment -> 0.80)
    Month 30.5:   Full recovery (sentiment -> 1.0)
    """
    platform = AdPlatform("AdGiant")
    advertiser = Advertiser(
        "PosterShop",
        product_price=PRODUCT_PRICE,
        production_cost=PRODUCTION_COST,
        tiers=EXTENDED_TIERS,
        platform=platform,
        evaluation_interval=1.0,
    )

    sim = Simulation(
        entities=[platform, advertiser],
        duration=duration_months + 1,
    )

    for e in advertiser.start_events():
        sim.schedule(e)

    # Sentiment schedule
    shifts = [
        (6.5, 0.90),   # Mild slowdown
        (12.5, 0.70),  # Recession
        (18.5, 0.40),  # Deep recession
        (24.5, 0.80),  # Recovery
        (30.5, 1.00),  # Full recovery
    ]
    for time_s, sentiment in shifts:
        sim.schedule(
            Event(
                time=Instant.from_seconds(time_s),
                event_type="SentimentChange",
                target=advertiser,
                context={"metadata": {"sentiment": sentiment}},
            )
        )

    summary = sim.run()
    return AAAResult(advertiser=advertiser, platform=platform, summary=summary)


# =============================================================================
# Visualization
# =============================================================================


def visualize_sensitivity(
    tiers: list[AudienceTier],
    margin: float,
    output_dir: Path,
    filename: str = "aaa_sensitivity.png",
) -> None:
    """Generate a sensitivity chart showing the AAA amplification effect."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    sentiments = [i / 100 for i in range(1, 101)]
    profits = []
    revenues = []
    active_counts = []

    for s in sentiments:
        active = [t for t in tiers if t.is_profitable(s, margin)]
        profits.append(sum(t.tier_profit(s, margin) for t in active))
        revenues.append(sum(t.tier_platform_revenue(s, margin) for t in active))
        active_counts.append(len(active))

    # Normalize to percentage of baseline (sentiment=1.0)
    base_profit = profits[-1] if profits[-1] > 0 else 1
    base_revenue = revenues[-1] if revenues[-1] > 0 else 1
    profit_pct = [p / base_profit * 100 for p in profits]
    revenue_pct = [r / base_revenue * 100 for r in revenues]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    # Top: absolute values
    ax1.plot(sentiments, [p / 1000 for p in profits], "b-", linewidth=2,
             label="Advertiser Profit")
    ax1.plot(sentiments, [r / 1000 for r in revenues], "r-", linewidth=2,
             label="Platform Revenue")
    ax1.set_ylabel("$/month (thousands)")
    ax1.set_title("Adverse Advertising Amplification: Revenue vs. Consumer Sentiment")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Mark breakeven points
    for t in tiers:
        be = t.breakeven_sentiment(margin)
        if 0 < be < 1:
            ax1.axvline(x=be, color="gray", linestyle=":", alpha=0.5)
            ax1.annotate(f"{t.name}\nshuts off",
                         xy=(be, 0), xytext=(be, ax1.get_ylim()[1] * 0.3),
                         ha="center", fontsize=8, color="gray",
                         arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5))

    # Active tier count on secondary axis
    ax1b = ax1.twinx()
    ax1b.fill_between(sentiments, active_counts, alpha=0.1, color="green",
                       step="post")
    ax1b.set_ylabel("Active Tiers", color="green")
    ax1b.tick_params(axis="y", labelcolor="green")
    ax1b.set_ylim(0, len(tiers) + 1)

    # Bottom: percentage of baseline
    ax2.plot(sentiments, profit_pct, "b-", linewidth=2,
             label="Advertiser Profit (% of baseline)")
    ax2.plot(sentiments, revenue_pct, "r-", linewidth=2,
             label="Platform Revenue (% of baseline)")
    ax2.plot(sentiments, [s * 100 for s in sentiments], "k--", linewidth=1,
             alpha=0.5, label="Linear (no amplification)")
    ax2.set_xlabel("Consumer Sentiment")
    ax2.set_ylabel("% of Baseline")
    ax2.set_title("Amplification Effect: % Loss vs. Sentiment Drop")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1.05)
    ax2.set_ylim(-5, 110)

    fig.tight_layout()
    path = output_dir / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def visualize_timeline(result: AAAResult, output_dir: Path,
                       filename: str = "aaa_timeline.png") -> None:
    """Generate a timeline chart from simulation results."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    adv = result.advertiser
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # Sentiment
    ax = axes[0]
    times = adv.sentiment_data.times()
    values = adv.sentiment_data.raw_values()
    ax.plot(times, values, "k-", linewidth=2)
    ax.fill_between(times, values, alpha=0.15, color="blue")
    ax.set_ylabel("Consumer Sentiment")
    ax.set_title("AAA Simulation Timeline")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)

    # Active tiers
    ax = axes[1]
    times = adv.active_tier_data.times()
    values = adv.active_tier_data.raw_values()
    ax.step(times, values, "g-", linewidth=2, where="post")
    ax.fill_between(times, values, alpha=0.15, color="green", step="post")
    ax.set_ylabel("Active Tiers")
    ax.set_ylim(0, len(adv.tiers) + 0.5)
    ax.set_yticks(range(len(adv.tiers) + 1))
    ax.grid(True, alpha=0.3)

    # Advertiser profit
    ax = axes[2]
    times = adv.profit_data.times()
    values = adv.profit_data.raw_values()
    ax.bar(times, [v / 1000 for v in values], width=0.8, alpha=0.7, color="blue")
    ax.set_ylabel("Advertiser Profit\n($k/month)")
    ax.grid(True, alpha=0.3, axis="y")

    # Platform revenue
    ax = axes[3]
    times = adv.platform_revenue_data.times()
    values = adv.platform_revenue_data.raw_values()
    ax.bar(times, [v / 1000 for v in values], width=0.8, alpha=0.7, color="red")
    ax.set_ylabel("Platform Revenue\n($k/month)")
    ax.set_xlabel("Month")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = output_dir / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# Main
# =============================================================================


def print_summary(label: str, result: AAAResult) -> None:
    """Print simulation summary."""
    adv = result.advertiser
    plat = result.platform

    print(f"\n  Simulation: {label}")
    print(f"  Periods evaluated: {adv.stats.periods_evaluated}")
    print(f"  Tier shutoff events: {adv.stats.tier_shutoff_events}")
    print(f"  Total advertiser profit: ${adv.stats.total_profit:,.0f}")
    print(f"  Total platform revenue:  ${plat.stats.total_revenue:,.0f}")
    print(f"  {result.summary}")


def main() -> None:
    print("=" * 74)
    print("  ADVERSE ADVERTISING AMPLIFICATION (AAA)")
    print("  Demonstrating asymmetric impact of consumer sentiment on ad platforms")
    print("=" * 74)

    # -- Static Analysis: Essay Scenario ----------------------------------
    print("\n" + "-" * 74)
    print("  ESSAY SCENARIO (2 tiers)")
    print("-" * 74)
    print(f"\n  Product: ${PRODUCT_PRICE:.0f} price, "
          f"${PRODUCTION_COST:.0f} production cost, ${MARGIN:.0f} margin\n")
    print_tier_table(ESSAY_TIERS, MARGIN)
    analyze_sentiment_shift(ESSAY_TIERS, MARGIN, before=1.0, after=0.80)

    # -- Static Analysis: Extended Scenario -------------------------------
    print("\n\n" + "-" * 74)
    print("  EXTENDED SCENARIO (4 tiers)")
    print("-" * 74)
    print(f"\n  Product: ${PRODUCT_PRICE:.0f} price, "
          f"${PRODUCTION_COST:.0f} production cost, ${MARGIN:.0f} margin\n")
    print_tier_table(EXTENDED_TIERS, MARGIN)
    analyze_sentiment_shift(EXTENDED_TIERS, MARGIN, before=1.0, after=0.90)
    analyze_sentiment_shift(EXTENDED_TIERS, MARGIN, before=1.0, after=0.70)
    analyze_sentiment_shift(EXTENDED_TIERS, MARGIN, before=1.0, after=0.40)

    # -- Dynamic Simulation: Essay Scenario -------------------------------
    print("\n\n" + "-" * 74)
    print("  SIMULATION: Essay Scenario (recession at month 12)")
    print("-" * 74)

    essay_result = run_essay_scenario()
    print_summary("Essay (2-tier)", essay_result)

    # -- Dynamic Simulation: Extended Scenario ----------------------------
    print("\n\n" + "-" * 74)
    print("  SIMULATION: Extended Scenario (gradual recession + recovery)")
    print("-" * 74)

    extended_result = run_extended_scenario()
    print_summary("Extended (4-tier)", extended_result)

    # -- Visualization ----------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("\n  [matplotlib not installed, skipping charts]")
        return

    output_dir = Path("output/aaa")
    print(f"\n\n  Generating visualizations -> {output_dir.absolute()}")

    visualize_sensitivity(ESSAY_TIERS, MARGIN, output_dir,
                          "aaa_essay_sensitivity.png")
    visualize_sensitivity(EXTENDED_TIERS, MARGIN, output_dir,
                          "aaa_extended_sensitivity.png")
    visualize_timeline(essay_result, output_dir,
                       "aaa_essay_timeline.png")
    visualize_timeline(extended_result, output_dir,
                       "aaa_extended_timeline.png")

    print(f"\n  All visualizations saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
