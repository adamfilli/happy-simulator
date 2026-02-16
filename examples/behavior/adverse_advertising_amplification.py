"""Adverse Advertising Amplification (AAA) demonstration.

Simulates the asymmetric impact of consumer sentiment changes on
advertisers vs. advertising platforms. A small drop in consumer
spending causes a disproportionately large revenue loss for the platform.

Based on the AAA essay: .dev/aaa.md

## The Concentric Rings Model

```
                     +-----------------------------------------------+
                     |           Broad Audience (Men 25-34)           |
                     |    CPA: $40  |  600 sales/mo   |  s_min: 0.80 |
                     |                                               |
                     |       +-------------------------------+       |
                     |       |    Niche (Stock Art Lovers)    |       |
                     |       | CPA: $10 | 400 sales | s: 0.20|       |
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
  - Advertiser profit:  $22,000 -> $12,000  (-45.5%)
  - Platform revenue:   $28,000 -> $4,000   (-85.7%)

A 20% consumer sentiment drop causes an 85.7% platform revenue drop.
That's the AAA effect.

## Extended Model (4 Tiers)

Adds intermediate tiers to show the cascading shutoff effect as sentiment
gradually decreases. Each tier has a different breakeven sentiment:

  Tier 1 "Core Niche"     :  300 sales,  CPA $10  -> breakeven s=0.20
  Tier 2 "Adjacent Interest": 500 sales,  CPA $20  -> breakeven s=0.40
  Tier 3 "Demographic Match": 700 sales,  CPA $35  -> breakeven s=0.70
  Tier 4 "Broad Audience"  : 900 sales,  CPA $45  -> breakeven s=0.90
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from happysimulator import (
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
    AudienceTier("Niche (Stock Art)", base_monthly_sales=400, base_cpa=10.0),
    AudienceTier("Broad (Men 25-34)", base_monthly_sales=600, base_cpa=40.0),
]

EXTENDED_TIERS = [
    AudienceTier("Core Niche", base_monthly_sales=300, base_cpa=10.0),
    AudienceTier("Adjacent Interest", base_monthly_sales=500, base_cpa=20.0),
    AudienceTier("Demographic Match", base_monthly_sales=700, base_cpa=35.0),
    AudienceTier("Broad Audience", base_monthly_sales=900, base_cpa=45.0),
]

PRODUCT_PRICE = 100.0
PRODUCTION_COST = 50.0
MARGIN = PRODUCT_PRICE - PRODUCTION_COST


# =============================================================================
# Static analysis (no simulation needed)
# =============================================================================


def print_tier_table(tiers: list[AudienceTier], margin: float) -> None:
    """Print a summary table of tier economics."""
    print(
        f"  {'Tier':<28s} {'Sales/mo':>10s} {'CPA':>8s} {'Ad Spend':>10s} "
        f"{'Profit':>10s} {'Plat Rev':>10s} {'Breakeven':>10s}"
    )
    print(f"  {'-' * 28} {'-' * 10} {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")
    for t in tiers:
        print(
            f"  {t.name:<28s} {t.base_monthly_sales:>10,d} "
            f"${t.base_cpa:>6.0f} ${t.monthly_ad_spend:>8,.0f} "
            f"${t.tier_profit(1.0, margin):>8,.0f} "
            f"${t.tier_platform_revenue(1.0, margin):>8,.0f} "
            f"  s={t.breakeven_sentiment(margin):.2f}"
        )


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
            print(
                f"    {t.name:<28s} {sales:>6.0f} sales, CPA ${cpa:>6.2f}, profit ${profit:>8,.0f}"
            )
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
            print(
                f"    {t.name:<28s} {sales:>6.0f} sales, "
                f"CPA ${cpa:>6.2f}, profit ${profit:>8,.0f}{status}"
            )
        else:
            print(
                f"    {t.name:<28s}   SHUT OFF  "
                f"(CPA ${t.effective_cpa(after):.2f} >= ${margin:.0f} margin)"
            )
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

    print("\n  THE AMPLIFICATION:")
    print(f"    Consumer sentiment:  {sentiment_change:+.1f}%")
    if profit_before > 0:
        print(
            f"    Advertiser profit:   {(profit_after - profit_before) / profit_before * 100:+.1f}%"
        )
    if rev_before > 0:
        print(f"    Platform revenue:    {(rev_after - rev_before) / rev_before * 100:+.1f}%")
        if profit_before > 0 and profit_after != profit_before:
            advertiser_pct = abs((profit_after - profit_before) / profit_before * 100)
            platform_pct = abs((rev_after - rev_before) / rev_before * 100)
            if advertiser_pct > 0:
                print(
                    f"    Amplification factor: {platform_pct / advertiser_pct:.1f}x "
                    f"(platform loss is {platform_pct / advertiser_pct:.1f}x "
                    f"worse than advertiser)"
                )


# =============================================================================
# Smooth sentiment curve
# =============================================================================


def _cosine_interp(t: float, t_start: float, t_end: float, v_start: float, v_end: float) -> float:
    """Cosine interpolation between two values (smooth acceleration/deceleration)."""
    if t <= t_start:
        return v_start
    if t >= t_end:
        return v_end
    progress = (t - t_start) / (t_end - t_start)
    return v_start + (v_end - v_start) * (1 - math.cos(math.pi * progress)) / 2


def _schedule_smooth_sentiment(
    sim: Simulation,
    advertiser: Advertiser,
    t_start: float,
    t_end: float,
    s_start: float,
    s_end: float,
    step: float = 0.1,
) -> None:
    """Schedule frequent SentimentChange events along a smooth cosine curve."""
    t = t_start
    while t <= t_end:
        s = _cosine_interp(t, t_start, t_end, s_start, s_end)
        sim.schedule(
            Event(
                time=Instant.from_seconds(t),
                event_type="SentimentChange",
                target=advertiser,
                context={"metadata": {"sentiment": s}},
            )
        )
        t += step


# =============================================================================
# Dynamic simulation
# =============================================================================


@dataclass
class AAAScenario:
    """A prepared scenario (simulation built but not yet run)."""

    sim: Simulation
    advertiser: Advertiser
    platform: AdPlatform
    tiers: list[AudienceTier]


@dataclass
class AAAResult:
    """Results from an AAA simulation run."""

    advertiser: Advertiser
    platform: AdPlatform
    summary: SimulationSummary


def build_essay_scenario(duration_months: int = 24) -> AAAScenario:
    """Build the 2-tier scenario from the essay (without running).

    Months 1-6:   Normal sentiment (1.0)
    Months 6-20:  Smooth decline (1.0 -> 0.60)
    Months 20-24: Steady at 0.60
    """
    platform = AdPlatform("Meta")
    advertiser = Advertiser(
        "SellerShop",
        product_price=PRODUCT_PRICE,
        production_cost=PRODUCTION_COST,
        tiers=ESSAY_TIERS,
        platform=platform,
        evaluation_interval=1.0,
    )

    sim = Simulation(
        entities=[platform, advertiser],
        duration=duration_months + 1,
    )

    for e in advertiser.start_events():
        sim.schedule(e)

    _schedule_smooth_sentiment(sim, advertiser, t_start=6, t_end=20, s_start=1.0, s_end=0.60)

    return AAAScenario(sim=sim, advertiser=advertiser, platform=platform, tiers=ESSAY_TIERS)


def build_extended_scenario(duration_months: int = 36) -> AAAScenario:
    """Build the 4-tier extended scenario (without running).

    Months 1-6:   Normal sentiment (1.0)
    Months 6-30:  Smooth decline (1.0 -> 0.10)
    Months 30-36: Steady at 0.10
    """
    platform = AdPlatform("AdGiant")
    advertiser = Advertiser(
        "SellerShop",
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

    _schedule_smooth_sentiment(sim, advertiser, t_start=6, t_end=30, s_start=1.0, s_end=0.10)

    return AAAScenario(sim=sim, advertiser=advertiser, platform=platform, tiers=EXTENDED_TIERS)


def run_scenario(scenario: AAAScenario) -> AAAResult:
    """Run a prepared scenario and return results."""
    summary = scenario.sim.run()
    return AAAResult(
        advertiser=scenario.advertiser,
        platform=scenario.platform,
        summary=summary,
    )


# =============================================================================
# Visualization
# =============================================================================

# Regime background colors (number of active tiers -> color)
_REGIME_COLORS = {
    0: "#ef9a9a",  # red
    1: "#ffcc80",  # orange
    2: "#fff176",  # yellow
    3: "#a5d6a7",  # green
    4: "#81c784",  # dark green
}

_REGIME_LABELS = {
    0: "0 tiers",
    1: "1 tier",
    2: "2 tiers",
    3: "3 tiers",
    4: "4 tiers",
}


def _add_regime_background(
    ax,
    times: list[float],
    tier_counts: list[float],
    max_tiers: int,
    x_max: float,
) -> None:
    """Draw colored vertical bands showing the advertising regime (active tier count)."""
    for i in range(len(times)):
        count = int(tier_counts[i])
        color = _REGIME_COLORS.get(count, _REGIME_COLORS[0])
        x_start = times[i]
        x_end = times[i + 1] if i + 1 < len(times) else x_max
        ax.axvspan(x_start, x_end, alpha=0.35, color=color, linewidth=0, zorder=0)


def _add_regime_legend(ax, max_tiers: int) -> None:
    """Add a compact legend for the regime background colors."""
    import matplotlib.patches as mpatches

    handles = []
    for n in range(max_tiers, -1, -1):
        color = _REGIME_COLORS.get(n, _REGIME_COLORS[0])
        label = _REGIME_LABELS.get(n, f"{n} tiers")
        handles.append(mpatches.Patch(facecolor=color, alpha=0.5, label=label))
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.8, title="Ad Tiers")


def _fmt_dollars(x, _pos) -> str:
    """Format axis tick as dollars."""
    if abs(x) >= 1000:
        return f"${x:,.0f}"
    return f"${x:.0f}"


def _fmt_dollars_k(x, _pos) -> str:
    """Format axis tick as dollars in thousands."""
    return f"${x / 1000:,.0f}k"


def visualize_timeline(
    result: AAAResult,
    tiers: list[AudienceTier],
    margin: float,
    output_dir: Path,
    filename: str = "aaa_timeline.png",
) -> None:
    """Generate stacked time series charts from simulation results.

    Charts (top to bottom):
      1. Consumer Sentiment
      2. Seller Sales (units/month)
      3. Seller Revenue ($/month)
      4. Seller Ad Spend per Sale (CPA, $/sale)
      5. Seller Margin %
      6. Seller Profit ($/month)
      7. Advertiser (Platform) Revenue ($/month)

    Each chart has background shading showing the advertising regime
    (how many audience tiers are active).
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    output_dir.mkdir(parents=True, exist_ok=True)

    adv = result.advertiser
    max_tiers = len(tiers)

    # Extract data
    times = adv.sentiment_data.times()
    sentiments = adv.sentiment_data.raw_values()
    tier_counts = adv.active_tier_data.raw_values()
    sales = adv.total_sales_data.raw_values()
    revenues = adv.gross_revenue_data.raw_values()
    cpas = adv.blended_cpa_data.raw_values()
    margin_pcts = adv.margin_pct_data.raw_values()
    profits = adv.profit_data.raw_values()
    plat_revs = adv.platform_revenue_data.raw_values()

    x_max = max(times) + 1 if times else 1

    fig, axes = plt.subplots(7, 1, figsize=(14, 22), sharex=True)
    fig.suptitle(
        "Adverse Advertising Amplification (AAA)",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    chart_specs = [
        {
            "data": sentiments,
            "ylabel": "Consumer\nSentiment",
            "color": "#212121",
            "ylim": (0, 1.1),
            "fmt": None,
            "fill": True,
            "fill_color": "#90CAF9",
        },
        {
            "data": sales,
            "ylabel": "Seller\nSales/mo",
            "color": "#1976D2",
            "ylim": None,
            "fmt": "{x:,.0f}",
            "fill": False,
        },
        {
            "data": revenues,
            "ylabel": "Seller\nRevenue/mo",
            "color": "#388E3C",
            "ylim": None,
            "fmt": _fmt_dollars_k,
            "fill": False,
        },
        {
            "data": cpas,
            "ylabel": "Seller Ad\nSpend/Sale",
            "color": "#E64A19",
            "ylim": None,
            "fmt": _fmt_dollars,
            "fill": False,
        },
        {
            "data": margin_pcts,
            "ylabel": "Seller\nMargin %",
            "color": "#7B1FA2",
            "ylim": (0, max(margin_pcts) * 1.15) if margin_pcts else (0, 50),
            "fmt": "{x:.0f}%",
            "fill": False,
        },
        {
            "data": profits,
            "ylabel": "Seller\nProfit/mo",
            "color": "#0D47A1",
            "ylim": None,
            "fmt": _fmt_dollars_k,
            "fill": False,
        },
        {
            "data": plat_revs,
            "ylabel": "Platform\nRevenue/mo",
            "color": "#C62828",
            "ylim": None,
            "fmt": _fmt_dollars_k,
            "fill": False,
        },
    ]

    for ax, spec in zip(axes, chart_specs):
        _add_regime_background(ax, times, tier_counts, max_tiers, x_max)

        ax.plot(times, spec["data"], "-", linewidth=2.2, color=spec["color"], zorder=2)

        if spec.get("fill"):
            ax.fill_between(
                times,
                spec["data"],
                alpha=0.12,
                color=spec.get("fill_color", spec["color"]),
                zorder=1,
            )

        ax.set_ylabel(spec["ylabel"], fontsize=10, fontweight="bold")

        if spec["ylim"]:
            ax.set_ylim(spec["ylim"])
        else:
            vals = spec["data"]
            if vals:
                ymax = max(vals)
                ymin = min(vals)
                pad = (ymax - ymin) * 0.1 if ymax > ymin else ymax * 0.1
                ax.set_ylim(max(0, ymin - pad), ymax + pad)

        if spec["fmt"]:
            if callable(spec["fmt"]):
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(spec["fmt"]))
            else:
                ax.yaxis.set_major_formatter(mticker.StrMethodFormatter(spec["fmt"]))

        ax.grid(True, alpha=0.25, zorder=0)
        ax.tick_params(axis="both", labelsize=9)

    # Regime legend on the top chart
    _add_regime_legend(axes[0], max_tiers)

    # X-axis label on bottom chart
    axes[-1].set_xlabel("Month", fontsize=11, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.99])
    path = output_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
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


def _build_visual_charts(scenario: AAAScenario) -> list:
    """Build Chart objects for the visual debugger."""
    from happysimulator.visual import Chart

    adv = scenario.advertiser
    return [
        Chart(adv.sentiment_data, title="Consumer Sentiment", y_label="sentiment"),
        Chart(adv.total_sales_data, title="Seller Sales/mo", y_label="units"),
        Chart(adv.gross_revenue_data, title="Seller Revenue/mo", y_label="$"),
        Chart(adv.blended_cpa_data, title="Seller Ad Spend/Sale", y_label="$/sale"),
        Chart(adv.margin_pct_data, title="Seller Margin %", y_label="%"),
        Chart(adv.profit_data, title="Seller Profit/mo", y_label="$"),
        Chart(adv.platform_revenue_data, title="Platform Revenue/mo", y_label="$"),
        Chart(adv.active_tier_data, title="Active Tiers", y_label="tiers"),
    ]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Adverse Advertising Amplification (AAA)")
    parser.add_argument(
        "--visual",
        action="store_true",
        help="Launch the browser-based visual debugger instead of saving charts",
    )
    parser.add_argument(
        "--scenario",
        choices=["essay", "extended"],
        default="extended",
        help="Which scenario to run in visual mode (default: extended)",
    )
    args = parser.parse_args()

    if args.visual:
        from happysimulator.visual import serve

        scenario = (
            build_essay_scenario() if args.scenario == "essay" else build_extended_scenario()
        )
        charts = _build_visual_charts(scenario)
        print(f"  Launching visual debugger for {args.scenario} scenario...")
        serve(scenario.sim, charts=charts)
        return

    print("=" * 74)
    print("  ADVERSE ADVERTISING AMPLIFICATION (AAA)")
    print("  Demonstrating asymmetric impact of consumer sentiment on ad platforms")
    print("=" * 74)

    # -- Static Analysis: Essay Scenario ----------------------------------
    print("\n" + "-" * 74)
    print("  ESSAY SCENARIO (2 tiers)")
    print("-" * 74)
    print(
        f"\n  Product: ${PRODUCT_PRICE:.0f} price, "
        f"${PRODUCTION_COST:.0f} production cost, ${MARGIN:.0f} margin\n"
    )
    print_tier_table(ESSAY_TIERS, MARGIN)
    analyze_sentiment_shift(ESSAY_TIERS, MARGIN, before=1.0, after=0.80)

    # -- Static Analysis: Extended Scenario -------------------------------
    print("\n\n" + "-" * 74)
    print("  EXTENDED SCENARIO (4 tiers)")
    print("-" * 74)
    print(
        f"\n  Product: ${PRODUCT_PRICE:.0f} price, "
        f"${PRODUCTION_COST:.0f} production cost, ${MARGIN:.0f} margin\n"
    )
    print_tier_table(EXTENDED_TIERS, MARGIN)
    analyze_sentiment_shift(EXTENDED_TIERS, MARGIN, before=1.0, after=0.90)
    analyze_sentiment_shift(EXTENDED_TIERS, MARGIN, before=1.0, after=0.70)
    analyze_sentiment_shift(EXTENDED_TIERS, MARGIN, before=1.0, after=0.40)

    # -- Dynamic Simulation: Essay Scenario -------------------------------
    print("\n\n" + "-" * 74)
    print("  SIMULATION: Essay Scenario (smooth decline)")
    print("-" * 74)

    essay_scenario = build_essay_scenario()
    essay_result = run_scenario(essay_scenario)
    print_summary("Essay (2-tier)", essay_result)

    # -- Dynamic Simulation: Extended Scenario ----------------------------
    print("\n\n" + "-" * 74)
    print("  SIMULATION: Extended Scenario (smooth decline, 4 tiers)")
    print("-" * 74)

    extended_scenario = build_extended_scenario()
    extended_result = run_scenario(extended_scenario)
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

    visualize_timeline(essay_result, ESSAY_TIERS, MARGIN, output_dir, "aaa_essay_timeline.png")
    visualize_timeline(
        extended_result, EXTENDED_TIERS, MARGIN, output_dir, "aaa_extended_timeline.png"
    )

    print(f"\n  All visualizations saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
