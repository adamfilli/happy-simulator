"""Adverse Advertising Amplification v3 — Consumers as Entities.

Builds on v2 by promoting each of the 10,000 consumers from passive
``ConsumerRecord`` dataclasses to proper ``Entity`` subclasses.  The ad
funnel logic (exposure → click → conversion) moves from the Platform
into each Consumer's ``handle_event()``, while the Platform becomes a
thin router that picks a random Consumer and forwards the Visit.

## Architecture

- **Consumer** (Entity): 10,000 individual consumers, each with segment,
  sentiment, saturation.  Handles the probabilistic ad funnel.
- **Platform** (Entity): Receives Visit events from Source, picks a
  random Consumer, forwards the event.  Also handles SentimentAdjust.
- **SellerEntity** (Entity): Tracks per-segment profit, weekly evaluation
  deactivates unprofitable segments.  Unchanged from v2.
- **AdvertiserEntity** (Entity): Platform revenue collector, manages
  active segment set.  Unchanged from v2.

## Event Flow (v2 → v3)

    v2: Source → Platform (runs funnel) → Advertiser/Seller
    v3: Source → Platform (router) → Consumer (runs funnel) → Advertiser/Seller

One extra hop per visit — event count roughly doubles for the routing step.

## Segment Economics (unchanged from v2)

Click rates are fixed; only conversion rate is modulated by sentiment.
CPA = CPC / (conv_rate * eff_sentiment). Breakeven when CPA >= margin ($25).

| Segment       | Size  | Click% | Conv%|click | CPC   | CPA@0.85 | Breakeven |
|---------------|-------|--------|------|------|----------|-----------|
| Core Niche    | 1,000 |   8%   |  50% | $0.50 |  $1.18  |   0.04    |
| Adjacent      | 1,500 |   5%   |  20% | $2.60 | $15.29  |   0.52    |
| Demographic   | 3,000 |   3%   |  10% | $1.60 | $18.82  |   0.64    |
| Broad         | 4,500 |  1.5%  |   7% | $1.20 | $20.17  |   0.69    |

Time: 1 unit = 1 hour, 24 weeks = 4,032 hours.
Seller starts with core_niche only, expands when profitable.
Sentiment decline: weeks 9-13 via cosine interpolation (after full expansion stabilises).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

from happysimulator import (
    Data,
    Duration,
    Entity,
    Event,
    Instant,
    Simulation,
    Source,
)

# =============================================================================
# Constants (identical to v2)
# =============================================================================

HOURS_PER_WEEK = 168
DURATION_WEEKS = 24
DURATION_HOURS = DURATION_WEEKS * HOURS_PER_WEEK  # 4,032

PRODUCT_PRICE = 50.0
COGS = 25.0
MARGIN = PRODUCT_PRICE - COGS  # $25

SEED = 42

SEGMENTS = {
    "core_niche": {"size": 1_000, "click_rate": 0.08, "conv_rate": 0.50, "cpc": 0.50},
    "adjacent": {"size": 1_500, "click_rate": 0.05, "conv_rate": 0.20, "cpc": 2.60},
    "demographic": {"size": 3_000, "click_rate": 0.03, "conv_rate": 0.10, "cpc": 1.60},
    "broad": {"size": 4_500, "click_rate": 0.015, "conv_rate": 0.07, "cpc": 1.20},
}

SEGMENT_NAMES = list(SEGMENTS.keys())
TOTAL_CONSUMERS = sum(s["size"] for s in SEGMENTS.values())  # 10,000

VISIT_RATE_PER_CONSUMER = 3.5 / HOURS_PER_WEEK  # ~0.02083/hr
AGGREGATE_VISIT_RATE = TOTAL_CONSUMERS * VISIT_RATE_PER_CONSUMER  # ~208/hr

SENTIMENT_DECLINE_START = 9 * HOURS_PER_WEEK  # hour 1512
SENTIMENT_DECLINE_END = 13 * HOURS_PER_WEEK  # hour 2184
SENTIMENT_DECLINE_TARGET = -0.35

EVAL_INTERVAL = HOURS_PER_WEEK  # weekly

SATURATION_INCREASE = 0.02  # per ad exposure
SATURATION_DECAY = 0.10  # per visit — fast decay means consumers "forget" ads quickly

# Regime background colors (number of active segments -> color)
_REGIME_COLORS = {
    0: "#ef9a9a",
    1: "#ffcc80",
    2: "#fff176",
    3: "#a5d6a7",
    4: "#81c784",
}

_REGIME_LABELS = {
    0: "0 segments",
    1: "1 segment",
    2: "2 segments",
    3: "3 segments",
    4: "4 segments",
}


# =============================================================================
# Smooth sentiment curve (reused from v1/v2)
# =============================================================================


def _cosine_interp(
    t: float, t_start: float, t_end: float, v_start: float, v_end: float
) -> float:
    """Cosine interpolation between two values."""
    if t <= t_start:
        return v_start
    if t >= t_end:
        return v_end
    progress = (t - t_start) / (t_end - t_start)
    return v_start + (v_end - v_start) * (1 - math.cos(math.pi * progress)) / 2


# =============================================================================
# Entities
# =============================================================================


class AdvertiserEntity(Entity):
    """Ad platform revenue collector. Manages active segment set.

    Receives:
        - AdClick: records CPC revenue
        - SegmentControl: activate/deactivate segments

    Unchanged from v2.
    """

    def __init__(self, name: str = "AdPlatform"):
        super().__init__(name)
        self.active_segments: set[str] = {"core_niche"}
        self.revenue_data = Data()
        self._period_revenue = 0.0

    def handle_event(self, event: Event) -> None:
        if event.event_type == "AdClick":
            cpc = event.context.get("cpc", 0.0)
            self._period_revenue += cpc
            self.revenue_data.add_stat(cpc, event.time)

        elif event.event_type == "SegmentControl":
            segment = event.context["segment"]
            active = event.context["active"]
            if active:
                self.active_segments.add(segment)
            else:
                self.active_segments.discard(segment)


class SellerEntity(Entity):
    """Product seller. Tracks per-segment profit, evaluates weekly.

    Receives:
        - Sale: records revenue from conversion
        - AdCost: records CPC cost
        - Evaluate: weekly self-scheduled evaluation timer

    Unchanged from v2.
    """

    def __init__(self, name: str, advertiser: AdvertiserEntity):
        super().__init__(name)
        self._advertiser = advertiser

        self._segment_sales: dict[str, int] = {s: 0 for s in SEGMENT_NAMES}
        self._segment_revenue: dict[str, float] = {s: 0.0 for s in SEGMENT_NAMES}
        self._segment_ad_spend: dict[str, float] = {s: 0.0 for s in SEGMENT_NAMES}

        self._consecutive_loss: dict[str, int] = {s: 0 for s in SEGMENT_NAMES}
        self._loss_weeks_to_deactivate = 3

        self._cooldown: dict[str, int] = {s: 0 for s in SEGMENT_NAMES}
        self._cooldown_base = 52
        self._deactivation_count: dict[str, int] = {s: 0 for s in SEGMENT_NAMES}

        self.sales_data = Data()
        self.revenue_data = Data()
        self.ad_spend_data = Data()
        self.cpa_data = Data()
        self.margin_pct_data = Data()
        self.profit_data = Data()
        self.segment_profit_data: dict[str, Data] = {s: Data() for s in SEGMENT_NAMES}
        self.active_segment_data = Data()

    def start_events(self) -> list[Event]:
        """Schedule the first evaluation timer."""
        return [
            Event(
                time=Instant.from_seconds(EVAL_INTERVAL),
                event_type="Evaluate",
                target=self,
            )
        ]

    def handle_event(self, event: Event) -> list[Event] | None:
        if event.event_type == "Sale":
            segment = event.context["segment"]
            self._segment_sales[segment] += 1
            self._segment_revenue[segment] += PRODUCT_PRICE

        elif event.event_type == "AdCost":
            segment = event.context["segment"]
            cpc = event.context["cpc"]
            self._segment_ad_spend[segment] += cpc

        elif event.event_type == "Evaluate":
            return self._evaluate(event)

        return None

    def _evaluate(self, event: Event) -> list[Event]:
        """Weekly evaluation: expand, contract, or hold segments."""
        total_sales = sum(self._segment_sales.values())
        total_revenue = sum(self._segment_revenue.values())
        total_ad_spend = sum(self._segment_ad_spend.values())
        total_cogs = total_sales * COGS
        total_profit = total_revenue - total_cogs - total_ad_spend

        self.sales_data.add_stat(total_sales, event.time)
        self.revenue_data.add_stat(total_revenue, event.time)
        self.ad_spend_data.add_stat(total_ad_spend, event.time)
        self.profit_data.add_stat(total_profit, event.time)

        for seg_name in SEGMENT_NAMES:
            s = self._segment_sales[seg_name]
            r = self._segment_revenue[seg_name]
            sp = self._segment_ad_spend[seg_name]
            self.segment_profit_data[seg_name].add_stat(r - s * COGS - sp, event.time)

        cpa = total_ad_spend / total_sales if total_sales > 0 else 0.0
        self.cpa_data.add_stat(cpa, event.time)

        if total_revenue > 0:
            margin_pct = (total_profit / total_revenue) * 100
        else:
            margin_pct = 0.0
        self.margin_pct_data.add_stat(margin_pct, event.time)

        control_events: list[Event] = []

        for seg_name in SEGMENT_NAMES:
            if self._cooldown[seg_name] > 0:
                self._cooldown[seg_name] -= 1

        # --- Contraction ---
        for seg_name in SEGMENT_NAMES:
            sales = self._segment_sales[seg_name]
            revenue = self._segment_revenue[seg_name]
            spend = self._segment_ad_spend[seg_name]
            cogs = sales * COGS
            seg_profit = revenue - cogs - spend

            currently_active = seg_name in self._advertiser.active_segments

            if currently_active and seg_profit < 0 and sales > 0:
                self._consecutive_loss[seg_name] += 1
                if self._consecutive_loss[seg_name] >= self._loss_weeks_to_deactivate:
                    control_events.append(
                        Event(
                            time=event.time,
                            event_type="SegmentControl",
                            target=self._advertiser,
                            context={"segment": seg_name, "active": False},
                        )
                    )
                    self._consecutive_loss[seg_name] = 0
                    self._deactivation_count[seg_name] += 1
                    self._cooldown[seg_name] = self._cooldown_base * (
                        2 ** (self._deactivation_count[seg_name] - 1)
                    )
            else:
                self._consecutive_loss[seg_name] = 0

        # --- Expansion ---
        if not control_events and total_profit > 0:
            for seg_name in SEGMENT_NAMES:
                if (
                    seg_name not in self._advertiser.active_segments
                    and self._cooldown[seg_name] == 0
                ):
                    control_events.append(
                        Event(
                            time=event.time,
                            event_type="SegmentControl",
                            target=self._advertiser,
                            context={"segment": seg_name, "active": True},
                        )
                    )
                    break

        active_count = len(self._advertiser.active_segments)
        self.active_segment_data.add_stat(active_count, event.time)

        self._segment_sales = {s: 0 for s in SEGMENT_NAMES}
        self._segment_revenue = {s: 0.0 for s in SEGMENT_NAMES}
        self._segment_ad_spend = {s: 0.0 for s in SEGMENT_NAMES}

        next_eval = Event(
            time=event.time + Duration.from_seconds(EVAL_INTERVAL),
            event_type="Evaluate",
            target=self,
        )

        return control_events + [next_eval]


class Consumer(Entity):
    """A single consumer in the simulation.

    Each consumer belongs to a segment and has individual sentiment and
    saturation state.  On receiving a Visit event (forwarded by Platform),
    the consumer runs the probabilistic ad funnel:
    exposure → click → conversion.
    """

    def __init__(
        self,
        consumer_id: int,
        segment: str,
        base_sentiment: float,
        platform: Platform,
        advertiser: AdvertiserEntity,
        seller: SellerEntity,
        rng: random.Random,
    ):
        super().__init__(f"consumer_{consumer_id:04d}")
        self.segment = segment
        self.base_sentiment = base_sentiment
        self.saturation = 0.0
        self._platform = platform
        self._advertiser = advertiser
        self._seller = seller
        self._rng = rng

    def handle_event(self, event: Event) -> list[Event] | None:
        if event.event_type != "Visit":
            return None

        # Decay saturation on every visit
        self.saturation = max(0.0, self.saturation - SATURATION_DECAY)

        # Check if segment is active
        if self.segment not in self._advertiser.active_segments:
            return None  # organic visit, no ad served

        seg_info = SEGMENTS[self.segment]

        # Effective sentiment for this consumer
        eff_sentiment = max(
            0.01, self.base_sentiment + self._platform.global_sentiment_adj
        )

        # Saturation multiplier (reduces conversion effectiveness)
        sat_mult = 1.0 - self.saturation * 0.5

        # Exposure (always shown an ad if segment is active)
        self.saturation = min(1.0, self.saturation + SATURATION_INCREASE)

        # Click probability — NOT affected by sentiment
        click_prob = seg_info["click_rate"]
        if self._rng.random() > click_prob:
            return None  # no click

        # Click happened — emit AdClick + AdCost
        cpc = seg_info["cpc"]
        results: list[Event] = [
            Event(
                time=event.time,
                event_type="AdClick",
                target=self._advertiser,
                context={"segment": self.segment, "cpc": cpc},
            ),
            Event(
                time=event.time,
                event_type="AdCost",
                target=self._seller,
                context={"segment": self.segment, "cpc": cpc},
            ),
        ]

        # Conversion probability (conditional on click) — AFFECTED by sentiment
        conv_prob = seg_info["conv_rate"] * eff_sentiment * sat_mult
        if self._rng.random() <= conv_prob:
            results.append(
                Event(
                    time=event.time,
                    event_type="Sale",
                    target=self._seller,
                    context={
                        "segment": self.segment,
                        "price": PRODUCT_PRICE,
                        "created_at": event.time,
                    },
                )
            )

        return results


class Platform(Entity):
    """Ad-serving platform — thin router in v3.

    Receives Visit events from Source, picks a random Consumer entity,
    and forwards the event.  Also handles SentimentAdjust to track the
    global sentiment adjustor (read by Consumer entities).
    """

    def __init__(
        self,
        name: str,
        consumers: list[Consumer],
        rng: random.Random,
    ):
        super().__init__(name)
        self._consumers = consumers
        self._rng = rng
        self._global_sentiment_adj = 0.0

        # Data for visualization
        self.sentiment_data = Data()

    @property
    def global_sentiment_adj(self) -> float:
        return self._global_sentiment_adj

    def handle_event(self, event: Event) -> list[Event] | None:
        if event.event_type == "Visit":
            consumer = self._rng.choice(self._consumers)
            return [Event(time=event.time, event_type="Visit", target=consumer)]
        elif event.event_type == "SentimentAdjust":
            self._global_sentiment_adj = event.context["adjustor"]
            self.sentiment_data.add_stat(self._global_sentiment_adj, event.time)
            return None
        return None


# =============================================================================
# Consumer pool builder
# =============================================================================


def _build_consumer_pool(
    platform: Platform,
    advertiser: AdvertiserEntity,
    seller: SellerEntity,
    rng: random.Random,
) -> list[Consumer]:
    """Create 10,000 Consumer entities with segment-weighted distribution."""
    consumers: list[Consumer] = []
    cid = 0
    for seg_name, seg_info in SEGMENTS.items():
        for _ in range(seg_info["size"]):
            base_sent = max(0.2, min(1.0, rng.gauss(0.85, 0.10)))
            consumers.append(
                Consumer(
                    consumer_id=cid,
                    segment=seg_name,
                    base_sentiment=base_sent,
                    platform=platform,
                    advertiser=advertiser,
                    seller=seller,
                    rng=rng,
                )
            )
            cid += 1
    return consumers


# =============================================================================
# Scenario builder
# =============================================================================


@dataclass
class AAAv3Scenario:
    """A prepared v3 scenario (built but not yet run)."""

    sim: Simulation
    platform: Platform
    seller: SellerEntity
    advertiser: AdvertiserEntity
    consumers: list[Consumer]


def build_scenario(seed: int = SEED) -> AAAv3Scenario:
    """Build the entity-based AAA simulation."""
    rng = random.Random(seed)

    # Create core entities
    advertiser = AdvertiserEntity("AdPlatform")
    seller = SellerEntity("PosterShop", advertiser)

    # Platform created first (consumers need a reference to it)
    # Consumers list populated after construction
    platform = Platform("Platform", [], rng)

    # Build consumer entities (each references platform, advertiser, seller)
    consumers = _build_consumer_pool(platform, advertiser, seller, rng)
    platform._consumers = consumers

    # Source: Poisson arrivals at ~208 visits/hour
    source = Source.poisson(
        rate=AGGREGATE_VISIT_RATE,
        target=platform,
        event_type="Visit",
        name="Traffic",
        stop_after=float(DURATION_HOURS),
    )

    sim = Simulation(
        sources=[source],
        entities=[platform, advertiser, seller, *consumers],
        duration=DURATION_HOURS + 1,
    )

    # Schedule seller evaluation timer
    for e in seller.start_events():
        sim.schedule(e)

    # Schedule sentiment events across entire simulation
    _schedule_sentiment(sim, platform)

    return AAAv3Scenario(
        sim=sim,
        platform=platform,
        seller=seller,
        advertiser=advertiser,
        consumers=consumers,
    )


def _schedule_sentiment(sim: Simulation, platform: Platform) -> None:
    """Schedule SentimentAdjust events across the entire simulation.

    Before and after the decline window the adjustor is flat (0.0 before,
    SENTIMENT_DECLINE_TARGET after).  During the decline window it follows
    a cosine interpolation.  Events are emitted every 12 hours so the
    sentiment chart always has data.
    """
    step = 12.0
    t = 0.1
    while t <= DURATION_HOURS:
        if t < SENTIMENT_DECLINE_START:
            adj = 0.0
        elif t > SENTIMENT_DECLINE_END:
            adj = SENTIMENT_DECLINE_TARGET
        else:
            adj = _cosine_interp(
                t, SENTIMENT_DECLINE_START, SENTIMENT_DECLINE_END, 0.0, SENTIMENT_DECLINE_TARGET
            )
        sim.schedule(
            Event(
                time=Instant.from_seconds(t),
                event_type="SentimentAdjust",
                target=platform,
                context={"adjustor": adj},
            )
        )
        t += step


# =============================================================================
# Visualization (identical charts to v2, just renamed)
# =============================================================================


def _add_regime_background(
    ax,
    times: list[float],
    tier_counts: list[float],
    max_tiers: int,
    x_max: float,
) -> None:
    """Draw colored vertical bands showing active segment count."""
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
        label = _REGIME_LABELS.get(n, f"{n} segments")
        handles.append(mpatches.Patch(facecolor=color, alpha=0.5, label=label))
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.8, title="Segments")


def _fmt_dollars(x, _pos) -> str:
    if abs(x) >= 1000:
        return f"${x:,.1f}"
    return f"${x:.1f}"


def _fmt_dollars_k(x, _pos) -> str:
    return f"${x / 1000:,.1f}k"


def _hours_to_weeks(times: list[float]) -> list[float]:
    """Convert hour timestamps to week numbers for x-axis."""
    return [t / HOURS_PER_WEEK for t in times]


def visualize_timeline(
    scenario: AAAv3Scenario,
    output_dir: Path,
    filename: str = "aaa_v3_timeline.png",
) -> None:
    """Generate 6 stacked time series charts."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    output_dir.mkdir(parents=True, exist_ok=True)

    seller = scenario.seller
    platform = scenario.platform
    advertiser = scenario.advertiser
    max_tiers = len(SEGMENT_NAMES)

    eval_times = seller.sales_data.times()
    eval_weeks = _hours_to_weeks(eval_times)
    sales = seller.sales_data.raw_values()
    revenues = seller.revenue_data.raw_values()
    cpas = seller.cpa_data.raw_values()
    profits = seller.profit_data.raw_values()
    tier_counts = seller.active_segment_data.raw_values()

    seg_profit_colors = {
        "core_niche": "#4CAF50",
        "adjacent": "#FF9800",
        "demographic": "#9C27B0",
        "broad": "#F44336",
    }
    seg_profits = {
        s: seller.segment_profit_data[s].raw_values() for s in SEGMENT_NAMES
    }

    sent_times = platform.sentiment_data.times()
    sent_weeks = _hours_to_weeks(sent_times)
    sent_values = platform.sentiment_data.raw_values()
    mean_effective = [0.85 + v for v in sent_values]

    adv_rev_weekly = _bucket_advertiser_revenue(advertiser, eval_times)

    x_min = 0
    x_max = DURATION_WEEKS

    fig, axes = plt.subplots(6, 1, figsize=(14, 20), sharex=True)
    fig.suptitle(
        "Adverse Advertising Amplification v3 (Consumers as Entities)",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    chart_specs = [
        {
            "times": sent_weeks,
            "data": mean_effective,
            "ylabel": "Consumer\nSentiment",
            "color": "#212121",
            "ylim": (0, 1.1),
            "fmt": None,
            "fill": True,
            "fill_color": "#90CAF9",
            "use_regime": False,
        },
        {
            "times": eval_weeks,
            "data": sales,
            "ylabel": "Seller\nSales/wk",
            "color": "#1976D2",
            "ylim": None,
            "fmt": "{x:,.0f}",
            "fill": False,
            "use_regime": True,
        },
        {
            "times": eval_weeks,
            "data": revenues,
            "ylabel": "Seller\nRevenue/wk",
            "color": "#388E3C",
            "ylim": None,
            "fmt": _fmt_dollars_k,
            "fill": False,
            "use_regime": True,
        },
        {
            "times": eval_weeks,
            "data": cpas,
            "ylabel": "Seller Ad\nSpend/Sale",
            "color": "#E64A19",
            "ylim": None,
            "fmt": _fmt_dollars,
            "fill": False,
            "use_regime": True,
        },
        {
            "times": eval_weeks,
            "data": profits,
            "ylabel": "Seller\nProfit/wk",
            "color": "#0D47A1",
            "ylim": None,
            "fmt": _fmt_dollars_k,
            "fill": False,
            "use_regime": True,
            "extra_lines": [
                {"times": eval_weeks, "data": seg_profits[s], "color": seg_profit_colors[s], "label": s}
                for s in SEGMENT_NAMES
            ],
        },
        {
            "times": eval_weeks,
            "data": adv_rev_weekly,
            "ylabel": "Platform\nRevenue/wk",
            "color": "#C62828",
            "ylim": None,
            "fmt": _fmt_dollars_k,
            "fill": False,
            "use_regime": True,
        },
    ]

    regime_weeks = _hours_to_weeks(eval_times) if eval_times else []

    for ax, spec in zip(axes, chart_specs):
        if spec["use_regime"] and regime_weeks and tier_counts:
            _add_regime_background(ax, regime_weeks, tier_counts, max_tiers, x_max)

        t = spec["times"]
        d = spec["data"]
        if t and d:
            ax.plot(t, d, "-", linewidth=2.2, color=spec["color"], zorder=3, label="Total")

            if spec.get("fill"):
                ax.fill_between(
                    t,
                    d,
                    alpha=0.12,
                    color=spec.get("fill_color", spec["color"]),
                    zorder=1,
                )

        all_vals = list(d) if d else []
        for extra in spec.get("extra_lines", []):
            et, ed = extra["times"], extra["data"]
            if et and ed:
                ax.plot(et, ed, "-", linewidth=1.2, color=extra["color"],
                        alpha=0.8, zorder=2, label=extra.get("label"))
                all_vals.extend(ed)
        if spec.get("extra_lines"):
            ax.legend(fontsize=7, loc="upper right", framealpha=0.8, ncol=3)

        ax.set_ylabel(spec["ylabel"], fontsize=10, fontweight="bold")

        if spec["ylim"]:
            ax.set_ylim(spec["ylim"])
        elif all_vals:
            ymax = max(all_vals)
            ymin = min(all_vals)
            pad = (ymax - ymin) * 0.1 if ymax > ymin else max(abs(ymax) * 0.1, 1)
            ax.set_ylim(min(0, ymin - pad), ymax + pad)

        if spec["fmt"]:
            if callable(spec["fmt"]):
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(spec["fmt"]))
            else:
                ax.yaxis.set_major_formatter(mticker.StrMethodFormatter(spec["fmt"]))

        ax.set_xlim(x_min, x_max)
        ax.grid(True, alpha=0.25, zorder=0)
        ax.tick_params(axis="both", labelsize=9)

    if regime_weeks and tier_counts:
        _add_regime_legend(axes[0], max_tiers)

    axes[-1].set_xlabel("Week", fontsize=11, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.99])
    path = output_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _bucket_advertiser_revenue(
    advertiser: AdvertiserEntity, eval_times: list[float]
) -> list[float]:
    """Aggregate advertiser revenue into weekly buckets aligned with eval_times."""
    if not eval_times:
        return []

    rev_samples = advertiser.revenue_data.values
    buckets: list[float] = []

    for i, end_t in enumerate(eval_times):
        start_t = eval_times[i - 1] if i > 0 else 0.0
        total = sum(v for t, v in rev_samples if start_t <= t < end_t)
        buckets.append(total)

    return buckets


# =============================================================================
# Visual debugger charts
# =============================================================================


def _build_visual_charts(scenario: AAAv3Scenario) -> list:
    """Build Chart objects for the visual debugger."""
    from happysimulator.visual import Chart

    seller = scenario.seller
    advertiser = scenario.advertiser
    platform = scenario.platform

    return [
        Chart(platform.sentiment_data, title="Sentiment Adjustor", y_label="adjustor"),
        Chart(seller.sales_data, title="Seller Sales/wk", y_label="units"),
        Chart(seller.revenue_data, title="Seller Revenue/wk", y_label="$"),
        Chart(seller.cpa_data, title="Seller CPA", y_label="$/sale"),
        Chart(seller.profit_data, title="Seller Profit/wk", y_label="$"),
        Chart(advertiser.revenue_data, title="Platform Revenue (clicks)", y_label="$"),
        Chart(seller.active_segment_data, title="Active Segments", y_label="count"),
    ]


# =============================================================================
# Main
# =============================================================================


def print_summary(scenario: AAAv3Scenario, summary) -> None:
    """Print simulation summary."""
    seller = scenario.seller
    advertiser = scenario.advertiser

    total_sales = sum(seller.sales_data.raw_values())
    total_revenue = sum(seller.revenue_data.raw_values())
    total_profit = sum(seller.profit_data.raw_values())
    total_plat_rev = sum(v for _, v in advertiser.revenue_data.values)

    print(f"\n  --- Simulation Results ---")
    print(f"  Duration: {DURATION_WEEKS} weeks ({DURATION_HOURS} hours)")
    print(f"  Consumers: {TOTAL_CONSUMERS:,} (each a full Entity)")
    print(f"  Total seller sales: {total_sales:,.0f}")
    print(f"  Total seller revenue: ${total_revenue:,.0f}")
    print(f"  Total seller profit: ${total_profit:,.0f}")
    print(f"  Total platform revenue: ${total_plat_rev:,.0f}")
    print(f"  {summary}")

    eval_times = seller.sales_data.times()
    sales_vals = seller.sales_data.raw_values()
    profit_vals = seller.profit_data.raw_values()
    tier_counts = seller.active_segment_data.raw_values()

    print(f"\n  {'Week':>6s} {'Sales':>8s} {'Profit':>10s} {'Segments':>10s}")
    print(f"  {'-' * 6} {'-' * 8} {'-' * 10} {'-' * 10}")
    for i, t in enumerate(eval_times):
        week = t / HOURS_PER_WEEK
        print(
            f"  {week:6.0f} {sales_vals[i]:8,.0f} ${profit_vals[i]:9,.0f} "
            f"{tier_counts[i]:10.0f}"
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Adverse Advertising Amplification v3 (Consumers as Entities)"
    )
    parser.add_argument(
        "--visual",
        action="store_true",
        help="Launch the browser-based visual debugger instead of saving charts",
    )
    args = parser.parse_args()

    if args.visual:
        from happysimulator.visual import serve

        scenario = build_scenario()
        charts = _build_visual_charts(scenario)
        print("  Launching visual debugger for AAA v3...")
        serve(scenario.sim, charts=charts)
        return

    print("=" * 74)
    print("  ADVERSE ADVERTISING AMPLIFICATION v3 (Consumers as Entities)")
    print("  10,000 individual consumers, each a full Entity subclass")
    print("=" * 74)

    scenario = build_scenario()
    print(f"\n  Running simulation ({DURATION_WEEKS} weeks, {TOTAL_CONSUMERS:,} consumer entities)...")
    summary = scenario.sim.run()
    print_summary(scenario, summary)

    try:
        import matplotlib

        matplotlib.use("Agg")
    except ImportError:
        print("\n  [matplotlib not installed, skipping charts]")
        return

    output_dir = Path("output/aaa")
    print(f"\n  Generating visualizations -> {output_dir.absolute()}")
    visualize_timeline(scenario, output_dir)
    print(f"\n  All visualizations saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
