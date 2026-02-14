"""Orchestrates benchmark scenarios, collects results, and generates reports."""

from __future__ import annotations

import json
import platform
import sys
import tracemalloc
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

BASELINE_PATH = Path(__file__).parent / "baseline.json"


@dataclass
class BenchmarkResult:
    """Result from a single benchmark scenario."""

    name: str
    events_processed: int
    wall_clock_s: float
    events_per_second: float
    peak_memory_mb: float
    extra: dict[str, float] = field(default_factory=dict)


def run_scenario(
    module,
    *,
    scale: float = 1.0,
    tracemalloc_top: int = 0,
) -> BenchmarkResult:
    """Run a single scenario module, returning its BenchmarkResult."""
    tracemalloc.start()
    try:
        result = module.run(scale=scale)
    finally:
        snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()

    if tracemalloc_top > 0:
        print(f"\n  tracemalloc top {tracemalloc_top} for '{result.name}':")
        stats = snapshot.statistics("lineno")
        for i, stat in enumerate(stats[:tracemalloc_top]):
            print(f"    #{i + 1}: {stat}")

    return result


def run_all(
    scenarios: dict[str, object],
    *,
    scale: float = 1.0,
    tracemalloc_top: int = 0,
) -> list[BenchmarkResult]:
    """Run all scenarios and return their results."""
    results: list[BenchmarkResult] = []
    for name, module in scenarios.items():
        print(f"  Running '{name}'...", end="", flush=True)
        result = run_scenario(module, scale=scale, tracemalloc_top=tracemalloc_top)
        # Summary line
        if result.events_per_second > 0:
            print(f" {result.events_per_second:,.0f} events/sec  ({result.wall_clock_s:.3f}s)")
        else:
            print(f" done  ({result.wall_clock_s:.3f}s)")
        results.append(result)
    return results


def print_report(
    results: list[BenchmarkResult],
    baseline: dict | None = None,
) -> None:
    """Print a formatted console report."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    py_version = platform.python_version()

    print()
    print("=" * 78)
    print("  HAPPY-SIMULATOR PERFORMANCE REPORT")
    print(f"  Python {py_version} | {now}")
    print("=" * 78)
    print()
    print(
        f"  {'Scenario':<22s}  {'Events/sec':>12s}  {'Peak Mem (MB)':>14s}"
        f"  {'Wall (s)':>9s}  {'vs Baseline':>12s}"
    )
    print(
        f"  {'-' * 22}  {'-' * 12}  {'-' * 14}  {'-' * 9}  {'-' * 12}"
    )

    for r in results:
        eps_str = f"{r.events_per_second:>12,.0f}" if r.events_per_second > 0 else f"{'—':>12s}"
        wall_str = f"{r.wall_clock_s:>9.3f}" if r.wall_clock_s > 0 else f"{'—':>9s}"
        mem_str = f"{r.peak_memory_mb:>14.1f}"

        delta_str = ""
        if baseline and r.name in baseline:
            bl = baseline[r.name]
            if r.name == "memory_footprint":
                # Compare peak memory
                bl_val = bl.get("peak_memory_mb", 0)
                if bl_val > 0:
                    pct = (bl_val - r.peak_memory_mb) / bl_val * 100
                    sign = "+" if pct >= 0 else ""
                    delta_str = f"{sign}{pct:.1f}%"
            else:
                bl_val = bl.get("events_per_second", 0)
                if bl_val > 0:
                    pct = (r.events_per_second - bl_val) / bl_val * 100
                    sign = "+" if pct >= 0 else ""
                    delta_str = f"{sign}{pct:.1f}%"
            if not delta_str:
                delta_str = "(new)"
        elif baseline is not None:
            delta_str = "(new)"

        print(f"  {r.name:<22s}  {eps_str}  {mem_str}  {wall_str}  {delta_str:>12s}")

    # Extra metrics
    extras = [(r.name, r.extra) for r in results if r.extra]
    if extras:
        print()
        print("  Extra Metrics:")
        for name, extra in extras:
            parts = [f"{k}={v}" for k, v in extra.items()]
            print(f"    {name}: {', '.join(parts)}")

    print()
    print("=" * 78)


def save_baseline(results: list[BenchmarkResult]) -> None:
    """Save current results as the baseline."""
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "results": {r.name: asdict(r) for r in results},
    }
    BASELINE_PATH.write_text(json.dumps(payload, indent=2))
    print(f"  Baseline saved to {BASELINE_PATH}")


def load_baseline() -> dict[str, dict] | None:
    """Load baseline results, or None if no baseline exists."""
    if not BASELINE_PATH.exists():
        return None
    data = json.loads(BASELINE_PATH.read_text())
    return data.get("results")


def results_to_json(results: list[BenchmarkResult]) -> str:
    """Serialize results list to JSON string."""
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "results": [asdict(r) for r in results],
    }
    return json.dumps(payload, indent=2)
