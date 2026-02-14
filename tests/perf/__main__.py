"""CLI entry point: python -m tests.perf"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
from pathlib import Path

from tests.perf.runner import (
    list_checkpoints,
    load_baseline,
    load_checkpoint,
    print_report,
    results_to_json,
    run_all,
    run_scenario,
    save_baseline,
    save_checkpoint,
)
from tests.perf.scenarios import SCENARIOS


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m tests.perf",
        description="Happy-Simulator performance benchmarks",
    )
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()),
        help="Run a single scenario instead of all",
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save results as baseline.json for future comparison",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare against saved baseline (default when baseline exists)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run with cProfile; save .prof files to test_output/perf/",
    )
    parser.add_argument(
        "--tracemalloc-top",
        type=int,
        default=0,
        metavar="N",
        help="Show top N memory allocators after each scenario",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Multiply event counts by this factor (default: 1.0)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON to stdout",
    )
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="Save results as a checkpoint in tests/perf/data/ (date + git hash)",
    )
    parser.add_argument(
        "--list-checkpoints",
        action="store_true",
        help="List all saved checkpoints and exit",
    )
    parser.add_argument(
        "--compare-checkpoint",
        metavar="FILE",
        help="Compare against a specific checkpoint file in tests/perf/data/",
    )
    args = parser.parse_args()

    # List checkpoints and exit
    if args.list_checkpoints:
        checkpoints = list_checkpoints()
        if not checkpoints:
            print("  No checkpoints saved yet.")
        else:
            print(f"  {len(checkpoints)} checkpoint(s) in tests/perf/data/:")
            for cp in checkpoints:
                data = load_checkpoint(cp)
                ts = data.get("timestamp", "?")[:19]
                git = data.get("git_hash", "?")
                n = len(data.get("results", {}))
                print(f"    {cp.name:<36s}  {ts}  {git}  ({n} scenarios)")
        return

    # Select scenarios
    if args.scenario:
        scenarios = {args.scenario: SCENARIOS[args.scenario]}
    else:
        scenarios = SCENARIOS

    # Profile mode
    if args.profile:
        prof_dir = Path("test_output/perf")
        prof_dir.mkdir(parents=True, exist_ok=True)

        from tests.perf.runner import BenchmarkResult

        results = []
        for name, module in scenarios.items():
            print(f"  Profiling '{name}'...")
            profiler = cProfile.Profile()
            profiler.enable()
            result = run_scenario(
                module, scale=args.scale, tracemalloc_top=args.tracemalloc_top
            )
            profiler.disable()
            results.append(result)

            prof_path = prof_dir / f"{name}.prof"
            profiler.dump_stats(str(prof_path))
            print(f"    Saved profile to {prof_path}")
            print(f"    Top 20 by cumulative time:")
            stats = pstats.Stats(profiler)
            stats.sort_stats("cumulative")
            stats.print_stats(20)
    else:
        results = run_all(
            scenarios,
            scale=args.scale,
            tracemalloc_top=args.tracemalloc_top,
        )

    # JSON output
    if args.json:
        print(results_to_json(results))
        return

    # Load baseline for comparison
    baseline = None
    if args.compare_checkpoint:
        cp_path = Path(args.compare_checkpoint)
        if not cp_path.exists():
            # Try as a filename inside data/
            from tests.perf.runner import DATA_DIR

            cp_path = DATA_DIR / args.compare_checkpoint
        if cp_path.exists():
            cp_data = load_checkpoint(cp_path)
            baseline = cp_data.get("results")
            git_hash = cp_data.get("git_hash", "?")
            print(f"  Comparing against checkpoint: {cp_path.name} ({git_hash})")
        else:
            print(f"  Warning: checkpoint '{args.compare_checkpoint}' not found, skipping comparison")
    elif args.compare or (not args.save_baseline):
        baseline = load_baseline()

    print_report(results, baseline=baseline)

    if args.save_baseline:
        save_baseline(results)

    if args.checkpoint:
        path = save_checkpoint(results)
        print(f"  Checkpoint saved to {path}")


if __name__ == "__main__":
    main()
