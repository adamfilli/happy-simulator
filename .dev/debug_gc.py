"""Diagnostic script to investigate timeouts before t=10s in the GC collapse simulation.

Runs the GC collapse simulation with retries enabled and analyzes:
1. How many timeouts (retries) occur before vs after the first GC at t=10s
2. Queue depth samples during the first 10 seconds
"""

from examples.gc_caused_collapse import run_gc_collapse_simulation

GC_START = 10.0  # First GC pause starts at t=10s


def main():
    print("Running GC collapse simulation (retry_enabled=True, duration=30s, seed=42)...")
    result = run_gc_collapse_simulation(
        retry_enabled=True,
        duration_s=30,
        seed=42,
    )

    client = result.client
    attempts = client.attempts_by_time  # list of (time_s, attempt_number)

    # =========================================================================
    # 1. Analyze retries (attempt > 1) before vs after t=10s
    # =========================================================================
    retries_before = [(t, a) for t, a in attempts if a > 1 and t < GC_START]
    retries_after = [(t, a) for t, a in attempts if a > 1 and t >= GC_START]

    print("\n" + "=" * 70)
    print("RETRY ANALYSIS (attempt > 1 means a previous attempt timed out)")
    print("=" * 70)
    print(f"  Retries BEFORE t={GC_START}s (before first GC): {len(retries_before)}")
    print(f"  Retries AFTER  t={GC_START}s (after first GC):  {len(retries_after)}")

    if retries_before:
        print(f"\n  Detail of retries before t={GC_START}s:")
        for t, attempt in retries_before:
            print(f"    t={t:8.4f}s  attempt={attempt}")

    # Also count all attempts (including first attempts) to see total load
    attempts_before = [(t, a) for t, a in attempts if t < GC_START]
    first_attempts_before = [(t, a) for t, a in attempts_before if a == 1]
    print(f"\n  Total first-attempts (new requests) before t={GC_START}s: {len(first_attempts_before)}")
    print(f"  Total all-attempts before t={GC_START}s: {len(attempts_before)}")

    # =========================================================================
    # 2. Analyze timeouts more directly via client stats timeline
    # =========================================================================
    # The client tracks stats_timeouts as a running counter, but we can infer
    # from attempts_by_time: each attempt > 1 at time T means a timeout
    # happened at approximately T - retry_delay (50ms).
    # Let's look at what requests were timing out and when.
    print("\n" + "=" * 70)
    print("TIMEOUT TIMING ANALYSIS")
    print("=" * 70)
    print(f"  Client timeout setting: {client.timeout_s * 1000:.0f}ms")
    print(f"  Retry delay: {client.retry_delay_s * 1000:.0f}ms")
    print(f"  Total timeouts (all time): {client.stats_timeouts}")
    print(f"  Total retries (all time):  {client.stats_retries}")

    # Show the first 20 attempts to understand what's happening early
    print(f"\n  First 30 attempts (time, attempt#):")
    for i, (t, a) in enumerate(attempts[:30]):
        marker = " <-- RETRY (previous timed out)" if a > 1 else ""
        print(f"    [{i:3d}] t={t:8.4f}s  attempt={a}{marker}")

    # =========================================================================
    # 3. Queue depth in the first 10 seconds
    # =========================================================================
    queue_data = result.queue_depth_data.values  # list of (time_s, depth)
    queue_before = [(t, d) for t, d in queue_data if t < GC_START]

    print("\n" + "=" * 70)
    print(f"QUEUE DEPTH IN FIRST {GC_START}s")
    print("=" * 70)
    print(f"  Total samples before t={GC_START}s: {len(queue_before)}")

    if queue_before:
        depths = [d for _, d in queue_before]
        max_depth = max(depths)
        avg_depth = sum(depths) / len(depths)
        nonzero = sum(1 for d in depths if d > 0)

        print(f"  Max queue depth: {max_depth}")
        print(f"  Avg queue depth: {avg_depth:.2f}")
        print(f"  Samples with depth > 0: {nonzero} / {len(queue_before)}")

        # Show samples where depth > 0
        nonzero_samples = [(t, d) for t, d in queue_before if d > 0]
        if nonzero_samples:
            print(f"\n  Non-zero queue depth samples before t={GC_START}s:")
            for t, d in nonzero_samples[:50]:  # Limit output
                print(f"    t={t:8.4f}s  depth={d}")
            if len(nonzero_samples) > 50:
                print(f"    ... ({len(nonzero_samples) - 50} more)")
        else:
            print("  Queue was always empty before first GC.")

        # Show first 5 seconds at 1-second granularity
        print(f"\n  Queue depth by 1s buckets (first {GC_START}s):")
        bucket_size = 1.0
        for bucket_start in range(int(GC_START)):
            bucket_end = bucket_start + bucket_size
            bucket_samples = [d for t, d in queue_before if bucket_start <= t < bucket_end]
            if bucket_samples:
                print(
                    f"    [{bucket_start:2d}s - {bucket_end:2.0f}s): "
                    f"avg={sum(bucket_samples)/len(bucket_samples):5.2f}, "
                    f"max={max(bucket_samples):3.0f}, "
                    f"samples={len(bucket_samples)}"
                )

    # =========================================================================
    # 4. Service time analysis - are some requests just slow?
    # =========================================================================
    print("\n" + "=" * 70)
    print("LATENCY ANALYSIS (completed requests)")
    print("=" * 70)

    # Look at completions before t=10s
    early_completions = [
        (t.to_seconds(), lat)
        for t, lat in zip(client.completion_times, client.latencies_s)
        if t.to_seconds() < GC_START
    ]

    if early_completions:
        early_lats = [lat for _, lat in early_completions]
        timeout_ms = client.timeout_s * 1000
        slow_completions = [(t, lat) for t, lat in early_completions if lat * 1000 > timeout_ms]

        print(f"  Completions before t={GC_START}s: {len(early_completions)}")
        print(f"  Min latency: {min(early_lats)*1000:.1f}ms")
        print(f"  Max latency: {max(early_lats)*1000:.1f}ms")
        print(f"  Avg latency: {sum(early_lats)/len(early_lats)*1000:.1f}ms")
        print(f"  Completions with latency > timeout ({timeout_ms:.0f}ms): {len(slow_completions)}")

        # Latencies above 150ms (close to timeout)
        near_timeout = [(t, lat) for t, lat in early_completions if lat * 1000 > 150]
        if near_timeout:
            print(f"\n  Completions with latency > 150ms (before t={GC_START}s):")
            for t, lat in near_timeout[:20]:
                print(f"    completed_at={t:8.4f}s  latency={lat*1000:.1f}ms")
    else:
        print(f"  No completions before t={GC_START}s")

    # =========================================================================
    # 5. Summary / Hypothesis
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total client stats:")
    print(f"    Requests received:  {client.stats_requests_received}")
    print(f"    Attempts sent:      {client.stats_attempts_sent}")
    print(f"    Completions:        {client.stats_completions}")
    print(f"    Timeouts:           {client.stats_timeouts}")
    print(f"    Retries:            {client.stats_retries}")
    print(f"    Gave up:            {client.stats_gave_up}")
    print(f"    GC pauses recorded: {result.server.stats_gc_pauses}")
    print(f"    Server processed:   {result.server.stats_processed}")


if __name__ == "__main__":
    main()
