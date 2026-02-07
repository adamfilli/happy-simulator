│ Plan: examples/zipf_cache_cohorts.py                                                                                                                                            │
│                                                                                                                                                                                 │
│ Context                                                                                                                                                                         │
│                                                                                                                                                                                 │
│ Zipf-distributed traffic creates a "heavy hitter" effect: a small fraction of customers generate the majority of requests. When a cache with limited capacity sits in front of  │
│ a datastore, these heavy hitters stay in cache (high hit rate) while the long tail gets evicted before their next access (low hit rate). This example demonstrates that effect  │
│ with measurable per-cohort statistics.                                                                                                                                          │
│                                                                                                                                                                                 │
│ Architecture                                                                                                                                                                    │
│                                                                                                                                                                                 │
│ Source (100 req/s, constant rate)                                                                                                                                               │
│   → DistributedFieldProvider (samples customer_id from ZipfDistribution)                                                                                                        │
│     → CacheClient (Entity — tracks per-customer hit/miss)                                                                                                                       │
│       → SoftTTLCache (capacity=200, LRU eviction)                                                                                                                               │
│         → KVStore (pre-populated with 1000 customer records)                                                                                                                    │
│                                                                                                                                                                                 │
│ Single file: examples/zipf_cache_cohorts.py                                                                                                                                     │
│                                                                                                                                                                                 │
│ Components                                                                                                                                                                      │
│                                                                                                                                                                                 │
│ 1. CohortConfig — frozen dataclass with parameters (num_customers=1000, zipf_s=1.0, arrival_rate=100, duration=60s, cache_capacity=200, TTLs, seed)                             │
│ 2. CacheClient(Entity) — receives request events, reads from cache via yield from cache.get(key), tracks per-customer hits/misses using defaultdict(int). Hit/miss detection:   │
│ call cache.contains_cached(key) before yield from cache.get() — this is synchronous (no event processing between check and first yield), so it's reliable. With hard_ttl >>     │
│ duration, entries only leave cache via LRU eviction, making contains_cached a faithful proxy for hit vs miss.                                                                   │
│ 3. run_simulation(config) — wires up KVStore → SoftTTLCache → CacheClient → DistributedFieldProvider → Source, runs simulation, returns result.                                 │
│ 4. analyze_cohorts(result) — groups customers by Zipf rank into cohorts (Top 1%, 5%, 10%, 20%, Middle 30%, Bottom 50%). Computes per-cohort: traffic share (expected via        │
│ zipf.top_n_probability() and actual), request count, hit rate.                                                                                                                  │
│ 5. print_summary(result, cohorts) — prints config + formatted table of cohort results.                                                                                          │
│ 6. visualize_results(result, cohorts, output_dir) — 3-subplot figure:                                                                                                           │
│   - Bar chart: hit rate by cohort                                                                                                                                               │
│   - Bar chart: expected vs actual traffic share                                                                                                                                 │
│   - Bar chart: per-customer hit rate for top 100 customers by rank                                                                                                              │
│ 7. if __name__ == "__main__" — argparse CLI for key parameters (--customers, --zipf-s, --rate, --duration, --cache-size, --seed, --output, --no-viz)                            │
│                                                                                                                                                                                 │
│ Key imports                                                                                                                                                                     │
│                                                                                                                                                                                 │
│ - happysimulator: Entity, Event, Instant, Simulation, Source, ConstantArrivalTimeProvider, ConstantRateProfile, ZipfDistribution, DistributedFieldProvider                      │
│ - happysimulator.components.datastore: KVStore, SoftTTLCache                                                                                                                    │
│                                                                                                                                                                                 │
│ Expected output                                                                                                                                                                 │
│                                                                                                                                                                                 │
│ With default config (s=1.0, 1000 customers, cache=200):                                                                                                                         │
│ - Top 1% (10 customers): ~95%+ hit rate, ~27% of traffic                                                                                                                        │
│ - Top 10% (100 customers): ~85%+ hit rate                                                                                                                                       │
│ - Bottom 50% (500 customers): ~5-15% hit rate, ~9% of traffic                                                                                                                   │
│                                                                                                                                                                                 │
│ Verification                                                                                                                                                                    │
│                                                                                                                                                                                 │
│ python examples/zipf_cache_cohorts.py                                                                                                                                           │
│ python examples/zipf_cache_cohorts.py --zipf-s 1.5 --cache-size 100                                                                                                             │
│ pytest tests/integration/test_soft_ttl_cache.py -q  # existing tests still pass  