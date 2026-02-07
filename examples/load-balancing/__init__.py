"""Load balancing examples demonstrating consistent hashing benefits.

This package contains comprehensive examples showing:
- Consistent hashing vs round robin cache hit rates
- Fleet change impact comparison
- Virtual node distribution analysis
- Zipf distribution effects on load balancing

Usage:
    # Run from project root
    python examples/load-balancing/consistent_hashing_basics.py
    python examples/load-balancing/fleet_change_comparison.py
    python examples/load-balancing/vnodes_analysis.py
    python examples/load-balancing/zipf_effect.py
"""

from .common import (
    CachingServer,
    CachingServerStats,
    CustomerRequestProvider,
    AggregateMetrics,
    collect_aggregate_metrics,
    compute_key_distribution,
    plot_hit_rate_comparison,
    plot_key_distribution,
    plot_fleet_change_impact,
    customer_id_key_extractor,
    create_customer_consistent_hash,
    create_customer_ip_hash,
)

__all__ = [
    "CachingServer",
    "CachingServerStats",
    "CustomerRequestProvider",
    "AggregateMetrics",
    "collect_aggregate_metrics",
    "compute_key_distribution",
    "plot_hit_rate_comparison",
    "plot_key_distribution",
    "plot_fleet_change_impact",
    "customer_id_key_extractor",
    "create_customer_consistent_hash",
    "create_customer_ip_hash",
]
