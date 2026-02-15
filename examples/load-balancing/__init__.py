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
    AggregateMetrics,
    CachingServer,
    CachingServerStats,
    CustomerRequestProvider,
    collect_aggregate_metrics,
    compute_key_distribution,
    create_customer_consistent_hash,
    create_customer_ip_hash,
    customer_id_key_extractor,
    plot_fleet_change_impact,
    plot_hit_rate_comparison,
    plot_key_distribution,
)

__all__ = [
    "AggregateMetrics",
    "CachingServer",
    "CachingServerStats",
    "CustomerRequestProvider",
    "collect_aggregate_metrics",
    "compute_key_distribution",
    "create_customer_consistent_hash",
    "create_customer_ip_hash",
    "customer_id_key_extractor",
    "plot_fleet_change_impact",
    "plot_hit_rate_comparison",
    "plot_key_distribution",
]
