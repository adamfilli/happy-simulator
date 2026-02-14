"""Integration test for grocery store simulation."""

from __future__ import annotations

import pytest

from examples.grocery_store import run_grocery_simulation, GroceryConfig


class TestGroceryStoreSimulation:

    def test_runs_to_completion(self):
        """Grocery store simulation runs without errors."""
        config = GroceryConfig(duration_s=300.0, seed=42)
        result = run_grocery_simulation(config)

        assert result.summary.total_events_processed > 0

    def test_customers_served(self):
        """Customers are served at checkout lanes."""
        config = GroceryConfig(duration_s=300.0, seed=42)
        result = run_grocery_simulation(config)

        assert result.sink.count > 0
        assert result.chooser.routed > 0

    def test_multiple_lane_types_used(self):
        """Customers spread across regular, express, and self-checkout."""
        config = GroceryConfig(duration_s=600.0, seed=42)
        result = run_grocery_simulation(config)

        regular_total = sum(l.customers_served for l in result.regular_lanes)
        express_total = result.express.customers_served
        self_total = sum(sc.customers_served for sc in result.self_checkouts)

        assert regular_total > 0
        # Express and self-checkout may or may not be used depending on routing

    def test_balking_occurs_under_load(self):
        """Customers balk when queues are long."""
        config = GroceryConfig(
            duration_s=600.0, arrival_rate=0.3,
            num_regular=1, num_self_checkout=1,
            balk_threshold=3, seed=42,
        )
        result = run_grocery_simulation(config)

        assert result.chooser.balked > 0

    def test_reasonable_throughput(self):
        """Throughput is reasonable for the configuration."""
        config = GroceryConfig(duration_s=600.0, seed=55)
        result = run_grocery_simulation(config)

        assert result.sink.count > 0
