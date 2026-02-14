"""Integration test for coffee shop simulation."""

from __future__ import annotations

from examples.coffee_shop import run_coffee_shop_simulation, CoffeeShopConfig


class TestCoffeeShopSimulation:

    def test_runs_to_completion(self):
        config = CoffeeShopConfig(duration_s=300.0, seed=42)
        result = run_coffee_shop_simulation(config)
        assert result.summary.total_events_processed > 0

    def test_customers_served(self):
        config = CoffeeShopConfig(duration_s=300.0, seed=42)
        result = run_coffee_shop_simulation(config)
        assert result.sink.count > 0

    def test_all_drink_types_routed(self):
        config = CoffeeShopConfig(duration_s=600.0, seed=88)
        result = run_coffee_shop_simulation(config)
        assert result.router.total_routed > 0
        assert result.router.dropped == 0
