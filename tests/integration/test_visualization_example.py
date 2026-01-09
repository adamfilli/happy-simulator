"""
Example test showing how to generate and save visualizations.

Run with: pytest tests/test_visualization_example.py -v
Output will be in: test_output/test_visualization_example/<test_name>/
"""

import pytest
from pathlib import Path


class TestVisualizationExample:
    """Example tests that generate output files."""

    def test_save_matplotlib_plot(self, test_output_dir: Path):
        """Generate a matplotlib plot and save it."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")  # Non-interactive backend for tests
        import matplotlib.pyplot as plt

        # Create sample data
        x = [1, 2, 3, 4, 5]
        y = [1, 4, 9, 16, 25]

        # Create and save plot
        fig, ax = plt.subplots()
        ax.plot(x, y, "b-o", label="y = x²")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Sample Visualization")
        ax.legend()
        ax.grid(True)

        plot_path = test_output_dir / "sample_plot.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

        # Verify file was created
        assert plot_path.exists()
        print(f"\nPlot saved to: {plot_path}")

    def test_save_raw_data_csv(self, test_output_dir: Path):
        """Save raw data to CSV for later analysis."""
        import csv

        data = [
            {"time": 0.0, "value": 100, "event": "start"},
            {"time": 1.5, "value": 150, "event": "spike"},
            {"time": 3.0, "value": 120, "event": "settle"},
            {"time": 5.0, "value": 100, "event": "end"},
        ]

        csv_path = test_output_dir / "simulation_data.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["time", "value", "event"])
            writer.writeheader()
            writer.writerows(data)

        assert csv_path.exists()
        print(f"\nData saved to: {csv_path}")

    def test_save_json_results(self, test_output_dir: Path):
        """Save structured results as JSON."""
        import json

        results = {
            "simulation_name": "basic_test",
            "duration_seconds": 10.0,
            "metrics": {
                "total_events": 1000,
                "avg_latency_ms": 5.2,
                "p99_latency_ms": 12.8,
            },
            "config": {
                "arrival_rate": 100,
                "service_time": 0.008,
            },
        }

        json_path = test_output_dir / "results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

        assert json_path.exists()
        print(f"\nResults saved to: {json_path}")

    def test_with_timestamped_output(self, timestamped_output_dir: Path):
        """
        Use timestamped directory to keep multiple runs.
        Useful for comparing results across different test runs.
        """
        summary_path = timestamped_output_dir / "summary.txt"
        summary_path.write_text("This run completed successfully.\n")

        assert summary_path.exists()
        print(f"\nTimestamped output: {timestamped_output_dir}")
