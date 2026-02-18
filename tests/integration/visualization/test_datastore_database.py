"""Visual tests for database behavior."""

from __future__ import annotations

import numpy as np

from happysimulator.components.datastore import Database


class TestDatabaseVisualization:
    """Visual tests for database behavior."""

    def test_connection_pool_utilization(self, test_output_dir):
        """Visualize database connection pool behavior."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pool_sizes = [5, 10, 20, 50]
        num_queries = 200

        results = {}

        for pool_size in pool_sizes:
            db = Database(
                name="postgres", max_connections=pool_size,
                query_latency=0.005, connection_latency=0.001,
            )

            for i in range(num_queries):
                gen = db.execute(f"SELECT * FROM users WHERE id = {i}")
                try:
                    while True:
                        next(gen)
                except StopIteration:
                    pass

            results[pool_size] = {
                "connections_created": db.stats.connections_created,
                "queries": db.stats.queries_executed,
                "avg_latency": db.stats.avg_query_latency,
            }

        _fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax1 = axes[0, 0]
        ax1.bar(
            [str(s) for s in pool_sizes],
            [results[s]["connections_created"] for s in pool_sizes],
            color="steelblue", alpha=0.7,
        )
        ax1.set_xlabel("Pool Size")
        ax1.set_ylabel("Connections Created")
        ax1.set_title("Connections Created by Pool Size")
        ax1.grid(True, alpha=0.3, axis="y")

        db_tx = Database(name="db_tx", max_connections=10)
        tx_count = 50

        for i in range(tx_count):
            gen = db_tx.begin_transaction()
            tx = None
            try:
                while True:
                    next(gen)
            except StopIteration as e:
                tx = e.value

            for j in range(3):
                gen = tx.execute(f"UPDATE users SET name = 'user{i}_{j}' WHERE id = {j}")
                try:
                    while True:
                        next(gen)
                except StopIteration:
                    pass

            gen = tx.commit()
            try:
                while True:
                    next(gen)
            except StopIteration:
                pass

        ax2 = axes[0, 1]
        tx_stats = [
            db_tx.stats.transactions_started,
            db_tx.stats.transactions_committed,
            db_tx.stats.transactions_rolled_back,
        ]
        labels = ["Started", "Committed", "Rolled Back"]
        colors = ["#3498db", "#2ecc71", "#e74c3c"]
        ax2.bar(labels, tx_stats, color=colors, alpha=0.7)
        ax2.set_ylabel("Count")
        ax2.set_title("Transaction Statistics")
        ax2.grid(True, alpha=0.3, axis="y")

        ax3 = axes[1, 0]
        if db_tx.stats.query_latencies:
            ax3.hist(
                [l * 1000 for l in db_tx.stats.query_latencies],
                bins=20, color="purple", alpha=0.7, edgecolor="black",
            )
        ax3.set_xlabel("Latency (ms)")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Query Latency Distribution")
        ax3.grid(True, alpha=0.3)

        ax4 = axes[1, 1]
        ax4.axis("off")
        summary = f"""
Database Connection Pool Analysis

Test 1: Pool Size Impact
  - Pool sizes tested: {pool_sizes}
  - Queries per test: {num_queries}
  - Connections created varies with pool size

Test 2: Transaction Workload
  - Transactions: {tx_count}
  - Queries per transaction: 3
  - Started: {db_tx.stats.transactions_started}
  - Committed: {db_tx.stats.transactions_committed}
  - Rolled back: {db_tx.stats.transactions_rolled_back}

Query Statistics:
  - Total queries: {db_tx.stats.queries_executed}
  - Avg latency: {db_tx.stats.avg_query_latency * 1000:.2f}ms
  - P95 latency: {db_tx.stats.query_latency_p95 * 1000:.2f}ms

Connection Pool Benefits:
  - Reuses connections (avoids setup cost)
  - Limits concurrent connections
  - Queues requests when pool exhausted
"""
        ax4.text(
            0.1, 0.95, summary, transform=ax4.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox={"boxstyle": "round", "facecolor": "lightblue", "alpha": 0.5},
        )

        plt.suptitle("Database Connection Pool & Transaction Analysis", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(test_output_dir / "database_pool.png", dpi=150)
        plt.close()

        assert db_tx.stats.transactions_committed == tx_count

    def test_query_type_latency(self, test_output_dir):
        """Visualize latency differences by query type."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def query_latency(query: str) -> float:
            query_upper = query.upper()
            if query_upper.startswith("SELECT"):
                return 0.002 + np.random.exponential(0.001)
            if query_upper.startswith("INSERT"):
                return 0.003 + np.random.exponential(0.002)
            if query_upper.startswith("UPDATE"):
                return 0.004 + np.random.exponential(0.002)
            if query_upper.startswith("DELETE"):
                return 0.003 + np.random.exponential(0.001)
            return 0.005

        np.random.seed(42)
        db = Database(name="postgres", max_connections=20, query_latency=query_latency)

        query_types = {"SELECT": [], "INSERT": [], "UPDATE": [], "DELETE": []}

        for _ in range(100):
            for qtype in query_types:
                if qtype == "SELECT":
                    query = "SELECT * FROM users WHERE id = 1"
                elif qtype == "INSERT":
                    query = "INSERT INTO users (name) VALUES ('test')"
                elif qtype == "UPDATE":
                    query = "UPDATE users SET name = 'new' WHERE id = 1"
                else:
                    query = "DELETE FROM users WHERE id = 1"

                latency_before = len(db.stats.query_latencies)
                gen = db.execute(query)
                try:
                    while True:
                        next(gen)
                except StopIteration:
                    pass
                latency_after = len(db.stats.query_latencies)
                if latency_after > latency_before:
                    query_types[qtype].append(db.stats.query_latencies[-1])

        _fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        colors = {"SELECT": "#2ecc71", "INSERT": "#3498db", "UPDATE": "#e74c3c", "DELETE": "#9b59b6"}

        ax1 = axes[0, 0]
        data = [query_types[qt] for qt in query_types]
        bp = ax1.boxplot(data, tick_labels=list(query_types.keys()), patch_artist=True)
        for patch, qt in zip(bp["boxes"], query_types.keys(), strict=False):
            patch.set_facecolor(colors[qt])
            patch.set_alpha(0.7)
        ax1.set_ylabel("Latency (s)")
        ax1.set_title("Query Latency Distribution by Type")
        ax1.grid(True, alpha=0.3, axis="y")

        ax2 = axes[0, 1]
        avg_latencies = [np.mean(query_types[qt]) * 1000 for qt in query_types]
        bars = ax2.bar(
            query_types.keys(), avg_latencies,
            color=[colors[qt] for qt in query_types], alpha=0.7,
        )
        ax2.set_ylabel("Average Latency (ms)")
        ax2.set_title("Average Query Latency by Type")
        ax2.grid(True, alpha=0.3, axis="y")

        for bar, lat in zip(bars, avg_latencies, strict=False):
            ax2.annotate(
                f"{lat:.2f}ms", xy=(bar.get_x() + bar.get_width() / 2, lat + 0.1),
                ha="center", fontsize=9,
            )

        ax3 = axes[1, 0]
        throughputs = [1.0 / np.mean(query_types[qt]) for qt in query_types]
        ax3.bar(query_types.keys(), throughputs, color=[colors[qt] for qt in query_types], alpha=0.7)
        ax3.set_ylabel("Throughput (queries/s)")
        ax3.set_title("Theoretical Max Throughput by Type")
        ax3.grid(True, alpha=0.3, axis="y")

        ax4 = axes[1, 1]
        ax4.axis("off")
        summary = f"""
Query Performance Analysis

Queries Executed: {db.stats.queries_executed}
Queries per type: {len(query_types["SELECT"])} each

Latency Statistics (ms):
"""
        for qt in query_types:
            lats = [l * 1000 for l in query_types[qt]]
            summary += f"  {qt}: avg={np.mean(lats):.2f}, p50={np.percentile(lats, 50):.2f}, p99={np.percentile(lats, 99):.2f}\n"

        summary += """
Key Observations:
  - SELECT typically fastest (read-only)
  - UPDATE slowest (lock + write)
  - INSERT/DELETE similar (single write)
  - Variance depends on query complexity
"""
        ax4.text(
            0.1, 0.95, summary, transform=ax4.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox={"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.8},
        )

        plt.suptitle("Database Query Performance by Type", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(test_output_dir / "query_type_latency.png", dpi=150)
        plt.close()

        assert db.stats.queries_executed == 400
