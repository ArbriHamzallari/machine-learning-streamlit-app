"""Evaluation metrics for patrol allocation strategies.

These metrics allow comparison of different allocation strategies across
multiple dimensions: effectiveness (risk coverage), efficiency (distance),
and fairness (constraint violations).
"""

import pandas as pd
from typing import Any


def coverage_high_risk(
    assignments: dict[str, int],
    true_high_risk_zones: list[int]
) -> float:
    """
    Calculate fraction of high-risk zones covered by patrol assignments.

    This metric measures effectiveness: how well does the allocation strategy
    identify and respond to genuinely high-risk areas?

    Args:
        assignments: Dict mapping unit_id -> assigned zone_id
        true_high_risk_zones: List of zone IDs that are actually high-risk

    Returns:
        Coverage rate in [0, 1]: fraction of high-risk zones that received patrols
    """
    if len(true_high_risk_zones) == 0:
        return 1.0  # No high-risk zones to cover

    # Get set of zones that received patrols
    covered_zones = set(assignments.values())

    # Count how many high-risk zones were covered
    high_risk_set = set(true_high_risk_zones)
    covered_high_risk = covered_zones.intersection(high_risk_set)

    coverage_rate = len(covered_high_risk) / len(high_risk_set)
    return coverage_rate


def avg_distance(distances: list[float]) -> float:
    """
    Calculate average distance traveled across multiple time steps.

    This metric measures efficiency: lower average distance means patrols
    are staying closer to their previous positions, reducing response time
    and fuel costs.

    Args:
        distances: List of total distances for each time step

    Returns:
        Average distance per time step
    """
    if len(distances) == 0:
        return 0.0

    return sum(distances) / len(distances)


def summarize_run(log_df: pd.DataFrame) -> dict[str, Any]:
    """
    Summarize a complete simulation run from a log DataFrame.

    Expected log_df columns:
        - step: time step number
        - strategy: name of allocation strategy
        - distance: total distance traveled at this step
        - high_risk_coverage: coverage rate at this step (0-1)
        - fairness_violations: cumulative violations at this step (optional)

    Args:
        log_df: DataFrame with simulation log

    Returns:
        Summary dict with:
            - high_risk_coverage_rate: mean coverage across all steps
            - avg_distance_per_step: mean distance per step
            - total_fairness_violations: final cumulative violations (if available)
            - num_steps: total number of time steps
    """
    if len(log_df) == 0:
        return {
            "high_risk_coverage_rate": 0.0,
            "avg_distance_per_step": 0.0,
            "total_fairness_violations": 0,
            "num_steps": 0
        }

    # Calculate metrics
    high_risk_coverage_rate = log_df["high_risk_coverage"].mean()
    avg_distance_per_step = log_df["distance"].mean()

    # Get final fairness violations (cumulative)
    if "fairness_violations" in log_df.columns:
        total_fairness_violations = int(log_df["fairness_violations"].iloc[-1])
    else:
        total_fairness_violations = 0

    num_steps = len(log_df)

    return {
        "high_risk_coverage_rate": float(high_risk_coverage_rate),
        "avg_distance_per_step": float(avg_distance_per_step),
        "total_fairness_violations": total_fairness_violations,
        "num_steps": num_steps
    }


def compare_strategies(
    summaries: dict[str, dict[str, Any]]
) -> pd.DataFrame:
    """
    Compare multiple strategies side-by-side.

    Args:
        summaries: Dict mapping strategy_name -> summary dict (from summarize_run)

    Returns:
        DataFrame with strategies as rows and metrics as columns
    """
    return pd.DataFrame(summaries).T


def compute_pareto_efficiency(
    strategies: dict[str, dict[str, float]],
    maximize_metrics: list[str],
    minimize_metrics: list[str]
) -> dict[str, bool]:
    """
    Identify Pareto-efficient strategies.

    A strategy is Pareto-efficient if no other strategy is strictly better
    on all metrics.

    Args:
        strategies: Dict mapping strategy_name -> {metric: value}
        maximize_metrics: List of metric names to maximize (e.g., coverage)
        minimize_metrics: List of metric names to minimize (e.g., distance)

    Returns:
        Dict mapping strategy_name -> is_pareto_efficient (bool)
    """
    pareto_efficient = {}

    for strategy_a, metrics_a in strategies.items():
        is_dominated = False

        # Check if any other strategy dominates strategy_a
        for strategy_b, metrics_b in strategies.items():
            if strategy_a == strategy_b:
                continue

            # Check if strategy_b dominates strategy_a
            # (better on all metrics, strictly better on at least one)
            better_count = 0
            worse_count = 0

            for metric in maximize_metrics:
                if metrics_b.get(metric, 0) > metrics_a.get(metric, 0):
                    better_count += 1
                elif metrics_b.get(metric, 0) < metrics_a.get(metric, 0):
                    worse_count += 1

            for metric in minimize_metrics:
                if metrics_b.get(metric, float('inf')) < metrics_a.get(metric, float('inf')):
                    better_count += 1
                elif metrics_b.get(metric, float('inf')) > metrics_a.get(metric, float('inf')):
                    worse_count += 1

            # If strategy_b is better on at least one metric and not worse on any
            if better_count > 0 and worse_count == 0:
                is_dominated = True
                break

        pareto_efficient[strategy_a] = not is_dominated

    return pareto_efficient


if __name__ == "__main__":
    # Demonstration of metrics
    print("Metrics Demonstration")
    print("=" * 60)

    # Test 1: Coverage metric
    print("\n[1] High-Risk Coverage")
    print("-" * 60)
    assignments = {"unit_1": 10, "unit_2": 20, "unit_3": 30}
    true_high_risk = [10, 20, 40, 50]  # 2 out of 4 covered
    coverage = coverage_high_risk(assignments, true_high_risk)
    print(f"Assignments: {assignments}")
    print(f"High-risk zones: {true_high_risk}")
    print(f"Coverage rate: {coverage:.2%}")

    # Test 2: Average distance
    print("\n[2] Average Distance")
    print("-" * 60)
    distances = [10, 15, 12, 8, 20]
    avg_dist = avg_distance(distances)
    print(f"Distances: {distances}")
    print(f"Average: {avg_dist:.2f}")

    # Test 3: Summarize run
    print("\n[3] Run Summary")
    print("-" * 60)
    log_data = {
        "step": [1, 2, 3, 4, 5],
        "strategy": ["greedy"] * 5,
        "distance": [10, 15, 12, 8, 20],
        "high_risk_coverage": [0.5, 0.75, 0.5, 1.0, 0.75],
        "fairness_violations": [0, 0, 1, 1, 2]
    }
    log_df = pd.DataFrame(log_data)
    summary = summarize_run(log_df)
    print("Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # Test 4: Compare strategies
    print("\n[4] Strategy Comparison")
    print("-" * 60)
    summaries = {
        "random": {
            "high_risk_coverage_rate": 0.45,
            "avg_distance_per_step": 15.0,
            "total_fairness_violations": 0
        },
        "risk_greedy": {
            "high_risk_coverage_rate": 0.85,
            "avg_distance_per_step": 20.0,
            "total_fairness_violations": 8
        },
        "constrained_greedy": {
            "high_risk_coverage_rate": 0.75,
            "avg_distance_per_step": 12.0,
            "total_fairness_violations": 2
        }
    }
    comparison = compare_strategies(summaries)
    print(comparison)

    # Test 5: Pareto efficiency
    print("\n[5] Pareto Efficiency Analysis")
    print("-" * 60)
    pareto = compute_pareto_efficiency(
        summaries,
        maximize_metrics=["high_risk_coverage_rate"],
        minimize_metrics=["avg_distance_per_step", "total_fairness_violations"]
    )
    print("Pareto-efficient strategies:")
    for strategy, is_efficient in pareto.items():
        status = "YES" if is_efficient else "NO"
        print(f"  {strategy}: {status}")
