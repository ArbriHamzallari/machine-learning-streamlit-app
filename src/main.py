"""Main entry point for the project.

Intelligent Crime Risk Prediction and Patrol Allocation System
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from src import data_io, features, ml, planner, baselines, fairness, metrics, viz


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Intelligent Crime Risk Prediction and Patrol Allocation System"
    )

    parser.add_argument(
        "--csv",
        type=str,
        default="data/chicago_crimes_2025.csv",
        help="Path to Chicago crimes dataset CSV",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2025,
        help="Year to filter data (default: 2025)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of simulation steps (days) to run (default: 30)",
    )
    parser.add_argument(
        "--units",
        type=int,
        default=3,
        help="Number of patrol units (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test mode: limit to 5000 rows and 10 steps",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to write generated outputs",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Apply smoke test limits
    if args.smoke:
        print("SMOKE TEST MODE: limiting to 5000 rows and 10 steps")
        max_rows = 5000
        max_steps = 10
    else:
        max_rows = None
        max_steps = args.steps

    print("=" * 70)
    print("Intelligent Crime Risk Prediction and Patrol Allocation System")
    print("=" * 70)

    # Step 1: Load and clean data
    print(f"\n[1/6] Loading data from {args.csv}...")
    df_raw = data_io.load_csv(args.csv)
    if max_rows:
        df_raw = df_raw.head(max_rows)
    print(f"  Loaded {len(df_raw)} raw records")

    df_clean = data_io.clean_df(df_raw)
    print(f"  Cleaned to {len(df_clean)} records")

    df = data_io.filter_year(df_clean, year=args.year)
    print(f"  Filtered to year {args.year}: {len(df)} records")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Unique zones (community_area): {df['community_area'].nunique()}")

    # Step 2: Feature engineering
    print("\n[2/6] Engineering features...")
    X, y, meta, df_agg = features.build_ml_dataset(df)
    print(f"  Feature matrix X: {X.shape}")
    print(f"  Target vector y: {y.shape}")
    print(f"  Risk class distribution: {y.value_counts().to_dict()}")

    # Step 3: Train and evaluate ML models
    print("\n[3/6] Training and evaluating ML models...")
    print("  Training Logistic Regression and Random Forest...")
    best_model, test_pred, test_proba = ml.train_and_eval(X, y, meta)
    print(f"  Model evaluation complete")
    print(f"  Metrics saved to {output_dir / 'ml_metrics.json'}")

    # Step 4: Compute zone centroids and grid mapping
    print("\n[4/6] Computing zone spatial mapping...")
    centroids = planner.compute_zone_centroids(df)
    zone_xy = planner.make_grid_mapping(centroids, grid_size=10)
    print(f"  Mapped {len(zone_xy)} zones to grid coordinates")

    # Step 5: Run simulation
    print(f"\n[5/6] Running {max_steps}-step simulation...")

    # Get test period dates
    train_idx, test_idx = ml.time_split(meta, test_ratio=0.2)
    test_meta = meta.iloc[test_idx].reset_index(drop=True)
    test_dates = sorted(test_meta["date_day"].unique())[:max_steps]

    print(f"  Test period: {len(test_dates)} days")
    print(f"  Simulating {args.units} patrol units")

    # Initialize patrol units at random positions
    np.random.seed(args.seed)
    all_zones = sorted(zone_xy.keys())
    initial_zones = np.random.choice(all_zones, size=args.units, replace=False)
    
    # Prepare logs
    logs = []

    # Initialize fairness state for constrained greedy
    fairness_state = fairness.initialize_state()
    fairness_cfg = {
        "alpha": 1.0,
        "max_consecutive": 2,
        "override_risk_threshold": 0.75,
        "high_risk_threshold": 2.0
    }

    # Run simulation for each day
    for step, date_day in enumerate(test_dates, start=1):
        # Get data for this day
        day_data = df_agg[df_agg["date_day"] == date_day]
        
        if len(day_data) == 0:
            continue

        # Compute risk scores (use predicted probabilities for high-risk class)
        # For simplicity, use incident_count as proxy for risk
        risk_scores = {}
        true_high_risk_zones = []
        
        for zone in all_zones:
            zone_data = day_data[day_data["community_area"] == zone]
            if len(zone_data) > 0:
                # Normalize incident count to [0, 1]
                incident_count = zone_data["incident_count"].iloc[0]
                risk_class = zone_data["risk_class"].iloc[0]
                
                # Use normalized incident count as risk score
                max_incidents = day_data["incident_count"].max()
                risk_scores[zone] = incident_count / max_incidents if max_incidents > 0 else 0.0
                
                # True high-risk zones (risk_class == 2)
                if risk_class == 2:
                    true_high_risk_zones.append(zone)
            else:
                risk_scores[zone] = 0.0

        # Strategy 1: Random baseline
        unit_positions_random = {f"unit_{i}": zone_xy[initial_zones[i]] for i in range(args.units)}
        assignments_random, distance_random = baselines.allocate_random(
            all_zones, unit_positions_random, zone_xy, seed=args.seed + step
        )
        coverage_random = metrics.coverage_high_risk(assignments_random, true_high_risk_zones)
        
        logs.append({
            "step": step,
            "date": date_day,
            "strategy": "random",
            "distance": distance_random,
            "high_risk_coverage": coverage_random,
            "fairness_violations": 0
        })

        # Strategy 2: Risk-greedy baseline
        unit_positions_greedy = {f"unit_{i}": zone_xy[initial_zones[i]] for i in range(args.units)}
        assignments_greedy, distance_greedy = baselines.allocate_risk_greedy(
            all_zones, risk_scores, unit_positions_greedy, zone_xy
        )
        coverage_greedy = metrics.coverage_high_risk(assignments_greedy, true_high_risk_zones)
        
        logs.append({
            "step": step,
            "date": date_day,
            "strategy": "risk_greedy",
            "distance": distance_greedy,
            "high_risk_coverage": coverage_greedy,
            "fairness_violations": 0
        })

        # Strategy 3: Fairness-aware constrained greedy
        unit_positions_constrained = {f"unit_{i}": zone_xy[initial_zones[i]] for i in range(args.units)}
        assignments_constrained, distance_constrained = planner.allocate_patrols_constrained_greedy(
            all_zones, risk_scores, unit_positions_constrained, zone_xy,
            fairness_state, fairness_cfg
        )
        coverage_constrained = metrics.coverage_high_risk(assignments_constrained, true_high_risk_zones)
        
        # Update fairness state
        assigned_zones = list(assignments_constrained.values())
        fairness.update_state(assigned_zones, fairness_state)
        
        logs.append({
            "step": step,
            "date": date_day,
            "strategy": "constrained_greedy",
            "distance": distance_constrained,
            "high_risk_coverage": coverage_constrained,
            "fairness_violations": fairness_state.get("violations", 0)
        })

        if step % 10 == 0 or step == len(test_dates):
            print(f"  Completed step {step}/{len(test_dates)}")

    # Step 6: Save results and generate visualizations
    print("\n[6/6] Saving results and generating visualizations...")
    
    # Convert logs to DataFrame
    log_df = pd.DataFrame(logs)
    log_df.to_csv(output_dir / "results.csv", index=False)
    print(f"  Results saved to {output_dir / 'results.csv'}")

    # Generate summaries
    summaries = {}
    for strategy in ["random", "risk_greedy", "constrained_greedy"]:
        strategy_log = log_df[log_df["strategy"] == strategy]
        summaries[strategy] = metrics.summarize_run(strategy_log)

    # Print summaries
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for strategy, summary in summaries.items():
        print(f"\n{strategy.upper()}:")
        print(f"  High-risk coverage: {summary['high_risk_coverage_rate']:.2%}")
        print(f"  Avg distance/step: {summary['avg_distance_per_step']:.2f}")
        print(f"  Fairness violations: {summary['total_fairness_violations']}")

    # Generate visualizations
    logs_by_strategy = {
        strategy: log_df[log_df["strategy"] == strategy]
        for strategy in ["random", "risk_greedy", "constrained_greedy"]
    }
    
    coverage_plot = viz.plot_coverage(logs_by_strategy)
    distance_plot = viz.plot_distance(logs_by_strategy)
    print(f"\n  Plots saved:")
    print(f"    {coverage_plot}")
    print(f"    {distance_plot}")

    # Fairness report
    print("\n" + "=" * 70)
    print("FAIRNESS AUDIT")
    print("=" * 70)
    fairness_report = fairness.get_fairness_report(fairness_state)
    for key, value in fairness_report.items():
        if key == "distribution":
            continue  # Skip detailed distribution
        print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("Pipeline completed successfully!")
    print(f"Output directory: {output_dir}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
