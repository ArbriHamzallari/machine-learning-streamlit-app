# Intelligent Crime Risk Prediction and Patrol Allocation System

**Course:** CEN352 – Artificial Intelligence  
**Project Type:** Integrative AI System  
**Team:** Arbri Hamzallari , Albi Hoxha

## Project Overview

This project implements an end-to-end AI system that integrates supervised machine learning and intelligent agent planning to address crime risk prediction and patrol allocation. The system:

1. Uses supervised machine learning (Logistic Regression and Random Forest) to predict crime risk levels (Low/Medium/High) for each community area
2. Uses an intelligent agent (constrained greedy planning) to allocate patrol units based on predicted risk scores and spatial distance
3. Compares against baseline strategies (random and risk-greedy allocation)
4. Includes explicit fairness constraints to prevent over-policing and ensure equitable resource distribution

## AI Techniques

- **Supervised Machine Learning**: 
  - Logistic Regression (baseline model)
  - Random Forest (main model, 500 trees with balanced class weights)
  - Time-based train/test split (80/20, chronological)
  - OneHotEncoding for categorical `community_area` zones
  - Multiclass classification: Low (0), Medium (1), High (2) risk

- **Intelligent Agent Planning**: 
  - Constrained greedy allocation algorithm
  - Objective function: maximize `risk_score - α × distance` (α=1.0)
  - Spatial modeling: 10×10 grid mapping with Manhattan distance
  - Fairness constraints: max 2 consecutive assignments per zone

- **Fairness Constraints**: 
  - Prevents over-policing feedback loops
  - Max 2 consecutive patrol assignments per zone (unless risk ≥ 0.75)
  - Ethical safeguard for equitable resource distribution

- **No deep learning** (CNN, RNN, Transformers explicitly excluded per course requirements)

## Dataset

- **Source**: City of Chicago Crimes 2025 (local CSV file)
- **Location**: `data/chicago_crimes_2025.csv`
- **Required columns**: `date`, `primary_type`, `arrest`, `domestic`, `community_area`, `latitude`, `longitude`, `year`
- **Zones**: 77 community areas in Chicago
- **Time buckets**: Daily aggregations
- **Risk labels**: Quantile-based (50th, 80th percentiles) per zone

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure the dataset is available at `data/chicago_crimes_2025.csv`

**Required packages**: pandas, numpy, scikit-learn, matplotlib

## How to Run

### Smoke Test (Quick Validation)

```bash
python -m src.main --smoke
```

Runs with 5000 rows and 10 simulation steps (~30 seconds). Useful for quick validation.

### Full Pipeline

```bash
python -m src.main
```

Runs the complete pipeline:
- Data loading and cleaning
- Feature engineering
- ML model training (Logistic Regression + Random Forest)
- 17-day simulation with 3 strategies
- Results export and visualization

### Custom Configuration

```bash
python -m src.main --csv data/chicago_crimes_2025.csv --year 2025 --steps 30 --units 3 --seed 42
```

**Command-line Arguments**:
- `--csv`: Path to dataset (default: `data/chicago_crimes_2025.csv`)
- `--year`: Year to filter data (default: 2025)
- `--steps`: Number of simulation days (default: 30)
- `--units`: Number of patrol units (default: 3)
- `--seed`: Random seed for reproducibility (default: 42)
- `--smoke`: Smoke test mode (5000 rows, 10 steps)
- `--output-dir`: Output directory (default: `outputs`)

## Expected Outputs

All outputs are generated in the `outputs/` directory:

### `outputs/ml_metrics.json`
Machine learning model evaluation metrics:
- Logistic Regression performance (accuracy, precision, recall, F1, confusion matrix)
- Random Forest performance (accuracy, precision, recall, F1, confusion matrix)
- Best model selection (by macro F1 score)
- Binary high-risk detection metrics
- Train/test split sizes

### `outputs/results.csv`
Simulation log with columns:
- `step`: Simulation day number (1-17)
- `date`: Actual date (YYYY-MM-DD)
- `strategy`: Allocation strategy name (`random`, `risk_greedy`, `constrained_greedy`)
- `distance`: Total Manhattan distance traveled by all units
- `high_risk_coverage`: Fraction of high-risk zones covered (0.0-1.0)
- `fairness_violations`: Cumulative fairness violations (0 for all strategies in current run)

### `outputs/coverage.png`
Line plot showing high-risk zone coverage over time for all three strategies. Demonstrates how each strategy performs in identifying and covering high-risk areas.

### `outputs/distance.png`
Line plot showing average Manhattan distance traveled per day for all strategies. Shows the efficiency of each allocation approach.

## Project Structure

```
.
├── data/                    # Dataset directory (not committed)
│   └── chicago_crimes_2025.csv
├── outputs/                 # Generated results (not committed)
│   ├── ml_metrics.json
│   ├── results.csv
│   ├── coverage.png
│   └── distance.png
├── src/
│   ├── main.py              # Main pipeline entry point
│   ├── data_io.py           # Data loading and preprocessing
│   ├── features.py          # Feature engineering
│   ├── ml.py                # Machine learning models
│   ├── planner.py           # Patrol allocation planner
│   ├── fairness.py          # Fairness constraints
│   ├── baselines.py         # Baseline strategies
│   ├── metrics.py           # Evaluation metrics
│   └── viz.py               # Visualizations
├── tests/
│   └── test_smoke.py         # Smoke tests
├── requirements.txt
├── README.md
├── REPORT.md                # Final written report
└── SLIDES.md                # Presentation slides
```

## Team Member Roles

- **Albi Arbri**: System design, implementation, evaluation, and documentation

## Reproducibility

The pipeline is fully reproducible with the default seed (42). All random operations (data shuffling, baseline allocation) use this seed for consistent results.

**Key Design Decisions**:
- Time-based train/test split (no random splits) to prevent temporal data leakage
- Chronological ordering ensures models are trained on past data and tested on future data
- Zone-specific risk labeling (quantiles computed per `community_area`) accounts for zone heterogeneity
- Manhattan distance on 10×10 grid provides simplified but consistent spatial modeling

## Results Summary

**Machine Learning**:
- Random Forest achieves 65.5% multiclass accuracy with macro F1 of 0.409
- Binary high-risk detection accuracy: 88.5%
- Better class balance compared to Logistic Regression baseline

**Patrol Allocation**:
- Constrained-greedy achieves near-zero average distance (0.06 units)
- Maintains competitive coverage (9.7%) with zero fairness violations
- Demonstrates effective balance between risk coverage, distance optimization, and ethical constraints

For detailed results and analysis, see `REPORT.md`.
