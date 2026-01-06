# Intelligent Crime Risk Prediction and Patrol Allocation System

**CEN352 – Artificial Intelligence | 2025-26**

---

## Slide 1: Problem & Motivation

**Challenge**: Allocate limited police patrol resources across urban zones

**Objectives**:
- Predict crime risk levels from historical data
- Optimize patrol allocation considering risk, distance, and fairness
- Prevent over-policing feedback loops

**Real-World Impact**: 
- Efficient resource allocation improves public safety
- Ethical constraints ensure equitable community treatment

---

## Slide 2: AI Techniques

**1. Supervised Machine Learning**
- Logistic Regression (baseline)
- Random Forest (main model, 500 trees)
- Multiclass classification: Low/Medium/High risk
- Time-based train/test split (80/20)

**2. Intelligent Agent Planning**
- Constrained greedy allocation algorithm
- Objective: maximize risk coverage, minimize distance
- Spatial modeling: 10×10 grid with Manhattan distance

**3. Fairness Constraints**
- Max 2 consecutive assignments per zone
- Prevents over-policing feedback loops
- Ethical safeguard for equitable distribution

---

## Slide 3: Dataset

**Source**: City of Chicago Crimes 2025

**Columns**:
- Temporal: `date`, `year`
- Spatial: `community_area` (77 zones), `latitude`, `longitude`
- Crime: `primary_type`, `arrest`, `domestic`

**Preprocessing**:
- Daily zone-level aggregations
- Feature engineering: temporal, historical, behavioral
- Risk labels: quantile-based (Low/Medium/High)

**Train/Test**: Chronological split (first 80% dates → train, last 20% → test)

---

## Slide 4: System Architecture

**Pipeline Flow**:
```
Raw Data → Cleaning → Feature Engineering → ML Training
    ↓
Risk Predictions → Spatial Mapping → Patrol Allocation Agent
    ↓
Evaluation (Coverage, Distance, Fairness)
```

**Key Components**:
- ML models predict risk scores per zone-day
- Planning agent allocates patrols using risk + distance
- Fairness module enforces ethical constraints
- Evaluation compares 3 strategies: random, risk-greedy, constrained-greedy

---

## Slide 5: Machine Learning Results

**Multiclass Performance** (Random Forest):
- Accuracy: **65.5%**
- Macro F1: **0.409**
- Precision: 0.452, Recall: 0.431

**Binary High-Risk Detection**:
- Accuracy: **88.5%**
- Precision: 0.222, Recall: 0.014

**Interpretation**: 
- 65.5% accuracy reflects real-world data noise and class overlap
- Random Forest outperforms Logistic Regression (F1: 0.409 vs. 0.267)
- High binary accuracy due to class imbalance (majority = low-risk)

---

## Slide 6: Patrol Allocation Results

**Performance Comparison** (17-day simulation):

| Strategy | Avg Coverage | Avg Distance | Violations |
|----------|-------------|--------------|------------|
| Random | 9.7% | 15.71 | 0 |
| Risk-Greedy | 12.3% | 11.88 | 0 |
| Constrained-Greedy | 9.7% | **0.06** | **0** |

**Key Findings**:
- Constrained-greedy achieves **near-zero distance** (0.06 units)
- Maintains competitive coverage (9.7%) with **zero fairness violations**
- Risk-greedy achieves higher coverage (12.3%) but higher distance (11.88)

**Figures**: See REPORT.md Figure 1 (coverage) and Figure 2 (distance)

---

## Slide 7: Ethics & Fairness

**Problem**: Predictive policing can create feedback loops
- More patrols → more arrests → higher predicted risk → more patrols
- Disproportionately affects marginalized communities

**Solution**: Fairness constraint
- Max 2 consecutive assignments per zone
- Override allowed for genuine high-risk (threshold: 0.75)
- Promotes equitable resource distribution

**Trade-off**: 
- Constrained-greedy: 9.7% coverage, 0 violations
- Risk-greedy: 12.3% coverage, potential violations
- Balance between performance and ethical deployment

---

## Slide 8: Demo Command

**Quick Test** (smoke mode):
```bash
python -m src.main --smoke
```
Runs with 5000 rows, 10 steps (~30 seconds)

**Full Pipeline**:
```bash
python -m src.main
```
Complete evaluation: ML training + 17-day simulation

**Outputs**:
- `outputs/ml_metrics.json`: Model performance
- `outputs/results.csv`: Simulation log
- `outputs/coverage.png`: Coverage visualization
- `outputs/distance.png`: Distance visualization

---

## Slide 9: Future Work

**Temporal Improvements**:
- Weekly/bi-weekly aggregations for stability
- Real-time incremental learning

**Feature Enhancement**:
- External factors: weather, events, socioeconomic indicators
- Road network routing (replace grid-based distance)

**Fairness Refinement**:
- Adaptive fairness thresholds
- Demographic parity metrics
- Community feedback integration

**Optimization**:
- Multi-objective Pareto analysis
- Coverage vs. distance vs. fairness trade-offs

---

## Slide 10: Conclusion

**Contributions**:
- Integrated ML prediction with planning agent
- Fairness-aware allocation prevents over-policing
- Real-world evaluation on Chicago crime data

**Results**:
- 65.5% multiclass accuracy, 0.409 macro F1
- Near-zero patrol distance (0.06 units)
- Zero fairness violations

**Impact**: Demonstrates how AI techniques can address complex resource allocation problems while incorporating ethical safeguards.

---

**Repository**: [GitHub Link](https://github.com/classroom/cen352-term-project-2025-26-albi-arbri)

