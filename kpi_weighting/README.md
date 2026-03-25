# Phase 1 — KPI Discovery & Weighting Module

## Overview

This module determines which goalkeeper KPIs matter most for predicting career progression from lower leagues to top leagues. It uses **data-driven methods only** — no domain assumptions are used to set weights.

## Methodology

### Data Ingestion
- Loads 693 goalkeepers from `gk_dataset_final.csv`
- Aggregates Impect player scores from per-match JSON files (origin matches only — performances BEFORE transfer)
- Filters: position == GOALKEEPER, matchShare >= 0.5
- Features: 13 GK-specific scores + 14 general scores + 3 meta features = 30 features
- Features with >50% missing data are dropped; remaining NaN filled with median

### Target Variable
Binary classification: **PLAYS** (99 keepers who transferred up and play regularly) vs **REST** (all others: BENCH + STAYED + DROPPED).

### Six Weighting Methods

| Method | Type | What It Measures |
|--------|------|-----------------|
| **Random Forest** | Tree-based | Gini importance — how often a feature is used for splits |
| **XGBoost** | Tree-based | Gain importance — total loss reduction from splits on each feature |
| **Logistic Regression (L1)** | Linear | Absolute coefficient magnitude after Lasso regularization |
| **Mutual Information** | Information-theoretic | Non-linear dependence between feature and target |
| **Permutation Importance** | Model-agnostic | Decrease in RF accuracy when feature is shuffled |
| **Effect Size (Cohen's d)** | Statistical | Standardized mean difference between PLAYS and REST |

### Consensus Weight
1. Each method's scores are normalized to [0, 1] (min-max scaling)
2. The six normalized scores are averaged per feature
3. The average is re-normalized to sum to 1

Features that rank high across multiple methods are more likely to represent genuine signal. A feature appearing in the top 10 of only one method may be a method-specific artifact.

### Validation
- 5-fold stratified cross-validation reports AUC-ROC and F1 for both Random Forest and XGBoost
- This validates that the features collectively contain predictive signal

## Output Files

| File | Description |
|------|-------------|
| `kpi_weights_full.csv` | Complete weight table: every KPI with consensus weight, per-method scores, rank, direction, p-value, effect size |
| `kpi_discovery.csv` | Coverage statistics for all KPIs |
| `kpi_consensus_weights.png` | Bar chart of top 20 performance KPIs by consensus weight |
| `kpi_method_comparison.png` | Heatmap showing top 15 KPIs across all 6 methods |
| `kpi_weight_distribution.png` | Distribution of weights across all KPIs |

## How to Run

```bash
cd "Project 1 - Finding the new number 1"
python -m kpi_weighting.run
```

## Model Choice Rationale

- **Random Forest + XGBoost:** Tree-based models handle non-linear relationships and interactions naturally, which is important for sports data where feature interactions (e.g., passing accuracy * league strength) can matter.
- **Logistic Regression (L1):** Provides interpretable linear coefficients and built-in feature selection via Lasso penalty. Identifies which features have a clean linear relationship with progression.
- **Mutual Information:** Captures non-linear dependencies without assuming any model form. Useful as a model-free sanity check.
- **Permutation Importance:** Model-agnostic and less biased by correlated features than tree-based importance.
- **Effect Size:** Directly measures the magnitude of difference between groups, independent of any model.

The consensus approach guards against over-reliance on any single method's biases.

## Interpretation

- **Consensus weight > 0.05:** Strong predictor of progression
- **Consensus weight 0.02–0.05:** Moderate predictor
- **Consensus weight < 0.02:** Weak or no predictive value
- **Direction "higher":** PLAYS keepers score higher on this metric
- **Direction "lower":** PLAYS keepers score lower (rare — mostly context features like being in lower leagues)
- **p-value < 0.05:** Statistically significant difference between PLAYS and REST
