# Central Question — Methodology

## How the Sub-Questions Were Synthesized

### The Four Evidence Streams

Each KPI was evaluated independently by four analyses. The Central Question synthesis cross-references all four to identify metrics that consistently emerge as important.

| Stream | Question | Method | Output |
|--------|----------|--------|--------|
| **Phase 1** | Which KPIs matter most? | Consensus of 6 weighting methods | Normalized weight per KPI |
| **Q1** | Which metrics discriminate? | Mann-Whitney U + FDR correction + Cohen's d | Significance + effect size |
| **Q2** | Which features predict? | XGBoost SHAP values | SHAP importance rank |
| **Q3** | Which metrics are reliable? | CV + ICC + partial correlations | Reliability tier |

### Composite Scouting Score

For each performance KPI, a composite score was computed:

```
composite = 0.25 * norm(Phase1_weight)
           + 0.25 * norm(|Q1_cohens_d|)
           + 0.25 * norm(Q2_SHAP_value)
           + 0.25 * norm(1 - Q3_CV)
```

Each component is min-max normalized to [0,1] across all KPIs. Equal weighting (25% each) was chosen because no single evidence stream is inherently more trustworthy than the others.

### Classification Rules

**Core Predictor:** Must satisfy ALL of:
- Statistically significant (FDR-corrected p < 0.05) in Q1
- Classified as Tier 1 (Scout-ready) in Q3

**Supporting Indicator:** Must satisfy ANY of:
- Tier 1 or Tier 2 in Q3 AND composite score > 0.30
- Statistically significant AND composite score > 0.35

**Discarded:** Everything else — typically Tier 3 (Noise) metrics

### Key Disagreements Between Methods

1. **Foot Usage Ratio (Right/Left):** SHAP ranks it #1, but it's not statistically significant (p=0.79). Explanation: SHAP captures non-linear interactions (foot preference * league context), while Mann-Whitney tests marginal differences. The feature is important for the model but doesn't discriminate on its own. Classified as Supporting.

2. **Goal Kick Score:** SHAP ranks it #4, but not statistically significant. Similar explanation — goal kick quality interacts with league context in non-linear ways. Classified as Supporting.

3. **Pass Completion Over Expected:** Statistically significant (p=0.002, d=0.29) but classified as Tier 3 Noise (CV=63.8). This is a real signal drowning in noise. Despite being genuinely different between PLAYS and REST, the match-to-match variance is too high for practical scouting. Classified as Discarded.

4. **Long Range Shot Stopping:** Phase 1 ranks it #8 by consensus weight, but it's not significant in Q1 (p=0.18) and is Tier 3 Noise (CV=88.3). The high weight comes from XGBoost and Lasso picking up noise patterns. Classified as Discarded.

### Threshold Methodology

For each core predictor, optimal thresholds were determined via:

1. **Youden's J statistic:** Sweep across percentiles (10th to 90th in 5% steps), compute sensitivity + specificity - 1 at each threshold, select the maximum.
2. **Percentile interpretation:** Express the threshold as a percentile of the overall population for intuitive interpretation.
3. **Distribution comparison:** Overlay PLAYS and REST distributions to visually confirm the threshold separates groups.

### Validation Approach

The core+supporting model was validated with:
- **5-fold stratified cross-validation:** Each fold preserves the 14.3% PLAYS rate
- **Out-of-fold predictions:** All reported metrics are on data the model never trained on
- **Comparison to full model:** Tests whether removing noisy features hurts or helps

Result: Core+Supporting model (20 features) achieves AUC 0.797 vs full model (30 features) at AUC 0.769. Removing noisy shot-stopping features **improves** predictions, confirming they add noise, not signal.

### Case Study Selection

- **True positives:** PLAYS keepers with predicted probability > 0.3 (model would have flagged them)
- **False negatives:** PLAYS keepers with predicted probability < 0.15 (model would have missed them)
- **Hidden gems:** STAYED keepers with predicted probability > 0.3 (potential undiscovered talent)

## Reproducibility

All code is in `Central_Question/analysis.py`. Dependencies: pandas, numpy, scipy, scikit-learn, xgboost, matplotlib, seaborn. Data cached in `output/keeper_features.csv`.

Run: `python -m Central_Question.analysis`

All outputs saved to `Central_Question/results/`:
- `consolidated_table.csv` — Master cross-reference table
- `metric_classification.csv` — Core/Supporting/Discarded classification
- `thresholds.csv` — Actionable thresholds for core predictors
- `case_studies.csv` — All keepers with predicted probabilities
- 7 visualization plots
