# Finding the New Number 1

**OH Leuven International Week, March 2026**

## Central Question

> Which measurable on-pitch performances of a goalkeeper in a lower league predict whether he will succeed at a higher level?

## Three Learning Objectives

### 3.1 Which metrics make the difference?

**Distribution quality.** 45 KPIs show medium+ effect sizes (|Cohen's d| > 0.3) between progressors and non-progressors. The strongest:

| Category | Key KPIs | Effect Size | Meaning |
|----------|----------|------------|---------|
| Diagonal distribution | Successful diagonal passes | d = +0.50 | Line-breaking long passes |
| Attacking transitions | Passes in transition, bypassed opponents | d = +0.42 to +0.51 | Quick distribution after ball wins |
| Packing from deep | Bypassed opponents from first third | d = +0.38 to +0.47 | Eliminating players with one pass |
| Off-the-line play | Dribble distance, second ball wins | d = +0.41 to +0.46 | Sweeper-keeper behavior |
| Goal kicks (negative) | Unsuccessful goal kicks | d = -0.28 to -0.39 | Progressors play short |

Shot-stopping metrics do **not** differentiate progressors from non-progressors.

### 3.2 Can we predict progression?

**Yes.** Ensemble AUC = 0.787 (vs 0.500 random), using only on-pitch performance KPIs — no league strength, age, or team quality. Known progressors score a median of 78 vs 43 for STAYED keepers (35-point gap). The model is interpretable: it relies on the same distribution metrics identified in 3.1.

### 3.3 What is noise and what is signal?

**Signal:** 5 KPIs confirmed by Boruta as carrying real signal above random noise — all are passing/distribution metrics. 35% of selected KPIs are Tier 1 reliable (CV ≤ 0.5).

**Noise:** Shot-stopping KPIs have high match-to-match variance. Metrics correlating with team possession measure the team, not the keeper.

## Quick Start

```bash
pip install -r requirements.txt

# Step 1: Build the full KPI dataset (extracts 1,009 KPIs from match files, ~5 min)
python Pipeline/build_full_kpi_dataset.py

# Step 2: Run 6-method KPI experiments + data-driven feature count (~15 min)
python Pipeline/kpi_experiments.py

# Step 3: Run the scoring pipeline (~8 min)
python Pipeline/run_pipeline.py

# Step 4: Project higher-league performance (~2 min)
python Pipeline/projection.py
```

### Demo Notebook

```bash
cd Demo
jupyter notebook GK_Scouting_Demo.ipynb
```

## Dataset

- **693 goalkeepers** across **40+ leagues** (2019–2026)
- **1,009 raw KPIs** per keeper → 462 after filtering → **75 selected** (data-driven)
- Labelled by career outcome:
  - **PLAYS** (99): Transferred up and plays regularly
  - **BENCH** (31): Transferred up but benched
  - **STAYED** (435): Remained at same level
  - **DROPPED** (128): Moved to a lower league

## Results

| Status  | Median Score | n   |
|---------|-------------|-----|
| PLAYS   | 78          | 99  |
| BENCH   | 56          | 31  |
| STAYED  | 43          | 435 |
| DROPPED | 62          | 128 |

**Ensemble AUC: 0.787** (Random Forest + XGBoost, 5-fold CV, performance KPIs only)

## Project Structure

```
├── Pipeline/                              ← Production pipeline
│   ├── build_full_kpi_dataset.py          ← Extract 1,009 KPIs from match files
│   ├── kpi_experiments.py                 ← 6-method feature selection + data-driven N
│   ├── run_pipeline.py                    ← 3-step: find → weight → score
│   ├── projection.py                      ← Project higher-league performance
│   ├── build_projection_dataset.py        ← Load post-transfer KPIs
│   ├── generate_demo_assets.py            ← Generate exploration visualizations
│   └── output/                            ← All CSVs + visualizations (43 files)
├── Demo/
│   └── GK_Scouting_Demo.ipynb             ← Presentation notebook (47 cells)
├── Research/                              ← Earlier exploration & experiments
│   ├── Questions/                         ← Initial Q1/Q2/Q3 analyses
│   ├── KPIs/                              ← KPI weighting experiments
│   ├── Seperate KPI and Model Research/   ← Standalone model notebooks
│   ├── src/                               ← Modular research code
│   ├── shared/                            ← Shared data utilities
│   └── output/                            ← Research outputs
├── Docs/                                  ← Project briefing & literature
│   ├── Project briefing.docx
│   ├── Literature Review.docx
│   └── Kickoff.pptx
├── GK_Data/                               ← Raw data (693 keepers, match files)
├── requirements.txt
└── README.md
```

## Methodology

### KPI Selection (6 methods, data-driven justification)

| # | Method | Why? | Avg correlation with others |
|---|--------|------|----|
| 1 | XGBoost importance | Non-linear interactions (boosting) | 0.16 |
| 2 | Random Forest importance | Different non-linear approach (bagging) | 0.29 |
| 3 | Lasso (L1 regularization) | Linear perspective — catches different patterns | 0.10 |
| 4 | Mann-Whitney U + FDR | Statistical evidence with p-values and effect sizes | 0.26 |
| 5 | Boruta (50 iterations) | "Is this KPI better than random noise?" | 0.26 |
| 6 | Bootstrap stability (50 resamples) | "Is this KPI consistently selected?" | 0.17 |

Average pairwise correlation between methods: **0.21** — low, confirming each captures different signal.

### Data-Driven Feature Count

Tested 5 to 100 features with repeated 5-fold CV (3 seeds = 15 evaluations per point). The optimal N is the smallest where AUC reaches 99% of peak. Result: **75 features**.

### Scouting Score (1–100)

1. Select top 75 features by consensus weight
2. Train Random Forest + XGBoost ensemble with 5-fold CV
3. Out-of-fold probabilities (honest — each keeper scored by a model that never saw them)
4. Score = 60% model probability + 40% weighted KPI performance
5. Percentile-ranked to 1–100

### Higher-League Projection (Step 4)

Using 130 keepers who already transferred up, we measure how each KPI changes at the higher level:

- **3-layer approach:** retention rate (default) → linear regression → league-gap adjusted
- Best method per KPI chosen by 5-fold CV R²
- **Median retention = 1.00** — most distribution KPIs carry over unchanged
- 7/29 KPIs significantly change when stepping up (paired t-test)
- Projections are descriptive ("what typically happens") rather than precise individual predictions

### Signal vs Noise Tiers

| Tier | CV Range | Count | Scoutable? |
|------|----------|-------|------------|
| Tier 1 (Reliable) | ≤ 0.5 | 26 | Yes — consistent match-to-match |
| Tier 2 (Moderate) | 0.5–1.0 | 47 | Yes with 5+ matches |
| Tier 3 (Noisy) | > 1.0 | 2 | No — too much variance |
