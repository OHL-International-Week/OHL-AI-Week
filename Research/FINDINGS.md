# Goalkeeper Talent Identification — Findings

**OH Leuven | UCLL Project Week | March 2026**

---

## The Central Question

> Which measurable performances of a goalkeeper in a lower league predict whether he will succeed at a higher level?

We analysed 693 goalkeepers across 40+ leagues (seasons 2019–2026) using Impect player scores. Each keeper is labelled by what happened to their career: **PLAYS** (99) — transferred up and plays regularly, **BENCH** (31) — transferred up but sits on the bench, **STAYED** (435) — never moved, **DROPPED** (128) — dropped to a lower league.

---

## 1. Which Metrics Make the Difference?

### The answer: distribution and ball-playing ability, not shot-stopping

The single most important finding is that **a goalkeeper's ability with his feet — not his hands — is what distinguishes keepers who progress from those who don't.** This aligns with the key finding from Jamil et al. (2021), who reached the same conclusion using a completely different dataset and definition of success.

### The 11 statistically significant KPIs (PLAYS vs REST, Mann-Whitney U)

| Rank | KPI | PLAYS higher? | Effect Size | p-value |
|------|-----|---------------|-------------|---------|
| 1 | **Defensive IMPECT Score** | Yes | 0.39 | 0.000004 |
| 2 | **Passing Accuracy** | Yes | 0.28 | 0.0015 |
| 3 | **IMPECT Score (without goals)** | Yes | 0.38 | 0.0019 |
| 4 | **Pass Completion Over Expected** | Yes | 0.29 | 0.0019 |
| 5 | **IMPECT Score (with post-shot xG)** | Yes | 0.38 | 0.0020 |
| 6 | **IMPECT Score (overall)** | Yes | 0.38 | 0.0021 |
| 7 | **Total Touches** | Yes | 0.38 | 0.0027 |
| 8 | **Successful Launches %** | Yes | 0.24 | 0.011 |
| 9 | **Caught High Balls %** | Yes | 0.34 | 0.019 |
| 10 | **Low Pass Score** | Yes | 0.23 | 0.026 |
| 11 | **Diagonal Pass Score** | Yes | 0.26 | 0.027 |

**What this means in plain language:**

- **Defensive IMPECT** (p=0.000004): The strongest signal. Keepers who progress have measurably higher overall defensive impact — they influence the game more when the opponent has the ball.
- **Passing Accuracy + Pass Completion Over Expected** (p<0.002): Keepers who progress are better passers, and crucially, they exceed the expected pass completion rate for their situation. They don't just attempt easy passes — they complete difficult ones.
- **Total Touches** (p=0.003): Progressing keepers touch the ball more. They are more involved in build-up play, not just passive shot-stoppers.
- **Successful Launches %** (p=0.011): When they do go long, they find their target more often.
- **Caught High Balls %** (p=0.019): Aerial dominance — claiming crosses rather than punching or being beaten.
- **Low Pass Score + Diagonal Pass Score** (p<0.03): Short and diagonal passing quality both matter separately.

### What does NOT predict progression

- **Prevented Goals (post-shot xG)** — not significant (p=0.90). Shot-stopping, the traditional measure of GK quality, does not distinguish keepers who progress from those who don't.
- **Prevented Goals (shot-based xG)** — not significant (p=0.37).
- **Shot-stopping by shot type** (close range, mid range, headers) — mostly not significant.
- **Offensive IMPECT Score** — not significant (p=0.57). Goalkeepers don't need to contribute offensively beyond distribution.
- **Progression Score** — not significant (p=0.49).

### Why shot-stopping doesn't predict progression

This is not because shot-stopping doesn't matter — it clearly does at the highest level. The likely explanations are:

1. **Noise:** Shot-stopping metrics have extremely high match-to-match variance (coefficient of variation >50x). With typical sample sizes in our data (5–30 matches), the signal drowns in noise. The literature suggests 150+ shots faced before shot-stopping becomes reliable (Willis, 2023).
2. **Floor effect:** Most professional goalkeepers are competent shot-stoppers. The variance between them is small. Distribution quality has a much wider spectrum.
3. **Selection:** Teams scouting for higher leagues may already filter on basic shot-stopping ability, making it a non-differentiator in our dataset.

---

## 2. Can We Predict Progression?

### Yes — with caveats

We trained three models to predict whether a goalkeeper will progress (PLAYS vs REST):

| Model | F1 (PLAYS) | AUC-ROC | Interpretation |
|-------|------------|---------|----------------|
| Logistic Regression | 0.36 | 0.72 | Best recall (61%), finds most PLAYS keepers but with many false positives |
| Random Forest | 0.28 | 0.78 | Conservative — high precision but misses many PLAYS keepers |
| **XGBoost** | **0.40** | **0.77** | **Best balance — identifies 36% of PLAYS keepers at 45% precision** |

**What this means:** The model significantly outperforms random guessing (baseline: 14% PLAYS rate). An AUC of 0.77 means the model correctly ranks a random PLAYS keeper above a random REST keeper 77% of the time. This is a useful signal, not a crystal ball.

**For the UP vs NOT UP target** (PLAYS+BENCH vs STAYED+DROPPED, 130 vs 563):
- XGBoost achieved AUC=0.80 — the strongest result, because it has more positive examples.

**Multi-class** (all 4 categories): F1-macro ~0.47. The model can distinguish PLAYS, STAYED, and DROPPED reasonably well, but BENCH (n=31) is too small to predict.

### Feature importance across 4 methods

Features that consistently appear in the top 10 across Logistic Regression, Random Forest, XGBoost, and Permutation Importance:

| Feature | Methods in Top 10 | Category |
|---------|-------------------|----------|
| Origin league strength | 4/4 | Context |
| Number of matches | 4/4 | Context |
| Age | 3/4 | Context |
| Defensive IMPECT Score | 3/4 | **Performance** |
| Passing Accuracy | 3/4 | **Performance** |
| Foot Usage Ratio (R/L) | 3/4 | **Performance** |
| Successful Launches % | 2/4 | **Performance** |
| Caught High Balls % | 2/4 | **Performance** |
| Goal Kick Score | 2/4 | **Performance** |
| 1v1 Shot Stopping (PSxG) | 2/4 | **Performance** |
| Long Range Shot Stopping | 2/4 | **Performance** |

**Context vs Performance:** The model uses both. League strength and age are strong predictors (a 21-year-old in the Regionalliga is more likely to progress than a 32-year-old in the same league). But performance KPIs — especially defensive impact, passing, and aerial play — provide signal beyond what context alone offers.

---

## 3. What Is Noise and What Is Signal?

### Most reliable metrics (lowest match-to-match variation)

| Metric | Coefficient of Variation |
|--------|------------------------|
| Foot Usage Ratio | 0.11 |
| Passing Accuracy | 0.17 |
| Total Touches | 0.26 |
| Goal Kick Score | 0.42 |
| Low Pass Score | 0.53 |
| IMPECT Scores | 0.55 |
| Successful Launches % | 0.58 |

### Least reliable metrics (too noisy to scout on individually)

| Metric | Coefficient of Variation |
|--------|------------------------|
| Prevented Goals (post-shot xG) | 71.2 |
| Prevented Goals (shot-based xG) | 53.3 |
| Pass Completion Over Expected | 63.8 |
| Close Range Shot Stopping | 14.0 |
| 1v1 Shot Stopping | 13.4 |
| Header Shot Stopping | 12.8 |

**Key insight:** The most predictive metrics are also the most reliable ones. Passing accuracy, defensive IMPECT, and total touches are both strong predictors AND stable across matches. Shot-stopping metrics are both unreliable AND non-predictive — a double problem for scouting.

---

## 4. The Scouting Score (1–100)

### How it works

We combine two signals:

1. **Model Score (60%):** XGBoost trained on all 30 features (performance + context) to predict P(PLAYS). This captures complex non-linear interactions and uses league strength and age as context.

2. **Performance Score (40%):** A weighted composite of only the goalkeeper performance KPIs, using permutation importance weights. This ensures the score reflects actual goalkeeper quality, not just where they happen to play.

**Final Score** = 60% Model + 40% Performance, scaled to 1–100.

### Validation — the score works

| Status | Median Score | Mean Score | n |
|--------|-------------|------------|---|
| **PLAYS** | **86** | **85** | 99 |
| BENCH | 59 | 59 | 31 |
| DROPPED | 54 | 51 | 128 |
| STAYED | 41 | 41 | 435 |

PLAYS keepers score 45 points higher than STAYED keepers on median. The ordering is correct: PLAYS > BENCH > DROPPED > STAYED.

---

## 5. Recommendations for OHL

### What to prioritise when screening goalkeepers with data

1. **Defensive IMPECT Score** — the single strongest predictor. A keeper with an above-average Defensive IMPECT in a lower league is significantly more likely to succeed at a higher level. Weight this heavily.

2. **Passing metrics** — Passing Accuracy, Low Pass Score, Diagonal Pass Score, Pass Completion Over Expected. A keeper who is a reliable distributor has a proven edge. This aligns with modern football's demand for ball-playing goalkeepers.

3. **Total Touches + Successful Launches %** — keepers who are involved in play and accurate when going long. These are proxies for how much the team trusts the keeper with the ball.

4. **Caught High Balls %** — aerial dominance. The only traditional GK skill that shows up as a significant predictor.

5. **Do not over-weight shot-stopping metrics in data-driven scouting.** They are too noisy to be reliable with typical scouting sample sizes. Shot-stopping should be evaluated through video analysis, not event data alone.

### How to use the scouting score

- **Score > 80:** Strong candidate — very likely to have the statistical profile of a keeper who progresses. Prioritise for video review.
- **Score 60–80:** Promising — some positive signals, worth investigating further.
- **Score 40–60:** Average profile — no strong indicators either way.
- **Score < 40:** Below-average profile for progression.

The score is most useful as a **screening tool**: reduce 340 potential targets to a shortlist of 30–50, then apply traditional scouting methods (video, live viewing, character assessment) to the shortlist.

### Limitations to communicate

- The model is trained on historical data. The game evolves, and what predicted progression in 2020–2025 may shift.
- Small sample size for BENCH (31 keepers). We cannot reliably distinguish "will play" from "will ride the bench" after a transfer.
- Shot-stopping is under-represented in these findings due to metric noise, not because it doesn't matter. Consider supplementing with video-based shot-stopping assessments.
- The scouting score is a tool to inform decisions, not a replacement for expert judgement.

---

## Technical Appendix

### Data
- **Source:** Impect player scores (134 composite scores per match)
- **Population:** 693 goalkeepers, 40+ leagues, seasons 2019–2026
- **Features used:** 27 performance KPIs + 3 context features = 30 total
- **Training data:** Origin match data (performances BEFORE the keeper transferred)

### Models
- Logistic Regression (L2-regularised, C=0.1, balanced class weights)
- Random Forest (200 trees, max_depth=6, balanced weights)
- XGBoost (200 trees, max_depth=4, lr=0.05, scale_pos_weight)
- All evaluated with stratified 5-fold cross-validation

### Statistical tests
- Mann-Whitney U (non-parametric, two-sided) for PLAYS vs REST
- Kruskal-Wallis H for all 4 categories
- Effect size: Cohen's d

### Output files
- `scouting_scores.csv` — all 693 keepers with scores and breakdown
- `scouting_targets_ohl.csv` — filtered to leagues at or below Belgian Pro League level
- `kpi_weights.csv` — full KPI weight table with significance, direction, and reliability
- 15 visualisation plots in `output/`
