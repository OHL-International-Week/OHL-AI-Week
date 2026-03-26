# The Goalkeeper Scouting Playbook

**OH Leuven | Data-Driven Goalkeeper Recruitment**

---

## Executive Summary

**Which measurable performances of a goalkeeper in a lower league predict whether he will succeed at a higher level?**

**The answer: a goalkeeper's ability with his feet — not his hands — is what predicts career progression.** Across 693 goalkeepers from 40+ leagues (2019-2026), we found that distribution quality, defensive involvement, and ball-playing ability are the metrics that reliably separate keepers who progress to top leagues from those who don't. Shot-stopping metrics, the traditional measure of goalkeeper quality, show no significant difference between progressors and non-progressors — and are too noisy to scout on even if they did.

Seven core metrics pass all four tests (statistically significant, high model importance, reliable match-to-match, and not confounded by team/league effects). A model using only these core metrics plus context features achieves AUC-ROC of 0.80 — meaning it correctly ranks a progressing keeper above a non-progressing keeper 80% of the time.

PLAYS keepers pass an average of 4 out of 7 core thresholds. Non-progressors pass 2-3. A keeper who exceeds 5+ thresholds is a strong candidate for higher-level recruitment.

---

## The Core Predictors

These 7 metrics should be the foundation of data-driven goalkeeper scouting. Each one is: statistically significant (p < 0.05 after correction), important in prediction models (SHAP + tree importance), reliable across matches (CV < 0.65), and not a proxy for team/league quality.

### 1. Defensive IMPECT Score
**What it measures:** Impect's Defensive IMPECT score includes all defensive Packing KPIs — it quantifies how many opponents a goalkeeper effectively removes from play through defensive actions (interceptions, recoveries, duels won). Each action is weighted by its empirical influence on preventing goals.

**Why it predicts progression:** Keepers who progress are not passive shot-waiters. They actively read the game, intercept through-balls, and intervene before shots happen. This is the strongest single predictor of progression.

**Threshold:** Progressors median = 0.251, Non-progressors median = 0.230. Target: above 0.248 (65th percentile). Cohen's d = 0.39 (medium effect).

### 2. Total Touches
**What it measures:** Sum of all events in defensive play (directly after the opposition is in possession) and events in offensive play (whilst the team is in possession). Counts every ball contact including passes, goal kicks, catches, and punches.

**Why it predicts progression:** Progressing keepers are involved in build-up play. They receive back-passes, initiate attacks, and function as an extra defender in possession. A keeper with 31+ touches per match is more involved than one with 28.

**Threshold:** Progressors median = 31.0, Non-progressors median = 28.4. Target: above 30.9 (65th percentile). Cohen's d = 0.38.

### 3. Passing Accuracy (Pass Success Rate)
**What it measures:** The percentage of successful passes in relation to the total number of passes, excluding neutral passes. Includes low passes, diagonal passes, crosses, and set pieces. The most reliable metric in the dataset (CV = 0.17 — very stable match-to-match).

**Why it predicts progression:** Completing passes is the foundation of ball-playing ability. Teams building from the back need a goalkeeper they can trust under pressure. An accuracy above 80% indicates a keeper comfortable with the ball.

**Threshold:** Progressors median = 80.1%, Non-progressors median = 77.1%. Target: above 77.4% (49th percentile). Cohen's d = 0.28.

### 4-6. IMPECT Score (Overall / Without Goals / With Post-Shot xG)
**What it measures:** Three variants of Impect's overall composite score. All include 13 Packing KPIs (measuring how many opponents are bypassed through passes, dribbles, and other actions), weighted by their empirical influence on goal-scoring probability. The "without goals" variant excludes goals scored; the "with post-shot xG" variant replaces actual goals with post-shot expected goals for a smoother signal. All three are highly correlated (r > 0.99).

**Why it predicts progression:** These capture the total package — a keeper who influences the game across all phases. Use whichever variant is available.

**Threshold:** Progressors median = 0.344, Non-progressors median = 0.322. Target: above 0.342 (60th percentile). Cohen's d = 0.38.

### 7. Successful Launches % (Goal Kick Success Rate)
**What it measures:** The ratio of successful goal kicks to all goal kicks (Impect definition: `Successful Goal Kicks / (Successful + Unsuccessful)`). Measures how often a goalkeeper's goal kicks reach a teammate.

**Why it predicts progression:** Goal kick quality is a key tactical weapon. A keeper who can reliably find teammates from goal kicks gives his team an advantage in restarting play. Accuracy above 42% is the benchmark.

**Threshold:** Progressors median = 42.4%, Non-progressors median = 39.8%. Target: above 41.8% (60th percentile). Cohen's d = 0.24.

---

## Supporting Indicators

These metrics add value but come with caveats. Use them to enrich the picture, not as standalone criteria.

| Metric | What It Adds | Caveat |
|--------|-------------|--------|
| **Caught High Balls %** | Percentage of opposition high balls (from corners, crosses, free kicks in final third) that the GK catches. Aerial dominance — the only traditional GK skill that predicts progression (d=0.34) | Higher match-to-match variance (CV=0.96); need a full season to be reliable |
| **Low Pass Score** | Composite score covering all KPIs related to low passes (foot-played passes reaching max chest height). Includes bypassed opponents, ball losses, and completion from short distribution | Reliable (CV=0.53) but moderate effect size (d=0.23) |
| **Diagonal Pass Score** | Composite score for high balls played from centre/wide that switch play. Includes completion, bypassed opponents, and ball losses from diagonal distribution | Higher variance (CV=1.75); use cautiously |
| **Goal Kick Score** | Composite score specifically for the goal kick set piece — covers bypassed opponents, successful/unsuccessful outcomes, and xG created from goal kicks | Not statistically significant for progression, but SHAP ranks it 4th — suggests non-linear effects |
| **Foot Usage Ratio** | Ratio of passes made with right (or left) foot to total passes. Measures two-footedness | SHAP ranks it #1 but not statistically significant — likely captures tactical style rather than quality |
| **Defensive Touches Outside Box** | Total defensive ball contacts outside the penalty area per match. Proxy for sweeper-keeper activity | Moderate variance (CV=1.07); needs at least 10+ matches to measure |
| **Caught + Punched High Balls %** | Broader than Caught % — includes punched clearances in addition to catches, as a share of total opposition high balls | Borderline significance (p=0.055) |

---

## Discarded Metrics — What NOT to Scout On

These metrics looked promising on paper but failed the data tests. Understanding why they failed is just as valuable as knowing what works.

### All Shot-Stopping Metrics (Prevented Goals)
**Includes:** Prevented Goals (post-shot xG), Prevented Goals (shot-based xG), and all breakdowns by shot type (close range, mid range, long range, 1v1, headers).

**Why they fail:**
1. **Extreme noise:** Match-to-match coefficient of variation exceeds 50x for total prevented goals and 12-88x for shot-type breakdowns. A keeper's prevented goals in one match tells you almost nothing about their true ability.
2. **No discriminating power:** Not a single prevented goals metric is statistically significant for PLAYS vs REST (all p > 0.17). Shot-stopping simply does not distinguish keepers who progress from those who don't in this dataset.
3. **Sample size problem:** The literature suggests 150+ shots faced before shot-stopping becomes reliable. Most keepers in scouting windows face far fewer. Willis (2023) shows that even a 99th-percentile keeper can fluctuate between the 75th and 115th percentile in a single season.
4. **Floor effect:** Most professional goalkeepers are competent shot-stoppers. The variance between them is small compared to the variance in distribution skills.

**What to do instead:** Evaluate shot-stopping through video analysis, not event data. Data-driven scouting should focus on distribution and involvement.

### Pass Completion Over Expected
**Why it fails despite being statistically significant:** This metric has a Cohen's d of 0.29 (meaningful effect) and is significant (p=0.002), but its coefficient of variation is 63.8 — meaning it swings wildly from match to match. It's a genuine signal buried under too much noise to be useful for scouting windows of 5-30 matches.

---

## How to Use This Framework in Practice

### Quick Assessment (5 minutes per keeper)
Look at these three numbers:
1. **Defensive IMPECT Score** — above 0.248?
2. **Passing Accuracy** — above 77%?
3. **Total Touches** — above 31 per match?

If yes to all three: strong candidate. Proceed to full profile.

### Full Profile (15 minutes per keeper)
Check all 7 core metrics against thresholds. Count how many they exceed:
- **5-7 thresholds passed:** Strong candidate — prioritize for video review
- **3-4 thresholds passed:** Promising — investigate further
- **0-2 thresholds passed:** Below average profile for progression

### Context Adjustments
- **League strength matters.** A keeper in the Regionalliga (strength 0.15) who passes 5+ thresholds is a stronger signal than one in the Eredivisie (strength 0.69) doing the same, because the former is doing it in a weaker environment and still progresses.
- **Age matters.** Younger keepers (under 25) with good profiles are more likely to progress than older keepers with identical stats.
- **Minimum matches.** Require at least 5+ matches for any assessment. 10+ matches is preferred. Below 5, the data is too thin.

### What This Framework Cannot Tell You
- **Shot-stopping quality.** Use video for this.
- **Personality and mentality.** Character assessment remains the province of human scouts.
- **Tactical fit.** A keeper who is excellent at distribution may not fit a team that plays direct.
- **Injury history.** Not captured in performance data.
- **Future development trajectory.** The model predicts based on current performance, not potential.

---

## Model Performance

| Metric | Value | Interpretation |
|--------|-------|---------------|
| **AUC-ROC** | 0.797 | Correctly ranks a random PLAYS keeper above a random REST keeper 80% of the time |
| **F1 Score** | 0.400 | Identifies 33% of PLAYS keepers at 45% precision |
| **Baseline** | 14.3% | Random guessing would find PLAYS keepers 14% of the time |
| **Improvement over random** | 3.1x | The model's precision is 3x better than random |

The model is best used as a **screening tool**: reduce a universe of 300+ potential targets to a shortlist of 30-50, then apply traditional scouting methods to the shortlist.

### Validation
- The core+supporting model (20 features) actually outperforms the full model (30 features) with AUC 0.797 vs 0.769. Removing noisy shot-stopping features improves predictions.
- Results are consistent with Jamil et al. (2021), who found the same pattern (distribution > shot-stopping) using a completely different dataset and definition of success.

### Limitations
- **Sample size:** 99 PLAYS keepers and 31 BENCH keepers. The model cannot reliably distinguish "will play" from "will ride the bench" after transfer.
- **Temporal validity:** Trained on 2019-2026 data. The game evolves; what predicts progression today may shift over 5-10 years.
- **League coverage:** 40+ leagues represented, but some leagues have very few keepers. Results are strongest for European leagues.
- **Metric availability:** Some GK-specific scores (prevented goals breakdowns) are only available from recent seasons, creating systematic gaps in older data.
- **An honest non-result is still valuable:** Shot-stopping's failure to predict progression is itself a finding — it means data-driven GK scouting should focus elsewhere, not that shot-stopping doesn't matter in absolute terms.

---

## Methodology Summary

### Data
- 693 goalkeepers from 40+ leagues, seasons 2019-2026
- Impect player scores (134 composite scores per match)
- Origin match data only (performances BEFORE the keeper transferred)
- Labels: PLAYS (99), BENCH (31), STAYED (435), DROPPED (128)

### Four-Phase Analysis
1. **Phase 1 — KPI Weighting:** Consensus of 6 methods (Random Forest, XGBoost, Lasso, Mutual Information, Permutation Importance, Effect Size) to weight every KPI
2. **Q1 — Discriminating Metrics:** Mann-Whitney U tests with Benjamini-Hochberg FDR correction; Cohen's d effect sizes
3. **Q2 — Prediction Models:** Logistic Regression, Random Forest, XGBoost with 5-fold stratified CV; SHAP analysis for interpretability
4. **Q3 — Signal vs Noise:** Coefficient of variation, intra-class correlation, partial correlations controlling for league strength, confounding analysis

### Synthesis
Each KPI was scored on four dimensions (weight, statistical effect, SHAP importance, reliability) and classified as Core, Supporting, or Discarded based on passing all four tests.
