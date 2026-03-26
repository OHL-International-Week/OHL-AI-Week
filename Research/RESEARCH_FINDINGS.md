# Phase 0 — Research Findings: Goalkeeper Analytics & Talent Identification

## 1. Goalkeeper Performance Analysis Frameworks

### The Four-Pillar Model (Yam, 2019 — StatsBomb/MIT Sloan)
The de facto standard for structuring goalkeeper evaluation divides the role into four pillars:
1. **Shot stopping** — saving shots, measured via goals prevented vs xG/PSxG
2. **Cross collection** — claiming high balls from crosses and corners (aggression + success rate)
3. **Defensive activity** — sweeper actions, touches outside the box, 1v1 decisions
4. **Distribution** — passing accuracy, goal kick quality, long ball success, build-up involvement

This framework was validated by identifying David De Gea as the top PL GK in 2017-18 and matching his profile to Dean Henderson in England's third tier — who later progressed.

### The Goalkeeper Value Model (Lamberts, 2025)
A more recent five-score model built from on-ball event data. Highlights how composite scores (like Impect's) are constructed from raw data and the biases inherent in metric design.

### Key Insight for This Project
Our Impect data covers shot stopping and distribution well, has limited coverage of aerial play and sweeper activity, and virtually no data on positioning or communication (which require tracking/video data).

---

## 2. Common Scouting KPIs for Goalkeepers

### Shot Stopping
- **xG faced** — expected goals from shots faced (pre-shot model)
- **PSxG (Post-Shot Expected Goals)** — accounts for shot placement in the goal mouth. Conceptually stronger than pre-shot xG because it isolates the goalkeeper's save difficulty. Built via extreme gradient boosting on shot trajectory data (StatsBomb, 2018).
- **Goals Prevented / GSAA (Goals Saved Above Average)** — difference between xG/PSxG and actual goals conceded. The primary shot-stopping metric.
- **Save percentage** — basic ratio, heavily influenced by shot quality faced
- **PSxG is preferred** over shot-based xG because it accounts for where the shot is heading, isolating GK skill from defensive quality (Stats Perform/Opta, 2019).

### Distribution
- **Passing accuracy** — completion rate of all passes
- **Pass completion over expected** — how much a GK exceeds the expected completion rate given pass difficulty
- **Goal kick quality** — effectiveness of goal kicks
- **Long ball / launch success %** — accuracy of long distribution
- **Low pass score / diagonal pass score** — quality of short and diagonal passing

### Aerial Play
- **Cross claiming %** — percentage of high balls caught
- **Caught + punched high balls %** — broader aerial intervention metric

### Sweeper Activity
- **Defensive touches outside own box** — proxy for how proactively the GK covers space behind the defensive line

### Modern GK Demands
Modern football demands goalkeepers who function as an extra defender in build-up play, not just as shot-stoppers. Total touches, involvement in possession, and passing quality are increasingly valued.

---

## 3. The Impect Scoring System

Impect is a German football data analytics company that provides composite player scores combining multiple KPIs. Key characteristics:

- **Player Scores (134 total):** Pre-aggregated composite scores per match. Include GK-specific scores (IDs 164-192) covering prevented goals, aerial play, launches, and sweeper activity. Also include general scores for passing, progression, offensive/defensive impact.
- **Player KPIs (1,458 total):** Raw per-match KPIs covering bypassed opponents (packing), ball wins/losses, passes by type, shots, pressing actions, and expected threat (PxT).
- **Event KPIs (103 total):** Most granular level — KPI values per individual action (each pass, dribble, duel).
- **IMPECT Score (Packing):** Impect's flagship metric counts how many opponents are bypassed by an action. The overall IMPECT Score aggregates this across all actions.
- **Prevented Goals metrics:** Available in two variants:
  - Shot-based xG (ID 184/186): uses pre-shot expected goals
  - Post-shot xG (ID 164/166): uses post-shot expected goals (preferred)
  - Breakdown by shot type: long range, mid range, close range, 1v1, headers (IDs 167-171)
  - **Limitation:** GK_PREVENTED_GOALS scores are only available from 2025-2026 season onwards for some competitions, creating systematic missingness.

---

## 4. Feature Importance Methods in Sports Analytics

### SHAP (SHapley Additive exPlanations)
- Based on cooperative game theory (Shapley values)
- Provides per-prediction explanations: which features pushed the prediction up or down
- TreeExplainer is efficient for tree-based models (XGBoost, Random Forest)
- **Best for:** Interpreting individual predictions, understanding feature interactions
- Produces beeswarm plots, dependence plots, and feature importance bar charts

### Permutation Importance
- Model-agnostic: measures decrease in model performance when a feature is randomly shuffled
- Accounts for feature interactions (unlike single-tree importance)
- Can be applied to any fitted model
- **Best for:** Robust feature ranking, less prone to bias from correlated features

### Mutual Information
- Information-theoretic measure of dependence between feature and target
- Captures non-linear relationships (unlike correlation)
- Does not assume any distribution
- **Best for:** Initial feature screening, detecting non-linear signals

### Tree-Based Feature Importance
- Random Forest: Mean decrease in impurity (Gini importance)
- XGBoost: Gain-based importance (total reduction in loss from splits on each feature)
- **Caveat:** Biased toward high-cardinality and correlated features

### Logistic Regression Coefficients
- L1 (Lasso): drives unimportant coefficients to exactly zero — built-in feature selection
- L2 (Ridge): shrinks coefficients proportionally — all features retained
- **Best for:** Linear signal detection, interpretable direction of effect

### Consensus Approach (Used in This Project)
Normalize importance scores from multiple methods to [0,1], then average. Features consistently ranked high across methods are more likely to represent genuine signal rather than method-specific artifacts.

---

## 5. Signal vs Noise in Match-Level Goalkeeper Data

### The Sample Size Problem (Willis, 2023)
- Season-to-season variation in goals prevented per 90 is approximately +/-0.2
- A GK whose career average is at the 99th percentile can fluctuate between the 75th and 115th percentile in any given season
- **150+ shots faced** needed for a reliable signal; 300+ for confidence
- Many keepers in our dataset have far fewer than 150 shots faced in their origin period

### Mean Reversion (Stats Perform/Opta, 2019)
- Even top GKs (e.g., De Gea after a standout season) tend to regress toward average in subsequent seasons
- High prevented goals numbers in a single season may not reflect stable underlying skill

### Coefficient of Variation as Reliability Metric
- CV = standard deviation / |mean| across matches for each keeper
- Low CV = stable metric (e.g., passing accuracy: CV ~0.17)
- High CV = noisy metric (e.g., prevented goals PSxG: CV ~71)
- The most predictive metrics tend to also be the most reliable ones

### Confounding Factors
- **League quality:** +5% prevented goals in the Premier League != +5% in a lower league
- **Team possession:** Distribution metrics are partly dictated by coaching system, not individual skill
- **Opponent quality:** Shot-stopping stats are reactive and dictated by opposition
- **Sample size per keeper:** Keepers with very few matches have unreliable aggregate scores

### Intra-Class Correlation (ICC)
- Measures how much of total variance is between-keeper (signal) vs within-keeper (noise)
- High ICC = the metric reliably distinguishes between keepers
- Low ICC = match-to-match variation swamps between-keeper differences

---

## 6. Key Literature Finding: Distribution > Shot-Stopping

### Jamil et al. (2021) — Nature Scientific Reports
The most directly comparable study to this project. Key findings:
- Analyzed 14,671 player-match observations using Logistic Regression, Gradient Boosting, and Random Forest
- Classified GKs as elite (Champions League) vs sub-elite
- **15 common features across all three algorithms distinguished elite from sub-elite**
- These features were **almost entirely passing and distribution features**, not shot-stopping metrics
- Conclusion: a GK's ability with his feet, not his hands, distinguishes elite from sub-elite
- Test accuracy: LR 0.67, RFC 0.66, GBC 0.66

### Hernandez-Beltran et al. (2023)
Systematic review found only 12 relevant GK performance studies up to 2023, confirming the field is in early stages. Main performance indicators: avoided goals %, distribution quality, and number of offensive/defensive actions.

### Implications for This Project
- We should expect distribution and ball-playing metrics to outweigh shot-stopping in predicting progression
- Shot-stopping may not differentiate because: (a) noise drowns signal in small samples, (b) most professional GKs are competent shot-stoppers (floor effect), (c) teams already filter on basic shot-stopping ability
- Our project uses a different definition of success (career progression rather than CL participation), which provides an independent test of the Jamil et al. finding

---

## 7. Methodological Considerations for This Dataset

### Class Imbalance
- PLAYS: 99, BENCH: 31, STAYED: 435, DROPPED: 128
- BENCH is too small for reliable separate prediction
- Binary framing (PLAYS vs REST or UP vs NOT_UP) is more practical
- Use balanced class weights, SMOTE, or stratified sampling

### Missing Data Patterns
- GK_PREVENTED_GOALS scores are systematically missing for older seasons (not MCAR)
- Median imputation is acceptable for sporadic missingness but problematic for systematic patterns
- Features with >50% missing should be dropped or analyzed separately

### Multiple Comparison Correction
- With 27+ features tested, some significant results will be false positives
- Benjamini-Hochberg FDR correction is appropriate for this scale of testing
- Report both raw and adjusted p-values

### Recommended Binary Target
- Primary: PLAYS vs REST (BENCH + STAYED + DROPPED) — predicts which keepers will progress AND play
- Secondary: UP vs NOT_UP (PLAYS + BENCH vs STAYED + DROPPED) — predicts any upward transfer

---

## Sources

1. StatsBomb / Knutson (2018) — Intro to Goalkeeper Analysis
2. Yam, D. (2019) — A Data Driven Goalkeeper Evaluation Framework (MIT Sloan)
3. Jamil, M. et al. (2021) — Using multiple ML algorithms to classify elite and sub-elite goalkeepers (Nature Scientific Reports)
4. Willis, S. / Cannon Stats (2023) — Advanced Stats 101 (GK metric reliability)
5. Lamberts, M. (2025) — Introducing the Goalkeeper Value Model
6. StatsBomb (2018) — Post-Shot Expected Goals methodology
7. Stats Perform/Opta (2019) — Introducing Expected Goals on Target
8. Hernandez-Beltran, V. et al. (2023) — Systematic review of GK performance indicators
9. West, J. (2018) — A review of the key demands for a football goalkeeper
10. Musa, R.M. et al. (2023) — Data Mining and ML in Evaluating Goalkeepers
