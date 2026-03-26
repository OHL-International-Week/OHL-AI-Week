# Q3 — What Is Noise and What Is Signal?

## The Question
Which goalkeeper metrics have too much match-to-match variance to be reliable for scouting? Which correlate too heavily with confounding factors (team quality, league level) rather than actual goalkeeper quality?

## Answer
The most predictive metrics are also the most reliable ones. **Passing accuracy, defensive IMPECT, and total touches** are both strong predictors AND stable across matches. **Shot-stopping metrics** are both unreliable AND non-predictive — a double problem for scouting.

### Tier List Summary

**Tier 1 (Scout-ready):** Low match-to-match variance, statistically significant, not heavily confounded
- Passing Accuracy (CV ~0.17)
- Defensive IMPECT Score (CV ~0.55)
- Total Touches (CV ~0.26)
- IMPECT Score overall (CV ~0.55)
- Low Pass Score (CV ~0.53)

**Tier 2 (Use with caution):** Moderate reliability or moderate effect
- Successful Launches % (CV ~0.58)
- Goal Kick Score (CV ~0.42)
- Diagonal Pass Score
- Caught High Balls %

**Tier 3 (Noise):** High variance, low ICC, or heavily confounded
- Prevented Goals (post-shot xG) — CV > 70
- Prevented Goals (shot-based xG) — CV > 50
- Close Range / 1v1 / Header Shot Stopping — CV > 10
- Pass Completion Over Expected — CV > 60

## Methodology

### 1. Coefficient of Variation (CV)
**What:** CV = mean(std across matches) / |mean(value)| for keepers with 3+ matches
**Why:** Measures how stable a metric is match-to-match. A keeper's passing accuracy varies little game to game (CV ~0.17), while prevented goals swing wildly (CV > 50).
**Interpretation:** CV < 0.5 = very stable, 0.5-1.0 = moderate, > 1.0 = noisy, > 5.0 = essentially random per match.

### 2. Intra-Class Correlation (ICC)
**What:** Ratio of between-keeper variance to total variance (between + within). Approximated from mean and std columns.
**Why:** A metric is useful for scouting only if it reliably distinguishes between keepers. High ICC means most variance is between keepers (the metric captures something stable about the player). Low ICC means match-to-match noise dominates.
**Interpretation:** ICC > 0.75 = excellent reliability, 0.5-0.75 = good, 0.25-0.5 = fair, < 0.25 = poor.

### 3. Partial Correlations
**What:** Spearman correlation between each metric and progression status (PLAYS vs REST), controlling for league strength and number of matches played.
**Why:** Some metrics correlate with progression simply because better leagues have better values. Partial correlation removes this confounding. A metric that remains significant after controlling for league strength is genuinely measuring goalkeeper quality.
**Method:** Uses Spearman (rank-based) partial correlation via pingouin library.

### 4. Confounding Analysis
**What:** Spearman correlation of each metric with `origin_median` (league quality score).
**Why:** Metrics with |r| > 0.3 with league strength are substantially confounded — their values reflect where the keeper plays, not how good they are. These require normalization or context-adjustment before use in scouting.

### 5. Tier Classification
Combines all four analyses into a practical tier list:
- **Tier 1:** CV < 1.0, ICC > 0.3, statistically significant (p < 0.05), not confounded (|r_league| < 0.3), effect size > 0.15
- **Tier 2:** Moderate on most criteria, or strong on some but weak on others
- **Tier 3:** High CV, low ICC, or heavily confounded

## Key Insights

### Why shot-stopping fails as a scouting metric
1. **Extreme noise:** CV > 50 means a keeper's prevented goals value in one match tells you almost nothing about their true ability
2. **Sample size requirement:** Literature suggests 150+ shots faced for reliability; most keepers in our dataset face far fewer
3. **Mean reversion:** Even elite GKs regress toward average season-to-season
4. **Not predictive anyway:** Even if we could measure it perfectly, it doesn't distinguish PLAYS from REST keepers in this dataset

### What makes a metric "scout-ready"
A metric is useful for data-driven scouting when it:
- Is stable enough to measure in 5-30 matches (CV < 1.0)
- Captures something about the player, not the team (ICC > 0.3)
- Actually predicts progression (significant effect)
- Is not just a proxy for league quality (low confounding)

Passing accuracy, defensive IMPECT, and total touches satisfy all four criteria.

## Output Files
- `coefficient_of_variation.csv` — CV for all metrics
- `icc_analysis.csv` — ICC approximations
- `partial_correlations.csv` — Simple and partial correlations
- `confounding_league_strength.csv` — Correlations with league quality
- `metric_tier_list.csv` — Full tier classification with all criteria
- `signal_vs_noise_scatter.png` — CV vs effect size scatter plot
- `cv_bar_chart.png` — CV bar chart with color-coded reliability
- `icc_bar_chart.png` — ICC bar chart
- `league_strength_vs_age.png` — Context visualization
- `tier_summary.png` — Tier distribution and effect sizes
