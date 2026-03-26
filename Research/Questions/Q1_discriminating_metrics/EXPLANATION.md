# Q1 — Which Metrics Make the Difference?

## The Question
Are there Impect player scores that differ significantly between goalkeepers who progress to top leagues and those who don't?

## Answer
**Yes.** Distribution and ball-playing ability — not shot-stopping — are the metrics that distinguish keepers who progress. This confirms the key finding from Jamil et al. (2021) using an independent dataset and a different definition of success.

The strongest discriminating metrics (after FDR correction) are:
1. **Defensive IMPECT Score** — progressing keepers have higher overall defensive impact
2. **Passing Accuracy / Pass Completion Over Expected** — they are better passers who complete difficult passes
3. **IMPECT Score (overall)** — higher general game impact
4. **Total Touches** — they are more involved in build-up play
5. **Successful Launches %** — accurate long distribution
6. **Caught High Balls %** — aerial dominance
7. **Low Pass / Diagonal Pass Score** — quality short and diagonal passing

Shot-stopping metrics (prevented goals, both pre- and post-shot xG) show **no significant difference** between PLAYS and REST keepers.

## Methodology

### Statistical Tests
- **Mann-Whitney U test** (non-parametric, two-sided): Tests whether the distribution of each metric differs between PLAYS (n=99) and REST (n=~580). Non-parametric was chosen because many metrics are not normally distributed.
- **Benjamini-Hochberg FDR correction**: Applied to correct for multiple comparisons (27+ tests). Controls the false discovery rate at 5%.
- **Kruskal-Wallis H test**: Tests for differences across all 4 categories (PLAYS, BENCH, STAYED, DROPPED).
- **Cohen's d effect size**: Standardized mean difference using pooled standard deviation. Measures practical significance beyond statistical significance.

### Why These Methods
- Mann-Whitney U is robust to non-normality and outliers, which are common in sports data
- FDR correction (vs. Bonferroni) was chosen because it is less conservative — appropriate when we have moderate number of tests and want to maximize discovery while controlling false positives
- Cohen's d provides interpretable effect magnitude: small (~0.2), medium (~0.5), large (~0.8)

### Visualizations
- Violin plots show full distribution shape per category (not just means)
- Box plots provide quick comparison of medians and IQR
- Correlation heatmap reveals which significant features measure overlapping concepts
- Effect size bar chart ranks all features by practical importance

## Key Findings Detail

### What does NOT predict progression
- **Prevented Goals (post-shot xG)**: p > 0.5, d ~ 0. Shot-stopping doesn't differentiate.
- **Prevented Goals (shot-based xG)**: Same — not significant.
- **Shot-stopping by type** (close/mid/long range, headers): Mostly non-significant.
- **Offensive IMPECT Score**: Not significant. Keepers don't need offensive impact.
- **Progression Score**: Not significant.

### Why shot-stopping doesn't discriminate
1. **Noise**: Shot-stopping metrics have CV > 50x, drowning signal in small samples
2. **Floor effect**: Most professional GKs are competent shot-stoppers; variance is small
3. **Selection bias**: Teams may already filter on basic shot-stopping before our dataset captures them

## Output Files
- `mann_whitney_plays_vs_rest.csv` — Full statistical test results with FDR-corrected p-values
- `kruskal_wallis_all_categories.csv` — Tests across all 4 categories
- `violin_top_discriminating.png` — Violin plots of top features
- `boxplots_by_status.png` — Box plots by career status
- `correlation_heatmap.png` — Correlations between significant features
- `effect_sizes.png` — Effect size bar chart with significance coloring
