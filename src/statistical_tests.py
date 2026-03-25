"""Statistical testing — Mann-Whitney U, Kruskal-Wallis, violin plots."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

from .config import OUTPUT, STATUS_ORDER


def run_statistical_tests(df_model, feature_cols_clean):
    """Run statistical tests and generate plot 06.

    Returns
    -------
    stat_df : DataFrame — PLAYS vs REST results
    kw_df : DataFrame   — Kruskal-Wallis results
    """
    print("\n" + "=" * 70)
    print("5. STATISTICAL TESTING")
    print("=" * 70)

    score_feature_cols = [c for c in feature_cols_clean if c.startswith("mean_")]

    # PLAYS vs REST (Mann-Whitney U)
    print("\n--- PLAYS vs REST (Mann-Whitney U test) ---")
    plays = df_model[df_model["status"] == "PLAYS"]
    rest = df_model[df_model["status"] != "PLAYS"]

    stat_results = []
    for col in score_feature_cols:
        p_vals = plays[col].dropna()
        r_vals = rest[col].dropna()
        if len(p_vals) < 5 or len(r_vals) < 5:
            continue
        u_stat, p_value = stats.mannwhitneyu(p_vals, r_vals, alternative="two-sided")
        effect_size = abs(p_vals.mean() - r_vals.mean()) / max(
            pd.concat([p_vals, r_vals]).std(), 1e-10
        )
        stat_results.append({
            "feature": col.replace("mean_", ""),
            "plays_mean": p_vals.mean(),
            "rest_mean": r_vals.mean(),
            "diff": p_vals.mean() - r_vals.mean(),
            "effect_size_d": effect_size,
            "p_value": p_value,
            "significant_005": p_value < 0.05,
            "significant_001": p_value < 0.01,
        })

    stat_df = pd.DataFrame(stat_results).sort_values("p_value")
    print(f"\nSignificant features (p < 0.05): {stat_df['significant_005'].sum()} / {len(stat_df)}")
    print(f"Highly significant (p < 0.01): {stat_df['significant_001'].sum()} / {len(stat_df)}")
    print(f"\nTop 20 most discriminating features (PLAYS vs REST):")
    print(stat_df.head(20).to_string(index=False))
    stat_df.to_csv(OUTPUT / "statistical_tests_plays_vs_rest.csv", index=False)

    # Kruskal-Wallis across all 4 categories
    print("\n--- Kruskal-Wallis across all 4 categories ---")
    kw_results = []
    for col in score_feature_cols:
        groups = [
            df_model[df_model["status"] == s][col].dropna().values
            for s in STATUS_ORDER
        ]
        groups = [g for g in groups if len(g) >= 3]
        if len(groups) < 3:
            continue
        h_stat, p_value = stats.kruskal(*groups)
        kw_results.append({
            "feature": col.replace("mean_", ""),
            "H_statistic": h_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
        })

    kw_df = pd.DataFrame(kw_results).sort_values("p_value")
    print(f"\nSignificant features (p < 0.05): {kw_df['significant'].sum()} / {len(kw_df)}")
    print(f"\nTop 15 features with significant category differences:")
    print(kw_df.head(15).to_string(index=False))
    kw_df.to_csv(OUTPUT / "statistical_tests_kruskal_wallis.csv", index=False)

    # Violin plots for top discriminating features
    top_features = stat_df.head(8)["feature"].tolist()
    top_cols = [f"mean_{f}" for f in top_features if f"mean_{f}" in df_model.columns]

    if top_cols:
        n_plots = len(top_cols)
        ncols = 2
        nrows = (n_plots + 1) // 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
        axes = axes.flatten()

        for i, col in enumerate(top_cols):
            ax = axes[i]
            sns.violinplot(data=df_model, x="status", y=col, order=STATUS_ORDER,
                           ax=ax, inner="box", cut=0)
            short_name = col.replace("mean_", "").replace("GK_", "")[:40]
            p_val = stat_df[stat_df["feature"] == col.replace("mean_", "")]["p_value"].values[0]
            ax.set_title(f"{short_name}\n(p={p_val:.4f})", fontsize=10)
            ax.set_xlabel("")

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Top Discriminating Features: PLAYS vs REST", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(OUTPUT / "06_top_discriminating_features.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: 06_top_discriminating_features.png")

    return stat_df, kw_df
