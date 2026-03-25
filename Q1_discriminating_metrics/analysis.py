#!/usr/bin/env python3
"""Q1 — Which metrics make the difference?

Determines which Impect player scores differ significantly between
goalkeepers who progress to top leagues and those who don't.

Methods:
- Mann-Whitney U tests (PLAYS vs REST) with Benjamini-Hochberg FDR correction
- Kruskal-Wallis H tests (all 4 categories)
- Cohen's d effect sizes
- Feature importance from Phase 1 KPI weights
- Violin/box plots and correlation heatmap

Run: python -m Q1_discriminating_metrics.analysis
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

from shared.data_utils import (
    load_definitions, load_and_aggregate_data, select_features,
    get_cache_path, STATUS_ORDER, META_COLS
)

OUTPUT = Path(__file__).resolve().parent / "results"
OUTPUT.mkdir(exist_ok=True)


def mann_whitney_tests(df_model, feature_cols):
    """PLAYS vs REST Mann-Whitney U tests with FDR correction."""
    print("\n--- Mann-Whitney U: PLAYS vs REST ---")

    score_cols = [c for c in feature_cols if c.startswith("mean_")]
    plays = df_model[df_model["status"] == "PLAYS"]
    rest = df_model[df_model["status"] != "PLAYS"]

    results = []
    for col in score_cols:
        p_vals = plays[col].dropna()
        r_vals = rest[col].dropna()
        if len(p_vals) < 5 or len(r_vals) < 5:
            continue

        u_stat, p_value = stats.mannwhitneyu(p_vals, r_vals, alternative="two-sided")

        # Cohen's d with pooled standard deviation
        pooled_std = np.sqrt(
            ((len(p_vals) - 1) * p_vals.std()**2 + (len(r_vals) - 1) * r_vals.std()**2)
            / (len(p_vals) + len(r_vals) - 2)
        )
        cohens_d = (p_vals.mean() - r_vals.mean()) / max(pooled_std, 1e-10)

        results.append({
            "feature": col.replace("mean_", ""),
            "feature_full": col,
            "plays_mean": p_vals.mean(),
            "plays_median": p_vals.median(),
            "rest_mean": r_vals.mean(),
            "rest_median": r_vals.median(),
            "mean_diff": p_vals.mean() - r_vals.mean(),
            "cohens_d": cohens_d,
            "abs_cohens_d": abs(cohens_d),
            "u_statistic": u_stat,
            "p_value_raw": p_value,
            "n_plays": len(p_vals),
            "n_rest": len(r_vals),
            "plays_higher": p_vals.mean() > r_vals.mean(),
        })

    stat_df = pd.DataFrame(results).sort_values("p_value_raw")

    # Benjamini-Hochberg FDR correction
    if len(stat_df) > 0:
        reject, p_adj, _, _ = multipletests(stat_df["p_value_raw"], method="fdr_bh")
        stat_df["p_value_fdr"] = p_adj
        stat_df["significant_fdr_005"] = p_adj < 0.05
        stat_df["significant_fdr_001"] = p_adj < 0.01
        stat_df["significant_raw_005"] = stat_df["p_value_raw"] < 0.05
    else:
        stat_df["p_value_fdr"] = []
        stat_df["significant_fdr_005"] = []

    n_sig_raw = stat_df["significant_raw_005"].sum()
    n_sig_fdr = stat_df["significant_fdr_005"].sum()
    print(f"\n  Significant (raw p < 0.05): {n_sig_raw} / {len(stat_df)}")
    print(f"  Significant (FDR-corrected p < 0.05): {n_sig_fdr} / {len(stat_df)}")
    print(f"\n  Top 15 discriminating features:")
    display_cols = ["feature", "cohens_d", "p_value_raw", "p_value_fdr",
                    "significant_fdr_005", "plays_higher"]
    print(stat_df[display_cols].head(15).to_string(index=False))

    stat_df.to_csv(OUTPUT / "mann_whitney_plays_vs_rest.csv", index=False)
    return stat_df


def kruskal_wallis_tests(df_model, feature_cols):
    """Kruskal-Wallis H tests across all 4 status categories."""
    print("\n--- Kruskal-Wallis H: All 4 Categories ---")

    score_cols = [c for c in feature_cols if c.startswith("mean_")]
    results = []

    for col in score_cols:
        groups = []
        for status in STATUS_ORDER:
            g = df_model[df_model["status"] == status][col].dropna().values
            if len(g) >= 3:
                groups.append(g)
        if len(groups) < 3:
            continue

        h_stat, p_value = stats.kruskal(*groups)
        results.append({
            "feature": col.replace("mean_", ""),
            "H_statistic": h_stat,
            "p_value": p_value,
            "significant_005": p_value < 0.05,
        })

    kw_df = pd.DataFrame(results).sort_values("p_value")

    # FDR correction
    if len(kw_df) > 0:
        _, p_adj, _, _ = multipletests(kw_df["p_value"], method="fdr_bh")
        kw_df["p_value_fdr"] = p_adj
        kw_df["significant_fdr"] = p_adj < 0.05

    print(f"\n  Significant (raw p < 0.05): {kw_df['significant_005'].sum()} / {len(kw_df)}")
    if "significant_fdr" in kw_df.columns:
        print(f"  Significant (FDR-corrected): {kw_df['significant_fdr'].sum()} / {len(kw_df)}")
    print(f"\n  Top 10:")
    print(kw_df.head(10).to_string(index=False))

    kw_df.to_csv(OUTPUT / "kruskal_wallis_all_categories.csv", index=False)
    return kw_df


def plot_violin_top_features(df_model, stat_df, n=8):
    """Violin plots for the top n discriminating features."""
    top = stat_df.head(n)
    top_cols = [f"mean_{f}" for f in top["feature"] if f"mean_{f}" in df_model.columns]
    if not top_cols:
        return

    ncols = 2
    nrows = (len(top_cols) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(top_cols):
        ax = axes[i]
        sns.violinplot(data=df_model, x="status", y=col, order=STATUS_ORDER,
                       ax=ax, inner="box", cut=0)
        feature_name = col.replace("mean_", "").replace("GK_", "")
        row = stat_df[stat_df["feature"] == col.replace("mean_", "")].iloc[0]
        ax.set_title(f"{feature_name[:40]}\n"
                     f"(d={row['cohens_d']:.2f}, p_raw={row['p_value_raw']:.4f}, "
                     f"p_fdr={row['p_value_fdr']:.4f})", fontsize=9)
        ax.set_xlabel("")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Top Discriminating Features: PLAYS vs REST\n"
                 "(Mann-Whitney U with FDR correction)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT / "violin_top_discriminating.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: violin_top_discriminating.png")


def plot_boxplots_by_status(df_model, stat_df, n=12):
    """Box plots of top features split by all 4 statuses."""
    top = stat_df.head(n)
    top_cols = [f"mean_{f}" for f in top["feature"] if f"mean_{f}" in df_model.columns]
    if not top_cols:
        return

    ncols = 3
    nrows = (len(top_cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.5 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(top_cols):
        ax = axes[i]
        sns.boxplot(data=df_model, x="status", y=col, order=STATUS_ORDER, ax=ax)
        name = col.replace("mean_", "").replace("GK_", "")[:35]
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Top Discriminating Metrics by Career Status", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT / "boxplots_by_status.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: boxplots_by_status.png")


def plot_correlation_heatmap(df_model, stat_df, n=15):
    """Correlation heatmap of the top significant features."""
    sig_features = stat_df[stat_df["significant_raw_005"]].head(n)
    cols = [f"mean_{f}" for f in sig_features["feature"] if f"mean_{f}" in df_model.columns]
    if len(cols) < 3:
        return

    corr = df_model[cols].corr()
    labels = [c.replace("mean_", "").replace("GK_", "")[:25] for c in cols]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, xticklabels=labels, yticklabels=labels,
                annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                ax=ax, linewidths=0.5, vmin=-1, vmax=1)
    ax.set_title("Correlation Between Top Discriminating Features", fontsize=13)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: correlation_heatmap.png")


def plot_effect_size_summary(stat_df):
    """Bar chart of effect sizes for all features, colored by significance."""
    df = stat_df.copy().sort_values("abs_cohens_d", ascending=True).tail(20)

    fig, ax = plt.subplots(figsize=(12, 9))
    colors = []
    for _, row in df.iterrows():
        if row.get("significant_fdr_001", False):
            colors.append("#1a9641")
        elif row.get("significant_fdr_005", False):
            colors.append("#a6d96a")
        elif row.get("significant_raw_005", False):
            colors.append("#fdae61")
        else:
            colors.append("#bdbdbd")

    ax.barh(range(len(df)), df["cohens_d"], color=colors)
    ax.set_yticks(range(len(df)))
    labels = [f.replace("GK_", "")[:35] for f in df["feature"]]
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Cohen's d (positive = PLAYS higher)")
    ax.set_title("Effect Sizes: PLAYS vs REST\n"
                 "Dark green = FDR p<0.01 | Light green = FDR p<0.05 | "
                 "Orange = raw p<0.05 | Grey = not significant",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUT / "effect_sizes.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: effect_sizes.png")


def integrate_phase1_weights(stat_df):
    """Load Phase 1 KPI weights and merge with statistical test results."""
    weights_path = Path(__file__).resolve().parent.parent / "kpi_weighting" / "output" / "kpi_weights_full.csv"
    if not weights_path.exists():
        print("  Phase 1 weights not found — run kpi_weighting/run.py first")
        return stat_df

    weights = pd.read_csv(weights_path)
    merged = stat_df.merge(
        weights[["feature_name", "consensus_weight", "rank"]],
        left_on="feature", right_on="feature_name", how="left"
    )
    merged.to_csv(OUTPUT / "discriminating_with_weights.csv", index=False)
    print(f"  Saved: discriminating_with_weights.csv (merged with Phase 1 weights)")
    return merged


def main():
    print("=" * 70)
    print("Q1 — WHICH METRICS MAKE THE DIFFERENCE?")
    print("=" * 70)

    # Load data
    score_defs, _, _ = load_definitions()
    cache = get_cache_path()
    dataset, df = load_and_aggregate_data(score_defs, cache_path=cache)
    df_model, feature_cols, gk_cols, general_cols = select_features(df, score_defs)

    # Statistical tests
    stat_df = mann_whitney_tests(df_model, feature_cols)
    kw_df = kruskal_wallis_tests(df_model, feature_cols)

    # Visualizations
    print("\n--- Visualizations ---")
    plot_violin_top_features(df_model, stat_df)
    plot_boxplots_by_status(df_model, stat_df)
    plot_correlation_heatmap(df_model, stat_df)
    plot_effect_size_summary(stat_df)

    # Integrate Phase 1 weights
    merged = integrate_phase1_weights(stat_df)

    # Summary
    print("\n" + "=" * 70)
    print("Q1 SUMMARY")
    print("=" * 70)
    n_sig = stat_df["significant_fdr_005"].sum() if "significant_fdr_005" in stat_df.columns else 0
    print(f"\n  {n_sig} features are significantly different between PLAYS and REST")
    print(f"  (after Benjamini-Hochberg FDR correction at alpha=0.05)")
    if n_sig > 0:
        sig = stat_df[stat_df["significant_fdr_005"]].sort_values("abs_cohens_d", ascending=False)
        print(f"\n  Significant features ranked by effect size:")
        for _, row in sig.iterrows():
            direction = "PLAYS higher" if row["plays_higher"] else "PLAYS lower"
            print(f"    {row['feature']:<45s}  d={row['cohens_d']:+.3f}  "
                  f"p_fdr={row['p_value_fdr']:.4f}  ({direction})")

    print(f"\n  All outputs saved to: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
