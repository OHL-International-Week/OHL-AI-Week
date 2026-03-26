#!/usr/bin/env python3
"""Q3 — What is noise and what is signal?

Identifies which metrics have too much match-to-match variance to be reliable,
and which correlate too heavily with confounding factors rather than actual
goalkeeper quality.

Methods:
- Coefficient of variation (match-to-match variance)
- Intra-class correlation (ICC) — between-keeper vs within-keeper variance
- Partial correlations controlling for league strength
- Confounding analysis (correlation with origin_median)
- Tier list classification

Run: python -m Q3_signal_vs_noise.analysis
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

try:
    import pingouin as pg
    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False
    print("WARNING: pingouin not installed. ICC analysis will be skipped.")

from shared.data_utils import (
    load_definitions, load_and_aggregate_data, select_features,
    get_cache_path, STATUS_ORDER, META_COLS
)

OUTPUT = Path(__file__).resolve().parent / "results"
OUTPUT.mkdir(exist_ok=True)


def compute_coefficient_of_variation(df_model, feature_cols):
    """Compute CV = mean(std) / |mean(mean)| for keepers with 3+ matches."""
    print("\n--- Match-to-Match Variance (Coefficient of Variation) ---")

    multi_match = df_model[df_model["n_matches_loaded"] >= 3]
    print(f"  Keepers with 3+ matches: {len(multi_match)}")

    score_cols = [c for c in feature_cols if c.startswith("mean_")]
    results = []

    for col in score_cols:
        std_col = col.replace("mean_", "std_")
        if std_col not in df_model.columns:
            continue

        mean_val = multi_match[col].mean()
        mean_std = multi_match[std_col].mean()
        median_std = multi_match[std_col].median()

        if abs(mean_val) > 1e-6:
            cv = mean_std / abs(mean_val)
        else:
            cv = float("inf")

        results.append({
            "feature": col.replace("mean_", ""),
            "feature_full": col,
            "mean_value": mean_val,
            "mean_match_std": mean_std,
            "median_match_std": median_std,
            "coeff_variation": cv,
        })

    cv_df = pd.DataFrame(results).sort_values("coeff_variation")
    cv_df = cv_df[cv_df["coeff_variation"] < float("inf")]

    print(f"\n  Most RELIABLE (lowest CV):")
    print(cv_df.head(10)[["feature", "coeff_variation", "mean_value", "mean_match_std"]].to_string(index=False))
    print(f"\n  Least RELIABLE (highest CV):")
    print(cv_df.tail(10)[["feature", "coeff_variation", "mean_value", "mean_match_std"]].to_string(index=False))

    cv_df.to_csv(OUTPUT / "coefficient_of_variation.csv", index=False)
    return cv_df


def compute_icc(df_model, feature_cols):
    """Compute Intra-Class Correlation for each metric.

    ICC measures how much of total variance is between-keeper (signal)
    vs within-keeper (noise). High ICC = reliable metric.

    Uses ICC(1,1) — one-way random effects, single measures.
    Requires match-level data; we approximate using mean and std.
    """
    if not HAS_PINGOUIN:
        print("\n--- ICC Analysis: SKIPPED (pingouin not installed) ---")
        return None

    print("\n--- Intra-Class Correlation (ICC) ---")

    score_cols = [c for c in feature_cols if c.startswith("mean_")]
    multi_match = df_model[df_model["n_matches_loaded"] >= 3].copy()

    results = []
    for col in score_cols:
        std_col = col.replace("mean_", "std_")
        if std_col not in df_model.columns:
            continue

        # Approximate ICC from mean and std
        # Between-keeper variance: var of keeper means
        between_var = multi_match[col].var()
        # Within-keeper variance: mean of keeper variances (std^2)
        within_var = (multi_match[std_col] ** 2).mean()

        if between_var + within_var > 0:
            icc_approx = between_var / (between_var + within_var)
        else:
            icc_approx = 0

        results.append({
            "feature": col.replace("mean_", ""),
            "icc_approx": icc_approx,
            "between_var": between_var,
            "within_var": within_var,
        })

    icc_df = pd.DataFrame(results).sort_values("icc_approx", ascending=False)

    print(f"\n  Highest ICC (most reliable — stable between keepers):")
    print(icc_df.head(10)[["feature", "icc_approx", "between_var", "within_var"]].to_string(index=False))
    print(f"\n  Lowest ICC (most noisy — match-to-match swamps between-keeper):")
    print(icc_df.tail(10)[["feature", "icc_approx", "between_var", "within_var"]].to_string(index=False))

    icc_df.to_csv(OUTPUT / "icc_analysis.csv", index=False)
    return icc_df


def compute_partial_correlations(df_model, feature_cols):
    """Partial correlations with progression status controlling for confounders."""
    print("\n--- Partial Correlations (controlling for league strength & matches) ---")

    score_cols = [c for c in feature_cols if c.startswith("mean_")]
    y = (df_model["status"] == "PLAYS").astype(float)

    results = []
    for col in score_cols:
        vals = df_model[col].dropna()
        if len(vals) < 20:
            continue

        # Simple correlation (unadjusted)
        r_simple, p_simple = stats.spearmanr(df_model[col], y)

        # Partial correlation controlling for origin_median and n_matches_loaded
        r_partial = np.nan
        p_partial = np.nan
        if HAS_PINGOUIN:
            temp = df_model[[col, "origin_median", "n_matches_loaded"]].copy()
            temp["target"] = y.values
            temp = temp.dropna()
            if len(temp) > 20:
                try:
                    partial = pg.partial_corr(
                        data=temp.reset_index(drop=True),
                        x=col, y="target",
                        covar=["origin_median", "n_matches_loaded"],
                        method="pearson"
                    )
                    r_partial = float(partial["r"].iloc[0])
                    p_partial = float(partial["p-val"].iloc[0])
                except Exception as e:
                    pass

        results.append({
            "feature": col.replace("mean_", ""),
            "r_simple": r_simple,
            "p_simple": p_simple,
            "r_partial": r_partial,
            "p_partial": p_partial,
            "r_drop": abs(r_simple) - abs(r_partial) if not np.isnan(r_partial) else np.nan,
        })

    pcorr_df = pd.DataFrame(results).sort_values("r_partial", ascending=False, key=abs, na_position="last")

    print(f"\n  Top 10 by partial correlation (after controlling for confounders):")
    print(pcorr_df.head(10)[["feature", "r_simple", "r_partial", "r_drop"]].to_string(index=False))
    print(f"\n  Features most affected by confounding (largest r_drop):")
    if "r_drop" in pcorr_df.columns:
        top_drop = pcorr_df.dropna(subset=["r_drop"]).nlargest(5, "r_drop")
        print(top_drop[["feature", "r_simple", "r_partial", "r_drop"]].to_string(index=False))

    pcorr_df.to_csv(OUTPUT / "partial_correlations.csv", index=False)
    return pcorr_df


def compute_confounding(df_model, feature_cols):
    """Correlation of each metric with origin_median (league quality confounder)."""
    print("\n--- Confounding Analysis (correlation with league strength) ---")

    score_cols = [c for c in feature_cols if c.startswith("mean_")]
    results = []

    for col in score_cols:
        vals = df_model[[col, "origin_median"]].dropna()
        if len(vals) < 20:
            continue
        r, p = stats.spearmanr(vals[col], vals["origin_median"])
        results.append({
            "feature": col.replace("mean_", ""),
            "r_league_strength": r,
            "abs_r_league": abs(r),
            "p_league": p,
            "confounded": abs(r) > 0.3,
        })

    conf_df = pd.DataFrame(results).sort_values("abs_r_league", ascending=False)

    n_confounded = conf_df["confounded"].sum()
    print(f"\n  Features with |r| > 0.3 with league strength: {n_confounded} / {len(conf_df)}")
    print(f"\n  Most confounded (high correlation with league strength):")
    print(conf_df.head(10)[["feature", "r_league_strength", "confounded"]].to_string(index=False))
    print(f"\n  Least confounded (independent of league):")
    print(conf_df.tail(10)[["feature", "r_league_strength", "confounded"]].to_string(index=False))

    conf_df.to_csv(OUTPUT / "confounding_league_strength.csv", index=False)
    return conf_df


def build_tier_list(cv_df, icc_df, pcorr_df, conf_df, stat_df_path=None):
    """Classify metrics into reliability tiers for scouting."""
    print("\n--- Building Metric Tier List ---")

    # Start with CV data
    tiers = cv_df[["feature", "coeff_variation"]].copy()

    # Merge ICC
    if icc_df is not None:
        tiers = tiers.merge(icc_df[["feature", "icc_approx"]], on="feature", how="left")

    # Merge confounding
    if conf_df is not None:
        tiers = tiers.merge(conf_df[["feature", "abs_r_league", "confounded"]], on="feature", how="left")

    # Merge partial correlations
    if pcorr_df is not None:
        tiers = tiers.merge(pcorr_df[["feature", "r_partial", "p_partial"]], on="feature", how="left")

    # Load statistical test results from Q1 if available
    q1_path = Path(__file__).resolve().parent.parent / "Q1_discriminating_metrics" / "results" / "mann_whitney_plays_vs_rest.csv"
    if q1_path.exists():
        stat_df = pd.read_csv(q1_path)
        tiers = tiers.merge(
            stat_df[["feature", "p_value_raw", "abs_cohens_d", "p_value_fdr"]],
            on="feature", how="left"
        )
    else:
        tiers["p_value_raw"] = np.nan
        tiers["abs_cohens_d"] = np.nan

    # Assign tiers
    def assign_tier(row):
        cv = row.get("coeff_variation", float("inf"))
        icc = row.get("icc_approx", 0)
        confounded = row.get("confounded", True)
        p = row.get("p_value_raw", 1.0)
        effect = row.get("abs_cohens_d", 0)

        # Tier 1: Low CV, high ICC, significant effect, not heavily confounded
        if cv < 1.0 and icc > 0.3 and not confounded and p < 0.05 and effect > 0.15:
            return "Tier 1 (Scout-ready)"

        # Tier 1 relaxed: significant with reasonable reliability
        if cv < 1.0 and p < 0.05 and effect > 0.2:
            return "Tier 1 (Scout-ready)"

        # Tier 2: Moderate reliability or moderate effect
        if cv < 2.0 and (p < 0.1 or effect > 0.1):
            return "Tier 2 (Use with caution)"

        if icc > 0.5 and not confounded:
            return "Tier 2 (Use with caution)"

        # Tier 3: everything else
        return "Tier 3 (Noise)"

    tiers["tier"] = tiers.apply(assign_tier, axis=1)

    # Summary
    for tier_name in ["Tier 1 (Scout-ready)", "Tier 2 (Use with caution)", "Tier 3 (Noise)"]:
        subset = tiers[tiers["tier"] == tier_name]
        print(f"\n  {tier_name}: {len(subset)} metrics")
        display_cols = ["feature", "coeff_variation"]
        if "icc_approx" in tiers.columns:
            display_cols.append("icc_approx")
        display_cols.extend(["abs_cohens_d", "confounded"])
        available = [c for c in display_cols if c in subset.columns]
        if len(subset) > 0:
            print(subset[available].to_string(index=False))

    tiers.to_csv(OUTPUT / "metric_tier_list.csv", index=False)
    return tiers


def plot_signal_vs_noise(cv_df, stat_df_path=None):
    """Scatter plot: CV (noise) vs effect size (signal)."""
    # Load effect sizes from Q1
    q1_path = Path(__file__).resolve().parent.parent / "Q1_discriminating_metrics" / "results" / "mann_whitney_plays_vs_rest.csv"
    if not q1_path.exists():
        print("  Q1 results not found — skipping signal vs noise plot")
        return

    stat_df = pd.read_csv(q1_path)
    merged = cv_df.merge(stat_df[["feature", "p_value_raw", "abs_cohens_d"]], on="feature", how="inner")

    if len(merged) < 5:
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(
        merged["coeff_variation"],
        merged["abs_cohens_d"],
        c=(-np.log10(merged["p_value_raw"].clip(lower=1e-10))).clip(0, 5),
        cmap="RdYlGn",
        s=80, alpha=0.7, edgecolors="grey", linewidth=0.5,
    )
    plt.colorbar(scatter, ax=ax, label="-log10(p-value)")

    ax.set_xlabel("Coefficient of Variation (match-to-match noise)", fontsize=12)
    ax.set_ylabel("Effect Size |Cohen's d| (PLAYS vs REST)", fontsize=12)
    ax.set_title("Signal vs Noise: Which Metrics Are Both Reliable AND Discriminating?", fontsize=13)

    # Annotate top features
    top = merged.nlargest(8, "abs_cohens_d")
    for _, row in top.iterrows():
        name = row["feature"].replace("GK_", "")[:25]
        ax.annotate(name, (row["coeff_variation"], row["abs_cohens_d"]),
                    fontsize=8, alpha=0.8, xytext=(5, 5),
                    textcoords="offset points")

    # Add quadrant labels
    ax.axhline(0.2, color="grey", linestyle=":", alpha=0.5)
    ax.axvline(1.0, color="grey", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT / "signal_vs_noise_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: signal_vs_noise_scatter.png")


def plot_cv_bar_chart(cv_df):
    """Bar chart of CV for all features."""
    df = cv_df.sort_values("coeff_variation", ascending=True).copy()
    df = df[df["coeff_variation"] < 100]  # exclude extreme outliers for readability

    fig, ax = plt.subplots(figsize=(12, max(8, len(df) * 0.35)))
    colors = ["#1a9641" if cv < 0.5 else "#a6d96a" if cv < 1.0 else "#fdae61" if cv < 3.0 else "#d7191c"
              for cv in df["coeff_variation"]]
    ax.barh(range(len(df)), df["coeff_variation"], color=colors)
    ax.set_yticks(range(len(df)))
    labels = [f.replace("GK_", "")[:30] for f in df["feature"]]
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Coefficient of Variation")
    ax.set_title("Match-to-Match Reliability of GK Metrics\n"
                 "Green = stable | Yellow = moderate | Red = noisy",
                 fontsize=12)
    ax.axvline(1.0, color="black", linestyle="--", alpha=0.5, label="CV = 1.0")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT / "cv_bar_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: cv_bar_chart.png")


def plot_icc_bar_chart(icc_df):
    """Bar chart of ICC for all features."""
    if icc_df is None or len(icc_df) == 0:
        return

    df = icc_df.sort_values("icc_approx", ascending=True).copy()

    fig, ax = plt.subplots(figsize=(12, max(8, len(df) * 0.35)))
    colors = ["#1a9641" if icc > 0.75 else "#a6d96a" if icc > 0.5 else "#fdae61" if icc > 0.25 else "#d7191c"
              for icc in df["icc_approx"]]
    ax.barh(range(len(df)), df["icc_approx"], color=colors)
    ax.set_yticks(range(len(df)))
    labels = [f.replace("GK_", "")[:30] for f in df["feature"]]
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("ICC (approximate)")
    ax.set_title("Intra-Class Correlation: Between-Keeper vs Within-Keeper Variance\n"
                 "Green = reliable (high ICC) | Red = noisy (low ICC)",
                 fontsize=12)
    ax.axvline(0.5, color="black", linestyle="--", alpha=0.5, label="ICC = 0.5")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT / "icc_bar_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: icc_bar_chart.png")


def plot_league_confounding(df_model):
    """Scatter: competition strength vs age, colored by status."""
    fig, ax = plt.subplots(figsize=(10, 7))
    for status in STATUS_ORDER:
        subset = df_model[df_model["status"] == status]
        ax.scatter(subset["origin_median"], subset["age"], alpha=0.5,
                   label=f"{status} (n={len(subset)})", s=40)
    ax.set_xlabel("Origin Competition Strength", fontsize=12)
    ax.set_ylabel("Age", fontsize=12)
    ax.set_title("Competition Strength vs Age by Career Status", fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT / "league_strength_vs_age.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: league_strength_vs_age.png")


def plot_tier_summary(tiers_df):
    """Visual summary of tier classification."""
    tier_counts = tiers_df["tier"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Pie chart
    ax = axes[0]
    colors_map = {
        "Tier 1 (Scout-ready)": "#1a9641",
        "Tier 2 (Use with caution)": "#fdae61",
        "Tier 3 (Noise)": "#d7191c",
    }
    labels = tier_counts.index.tolist()
    colors = [colors_map.get(l, "#bdbdbd") for l in labels]
    ax.pie(tier_counts.values, labels=labels, colors=colors, autopct="%1.0f%%",
           startangle=90, textprops={"fontsize": 10})
    ax.set_title("Metric Reliability Distribution", fontsize=13)

    # Bar chart by tier
    ax = axes[1]
    for i, tier_name in enumerate(["Tier 1 (Scout-ready)", "Tier 2 (Use with caution)", "Tier 3 (Noise)"]):
        subset = tiers_df[tiers_df["tier"] == tier_name]
        if len(subset) > 0:
            names = [f.replace("GK_", "")[:25] for f in subset["feature"]]
            ax.barh([f"{tier_name[:6]}: {n}" for n in names],
                    subset.get("abs_cohens_d", pd.Series([0]*len(subset))).fillna(0).values,
                    color=colors_map.get(tier_name, "#bdbdbd"), alpha=0.8)

    ax.set_xlabel("Effect Size |Cohen's d|")
    ax.set_title("Metrics by Tier and Effect Size", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT / "tier_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: tier_summary.png")


def main():
    print("=" * 70)
    print("Q3 — WHAT IS NOISE AND WHAT IS SIGNAL?")
    print("=" * 70)

    # Load data
    score_defs, _, _ = load_definitions()
    cache = get_cache_path()
    dataset, df = load_and_aggregate_data(score_defs, cache_path=cache)
    df_model, feature_cols, gk_cols, general_cols = select_features(df, score_defs)

    # Analyses
    cv_df = compute_coefficient_of_variation(df_model, feature_cols)
    icc_df = compute_icc(df_model, feature_cols)
    pcorr_df = compute_partial_correlations(df_model, feature_cols)
    conf_df = compute_confounding(df_model, feature_cols)

    # Build tier list
    tiers_df = build_tier_list(cv_df, icc_df, pcorr_df, conf_df)

    # Visualizations
    print("\n--- Visualizations ---")
    plot_signal_vs_noise(cv_df)
    plot_cv_bar_chart(cv_df)
    plot_icc_bar_chart(icc_df)
    plot_league_confounding(df_model)
    plot_tier_summary(tiers_df)

    # Summary
    print(f"\n{'=' * 70}")
    print("Q3 SUMMARY")
    print("=" * 70)
    for tier_name in ["Tier 1 (Scout-ready)", "Tier 2 (Use with caution)", "Tier 3 (Noise)"]:
        subset = tiers_df[tiers_df["tier"] == tier_name]
        print(f"\n  {tier_name}: {len(subset)} metrics")
        if len(subset) > 0:
            for _, row in subset.iterrows():
                cv = row.get("coeff_variation", "?")
                icc = row.get("icc_approx", "?")
                print(f"    - {row['feature']:<40s}  CV={cv:.2f}" +
                      (f"  ICC={icc:.2f}" if isinstance(icc, float) else ""))

    print(f"\n  All outputs saved to: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
