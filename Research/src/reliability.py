"""Signal vs noise — metric reliability analysis."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import OUTPUT, STATUS_ORDER


def run_reliability_analysis(df_model, feature_cols_clean, stat_df):
    """Analyse match-to-match variance and plot signal vs noise.

    Returns
    -------
    rel_df : DataFrame or None
    """
    print("\n" + "=" * 70)
    print("8. LEAGUE STRENGTH ANALYSIS")
    print("=" * 70)

    fig, ax = plt.subplots(figsize=(10, 6))
    for status in STATUS_ORDER:
        subset = df_model[df_model["status"] == status]
        ax.scatter(subset["origin_median"], subset["age"], alpha=0.5, label=status, s=40)
    ax.set_xlabel("Origin Competition Strength")
    ax.set_ylabel("Age")
    ax.set_title("Competition Strength vs Age by Status")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT / "10_league_strength_vs_age.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 10_league_strength_vs_age.png")

    print("\n" + "=" * 70)
    print("9. SIGNAL vs NOISE: METRIC RELIABILITY")
    print("=" * 70)

    score_feature_cols = [c for c in feature_cols_clean if c.startswith("mean_")]
    std_cols = [c for c in df_model.columns if c.startswith("std_")]
    if not std_cols:
        return None

    multi_match = df_model[df_model["n_matches_loaded"] >= 3]
    print(f"\nKeepers with 3+ matches: {len(multi_match)}")

    reliability = []
    for col in score_feature_cols:
        std_col = col.replace("mean_", "std_")
        if std_col in df_model.columns:
            mean_val = multi_match[col].mean()
            mean_std = multi_match[std_col].mean()
            if abs(mean_val) > 1e-6:
                cv = mean_std / abs(mean_val)
            else:
                cv = float("inf")
            reliability.append({
                "feature": col.replace("mean_", ""),
                "mean_value": mean_val,
                "mean_match_std": mean_std,
                "coeff_variation": cv,
            })

    rel_df = pd.DataFrame(reliability).sort_values("coeff_variation")
    print("\nMost RELIABLE metrics (lowest match-to-match variation):")
    print(rel_df.head(10).to_string(index=False))
    print("\nLeast RELIABLE metrics (highest match-to-match variation):")
    print(rel_df.tail(10).to_string(index=False))
    rel_df.to_csv(OUTPUT / "metric_reliability.csv", index=False)

    # Signal vs noise scatter
    merged = rel_df.merge(stat_df[["feature", "p_value", "effect_size_d"]],
                          on="feature", how="inner")
    if len(merged) > 5:
        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(
            merged["coeff_variation"],
            merged["effect_size_d"],
            c=(-np.log10(merged["p_value"])).clip(0, 5),
            cmap="RdYlGn",
            s=60,
            alpha=0.7,
        )
        plt.colorbar(scatter, ax=ax, label="-log10(p-value)")
        ax.set_xlabel("Coefficient of Variation (match-to-match noise)")
        ax.set_ylabel("Effect Size (PLAYS vs REST)")
        ax.set_title("Signal vs Noise: Which Metrics Are Both Reliable AND Discriminating?")

        top_both = merged.nlargest(8, "effect_size_d")
        for _, row in top_both.iterrows():
            short_name = row["feature"].replace("GK_", "")[:25]
            ax.annotate(short_name, (row["coeff_variation"], row["effect_size_d"]),
                        fontsize=7, alpha=0.8)

        plt.tight_layout()
        plt.savefig(OUTPUT / "11_signal_vs_noise.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: 11_signal_vs_noise.png")

    return rel_df
