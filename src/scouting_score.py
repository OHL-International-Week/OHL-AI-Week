"""GK Scouting Score — rank every goalkeeper 1–100 for transfer potential.

Separates PERFORMANCE KPIs (what the keeper does on the pitch) from
CONTEXT features (league strength, age, sample size) so the scouting
score reflects actual goalkeeper quality, not just where they play.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import xgboost as xgb

from .config import OUTPUT, STATUS_ORDER, META_COLS


# ── Belgian Pro League threshold for OHL scouting targets ──────────────
OHL_LEAGUE_CEILING = 0.60  # origin_median ≤ this → realistic transfer target

# Context features — important for prediction but NOT goalkeeper skills
CONTEXT_FEATURES = set(META_COLS)  # age, origin_median, n_matches_loaded


def compute_scouting_scores(df_model, feature_cols_clean, stat_df, rel_df):
    """Compute a 1–100 scouting score for every goalkeeper.

    Methodology:
      1. Train XGBoost on ALL features (including context) → P(PLAYS)
      2. Build a performance-only composite from GK performance KPIs
      3. Ensemble: 60% model + 40% performance composite
      4. Report KPI weights separately for performance vs context

    Returns
    -------
    scores_df : DataFrame with scouting scores and breakdown
    """
    print("\n" + "=" * 70)
    print("11. GK SCOUTING SCORE")
    print("=" * 70)

    X = df_model[feature_cols_clean].copy().fillna(df_model[feature_cols_clean].median())
    y_binary = (df_model["status"] == "PLAYS").astype(int)

    # Separate performance features from context
    perf_cols = [c for c in feature_cols_clean if c not in CONTEXT_FEATURES]
    context_cols = [c for c in feature_cols_clean if c in CONTEXT_FEATURES]

    print(f"\n  Performance KPIs: {len(perf_cols)}")
    print(f"  Context features: {len(context_cols)} ({', '.join(context_cols)})")

    # ── Step 1: KPI Weight Report ──────────────────────────────────────
    print("\n--- KPI Weights (Permutation Importance) ---")

    rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42,
        max_depth=6, min_samples_leaf=5,
    )
    rf.fit(X, y_binary)
    perm_imp = permutation_importance(rf, X, y_binary, n_repeats=20, random_state=42, n_jobs=-1)

    # Build weight table
    weights = pd.DataFrame({
        "feature": feature_cols_clean,
        "importance": perm_imp.importances_mean,
    }).sort_values("importance", ascending=False)

    # Categorize
    weights["category"] = weights["feature"].apply(
        lambda f: "context" if f in CONTEXT_FEATURES else "performance"
    )

    # Direction: higher or lower = better for PLAYS?
    plays = df_model[y_binary == 1]
    rest = df_model[y_binary == 0]
    directions = {}
    for col in feature_cols_clean:
        diff = plays[col].mean() - rest[col].mean()
        directions[col] = "higher is better" if diff > 0 else "lower is better"
    weights["direction"] = weights["feature"].map(directions)

    # Statistical significance
    if stat_df is not None and len(stat_df) > 0:
        stat_lookup = stat_df.set_index("feature")["p_value"].to_dict()
        effect_lookup = stat_df.set_index("feature")["effect_size_d"].to_dict()
        weights["p_value"] = weights["feature"].apply(
            lambda f: stat_lookup.get(f.replace("mean_", ""), np.nan)
        )
        weights["effect_size"] = weights["feature"].apply(
            lambda f: effect_lookup.get(f.replace("mean_", ""), np.nan)
        )
    else:
        weights["p_value"] = np.nan
        weights["effect_size"] = np.nan

    # Reliability
    if rel_df is not None and len(rel_df) > 0:
        rel_lookup = rel_df.set_index("feature")["coeff_variation"].to_dict()
        weights["reliability_cv"] = weights["feature"].apply(
            lambda f: rel_lookup.get(f.replace("mean_", ""), np.nan)
        )
    else:
        weights["reliability_cv"] = np.nan

    # Clean names
    weights["feature_name"] = weights["feature"].str.replace("mean_", "", regex=False)

    # Normalize performance weights to sum to 1 (only performance KPIs)
    perf_weights = weights[weights["category"] == "performance"].copy()
    perf_weights["weight"] = perf_weights["importance"].clip(lower=0)
    pw_total = perf_weights["weight"].sum()
    if pw_total > 0:
        perf_weights["weight"] = perf_weights["weight"] / pw_total

    # Also normalize all weights for reference
    weights["weight_all"] = weights["importance"].clip(lower=0)
    wa_total = weights["weight_all"].sum()
    if wa_total > 0:
        weights["weight_all"] = weights["weight_all"] / wa_total

    # Merge performance weights back
    weights = weights.merge(
        perf_weights[["feature", "weight"]], on="feature", how="left"
    )
    weights["weight"] = weights["weight"].fillna(0)

    print("\n  PERFORMANCE KPIs (what the goalkeeper does on the pitch):")
    perf_display = weights[weights["category"] == "performance"].sort_values("weight", ascending=False)
    print(perf_display[["feature_name", "weight", "direction", "p_value", "effect_size"]].head(15).to_string(index=False))

    print("\n  CONTEXT FEATURES (not goalkeeper skills — used in model, not in composite):")
    ctx_display = weights[weights["category"] == "context"]
    print(ctx_display[["feature_name", "weight_all", "direction"]].to_string(index=False))

    weights.to_csv(OUTPUT / "kpi_weights.csv", index=False)
    print("\nSaved: kpi_weights.csv")

    # ── Step 2: Signal A — Model probability ──────────────────────────
    print("\n--- Signal A: XGBoost P(PLAYS) — uses ALL features ---")

    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        scale_pos_weight=len(y_binary[y_binary == 0]) / max(len(y_binary[y_binary == 1]), 1),
        random_state=42, eval_metric="logloss", verbosity=0,
    )
    xgb_model.fit(X, y_binary)
    model_proba = xgb_model.predict_proba(X)[:, 1]
    model_score = pd.Series(model_proba, index=X.index).rank(pct=True) * 100

    print(f"  Model probability range: {model_proba.min():.3f} – {model_proba.max():.3f}")

    # ── Step 3: Signal B — Performance-only composite ─────────────────
    print("\n--- Signal B: Performance Composite — ONLY goalkeeper KPIs ---")

    X_pctile = X[perf_cols].rank(pct=True)

    # Invert "lower is better" features
    for _, row in perf_weights.iterrows():
        col = row["feature"]
        if row["direction"] == "lower is better" and col in X_pctile.columns:
            X_pctile[col] = 1 - X_pctile[col]

    # Weighted sum using performance-only weights
    perf_weight_dict = perf_weights.set_index("feature")["weight"].to_dict()
    composite_raw = pd.Series(0.0, index=X.index)
    for col in perf_cols:
        w = perf_weight_dict.get(col, 0)
        if w > 0:
            composite_raw += X_pctile[col] * w

    composite_score = composite_raw.rank(pct=True) * 100

    print(f"  Composite score range: {composite_score.min():.1f} – {composite_score.max():.1f}")

    # ── Step 4: Ensemble ──────────────────────────────────────────────
    print("\n--- Ensemble Score (60% model + 40% performance composite) ---")
    final_score = (0.6 * model_score + 0.4 * composite_score).clip(1, 100)

    # ── Step 5: Top-3 contributing performance features per keeper ────
    top3_features = []
    top_perf = perf_weights[perf_weights["weight"] > 0].head(10)
    for idx in X.index:
        contributions = []
        for _, wrow in top_perf.iterrows():
            col = wrow["feature"]
            fname = wrow["feature_name"]
            pctile_val = X_pctile.loc[idx, col]
            contribution = pctile_val * wrow["weight"]
            contributions.append((fname, contribution, pctile_val))
        contributions.sort(key=lambda x: x[1], reverse=True)
        top3 = contributions[:3]
        top3_features.append(
            " | ".join(f"{name} ({pct:.0%})" for name, _, pct in top3)
        )

    # ── Step 6: Build output DataFrame ────────────────────────────────
    scores_df = pd.DataFrame({
        "playerId": df_model["playerId"].values,
        "name": df_model["name"].values,
        "age": df_model["age"].values,
        "status": df_model["status"].values,
        "origin_team": df_model["origin_team"].values if "origin_team" in df_model.columns else "",
        "origin_comp": df_model["origin_comp"].values if "origin_comp" in df_model.columns else "",
        "origin_median": df_model["origin_median"].values,
        "n_matches": df_model["n_matches_loaded"].values,
        "scouting_score": final_score.values,
        "model_score": model_score.values,
        "performance_score": composite_score.values,
        "model_probability": model_proba,
        "top3_strengths": top3_features,
    }, index=X.index)

    scores_df = scores_df.sort_values("scouting_score", ascending=False)

    # Validation
    print("\nScore distribution by status:")
    for status in STATUS_ORDER:
        subset = scores_df[scores_df["status"] == status]
        if len(subset) > 0:
            print(f"  {status:10s}  median={subset['scouting_score'].median():.1f}  "
                  f"mean={subset['scouting_score'].mean():.1f}  "
                  f"n={len(subset)}")

    scores_df.to_csv(OUTPUT / "scouting_scores.csv", index=False)
    print(f"\nSaved: scouting_scores.csv ({len(scores_df)} keepers)")

    # ── Step 7: OHL Transfer Targets ──────────────────────────────────
    print(f"\n--- OHL Scouting Targets (origin_median ≤ {OHL_LEAGUE_CEILING}) ---")

    targets = scores_df[
        (scores_df["origin_median"] <= OHL_LEAGUE_CEILING)
        & (scores_df["status"] != "DROPPED")
    ].copy()

    print(f"  {len(targets)} keepers from leagues at or below Belgian Pro League level")
    print(f"\n  Top 30 transfer targets:")
    display_cols = ["name", "age", "origin_comp", "origin_median", "scouting_score",
                    "model_score", "performance_score", "status", "n_matches", "top3_strengths"]
    available_cols = [c for c in display_cols if c in targets.columns]
    print(targets.head(30)[available_cols].to_string(index=False))

    targets.to_csv(OUTPUT / "scouting_targets_ohl.csv", index=False)
    print(f"\nSaved: scouting_targets_ohl.csv ({len(targets)} keepers)")

    # ── Step 8: Visualisations ────────────────────────────────────────
    _plot_kpi_weights(weights)
    _plot_score_distribution(scores_df)
    _plot_top_targets(targets)
    _plot_score_validation(scores_df)

    return scores_df


def _plot_kpi_weights(weights):
    """Bar chart of KPI weights — performance vs context, with significance."""
    perf = weights[weights["category"] == "performance"].sort_values("weight", ascending=False).head(15)
    perf = perf.iloc[::-1]

    fig, ax = plt.subplots(figsize=(14, 9))
    colors = []
    for _, row in perf.iterrows():
        p = row["p_value"]
        if p < 0.01:
            colors.append("#1a9641")  # dark green — highly significant
        elif p < 0.05:
            colors.append("#a6d96a")  # light green — significant
        else:
            colors.append("#bdbdbd")  # grey — not significant
    bars = ax.barh(range(len(perf)), perf["weight"], color=colors)

    ax.set_yticks(range(len(perf)))
    labels = []
    for _, row in perf.iterrows():
        arrow = "+" if row["direction"] == "higher is better" else "-"
        labels.append(f"({arrow}) {row['feature_name']}")
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Weight in performance composite (normalized)")
    ax.set_title("Which GK Performance KPIs Predict Career Progression?\n"
                 "(+) = higher is better | (-) = lower is better\n"
                 "Dark green = p<0.01 | Light green = p<0.05 | Grey = not significant",
                 fontsize=12)

    plt.tight_layout()
    plt.savefig(OUTPUT / "12_kpi_weights.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 12_kpi_weights.png")


def _plot_score_distribution(scores_df):
    """Histogram of scouting scores coloured by status."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.hist(scores_df["scouting_score"], bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Scouting Score")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of GK Scouting Scores")
    ax.axvline(scores_df["scouting_score"].median(), color="red", linestyle="--",
               label=f"Median: {scores_df['scouting_score'].median():.1f}")
    ax.legend()

    ax = axes[1]
    for status in STATUS_ORDER:
        subset = scores_df[scores_df["status"] == status]
        if len(subset) > 0:
            ax.hist(subset["scouting_score"], bins=20, alpha=0.5, label=status)
    ax.set_xlabel("Scouting Score")
    ax.set_ylabel("Count")
    ax.set_title("Scouting Score by Career Status")
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT / "13_score_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 13_score_distribution.png")


def _plot_top_targets(targets):
    """Bar chart of top 30 scouting targets."""
    if len(targets) == 0:
        return

    top30 = targets.head(30).iloc[::-1]

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = {"PLAYS": "#2ecc71", "BENCH": "#f39c12", "STAYED": "#3498db", "DROPPED": "#e74c3c"}
    bar_colors = [colors.get(s, "#95a5a6") for s in top30["status"]]

    ax.barh(range(len(top30)), top30["scouting_score"], color=bar_colors)
    labels = [f"{row['name']} ({row['age']:.0f}, {row.get('origin_comp', '?')})"
              for _, row in top30.iterrows()]
    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Scouting Score (1–100)")
    ax.set_title("Top 30 GK Transfer Targets for OHL\n"
                 "(from leagues at or below Belgian Pro League level)")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=s) for s, c in colors.items()]
    ax.legend(handles=legend_elements, loc="lower right", title="Actual Status")

    plt.tight_layout()
    plt.savefig(OUTPUT / "14_top_scouting_targets.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 14_top_scouting_targets.png")


def _plot_score_validation(scores_df):
    """Box plot: scouting score by actual career status."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    sns.boxplot(data=scores_df, x="status", y="scouting_score", order=STATUS_ORDER, ax=ax)
    ax.set_title("Final Scouting Score")
    ax.set_ylabel("Score (1–100)")
    ax.set_xlabel("")

    ax = axes[1]
    sns.boxplot(data=scores_df, x="status", y="model_score", order=STATUS_ORDER, ax=ax)
    ax.set_title("Model Score (all features)")
    ax.set_ylabel("Score (0–100)")
    ax.set_xlabel("")

    ax = axes[2]
    sns.boxplot(data=scores_df, x="status", y="performance_score", order=STATUS_ORDER, ax=ax)
    ax.set_title("Performance Score (GK KPIs only)")
    ax.set_ylabel("Score (0–100)")
    ax.set_xlabel("")

    plt.suptitle("Score Validation: Do PLAYS Keepers Score Highest?", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT / "15_score_validation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 15_score_validation.png")
