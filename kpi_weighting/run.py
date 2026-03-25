#!/usr/bin/env python3
"""Phase 1 — KPI Discovery & Weighting Module

Standalone module that:
1. Ingests goalkeeper data and identifies all available KPIs
2. Uses data-driven methods to determine which KPIs predict progression
3. Assigns a numerical weight to every KPI via consensus of 6 methods
4. Outputs ranked weight tables and visualizations

Run: python -m kpi_weighting.run
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from shared.data_utils import (
    load_definitions, load_and_aggregate_data, select_features,
    get_cache_path, STATUS_ORDER, META_COLS, PROJECT_ROOT
)

OUTPUT = Path(__file__).resolve().parent / "output"
OUTPUT.mkdir(exist_ok=True)


def discover_kpis(df, score_defs, feature_cols, gk_cols, general_cols):
    """Identify all available KPIs and compute coverage statistics."""
    print("\n" + "=" * 70)
    print("KPI DISCOVERY")
    print("=" * 70)

    score_feature_cols = [c for c in feature_cols if c.startswith("mean_")]
    discovery = []
    for col in score_feature_cols:
        coverage = 1 - df[col].isnull().mean() if col in df.columns else 0
        category = "GK-specific" if col in gk_cols else (
            "General" if col in general_cols else "Meta"
        )
        discovery.append({
            "feature": col,
            "feature_name": col.replace("mean_", ""),
            "category": category,
            "coverage_pct": coverage,
            "mean": df[col].mean() if col in df.columns else np.nan,
            "std": df[col].std() if col in df.columns else np.nan,
            "n_non_null": df[col].notna().sum() if col in df.columns else 0,
        })

    disc_df = pd.DataFrame(discovery).sort_values("coverage_pct", ascending=False)
    print(f"\nTotal score features: {len(disc_df)}")
    print(f"  GK-specific: {len([c for c in score_feature_cols if c in gk_cols])}")
    print(f"  General: {len([c for c in score_feature_cols if c in general_cols])}")
    print(f"\nCoverage distribution:")
    print(f"  100% coverage: {(disc_df['coverage_pct'] == 1.0).sum()}")
    print(f"  >90% coverage: {(disc_df['coverage_pct'] > 0.9).sum()}")
    print(f"  >50% coverage: {(disc_df['coverage_pct'] > 0.5).sum()}")

    disc_df.to_csv(OUTPUT / "kpi_discovery.csv", index=False)
    return disc_df


def compute_weights(df_model, feature_cols):
    """Compute KPI weights using 6 data-driven methods.

    Methods:
    1. Random Forest feature importance
    2. XGBoost feature importance
    3. Logistic Regression (L1/Lasso) — absolute coefficients
    4. Mutual information scores
    5. Permutation importance (Random Forest)
    6. Statistical effect sizes (Cohen's d from Mann-Whitney U)

    Returns DataFrame with per-method scores and consensus weight.
    """
    print("\n" + "=" * 70)
    print("KPI WEIGHTING — 6 METHODS")
    print("=" * 70)

    X = df_model[feature_cols].copy()
    X = X.fillna(X.median())
    y = (df_model["status"] == "PLAYS").astype(int)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols, index=X.index)

    results = pd.DataFrame({"feature": feature_cols})

    # --- Method 1: Random Forest importance ---
    print("\n  [1/6] Random Forest feature importance...")
    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42,
        max_depth=6, min_samples_leaf=5
    )
    rf.fit(X, y)
    results["rf_importance"] = rf.feature_importances_

    # --- Method 2: XGBoost importance ---
    print("  [2/6] XGBoost feature importance...")
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        scale_pos_weight=n_neg / max(n_pos, 1),
        random_state=42, eval_metric="logloss", verbosity=0,
    )
    xgb_model.fit(X, y)
    results["xgb_importance"] = xgb_model.feature_importances_

    # --- Method 3: Logistic Regression (L1) ---
    print("  [3/6] Logistic Regression (Lasso) coefficients...")
    lr = LogisticRegression(
        max_iter=2000, penalty="l1", solver="saga",
        class_weight="balanced", random_state=42, C=0.1
    )
    lr.fit(X_scaled, y)
    results["lr_abs_coef"] = np.abs(lr.coef_[0])
    results["lr_coef"] = lr.coef_[0]

    # --- Method 4: Mutual information ---
    print("  [4/6] Mutual information scores...")
    mi_scores = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
    results["mutual_info"] = mi_scores

    # --- Method 5: Permutation importance ---
    print("  [5/6] Permutation importance (RF)...")
    perm = permutation_importance(rf, X, y, n_repeats=20, random_state=42, n_jobs=-1)
    results["perm_importance"] = perm.importances_mean

    # --- Method 6: Statistical effect size ---
    print("  [6/6] Mann-Whitney U effect sizes...")
    plays_mask = y == 1
    effect_sizes = []
    p_values = []
    for col in feature_cols:
        p_vals = X.loc[plays_mask, col].dropna()
        r_vals = X.loc[~plays_mask, col].dropna()
        if len(p_vals) < 5 or len(r_vals) < 5:
            effect_sizes.append(0)
            p_values.append(1.0)
            continue
        _, p_value = stats.mannwhitneyu(p_vals, r_vals, alternative="two-sided")
        pooled_std = pd.concat([p_vals, r_vals]).std()
        d = abs(p_vals.mean() - r_vals.mean()) / max(pooled_std, 1e-10)
        effect_sizes.append(d)
        p_values.append(p_value)
    results["effect_size"] = effect_sizes
    results["p_value"] = p_values

    # --- Consensus weighting ---
    print("\n  Computing consensus weights...")
    method_cols = ["rf_importance", "xgb_importance", "lr_abs_coef",
                   "mutual_info", "perm_importance", "effect_size"]
    for col in method_cols:
        col_min = results[col].min()
        col_max = results[col].max()
        rng = col_max - col_min
        if rng > 0:
            results[f"{col}_norm"] = (results[col] - col_min) / rng
        else:
            results[f"{col}_norm"] = 0

    norm_cols = [f"{c}_norm" for c in method_cols]
    results["consensus_weight"] = results[norm_cols].mean(axis=1)

    # Normalize to sum to 1
    total = results["consensus_weight"].sum()
    if total > 0:
        results["consensus_weight"] = results["consensus_weight"] / total

    # Add metadata
    results["rank"] = results["consensus_weight"].rank(ascending=False).astype(int)
    results["feature_name"] = results["feature"].str.replace("mean_", "", regex=False)

    # Direction
    plays_df = df_model[plays_mask]
    rest_df = df_model[~plays_mask]
    results["direction"] = results["feature"].apply(
        lambda f: "higher" if plays_df[f].mean() > rest_df[f].mean() else "lower"
    )

    # Category
    context_set = set(META_COLS)
    results["category"] = results["feature"].apply(
        lambda f: "context" if f in context_set else "performance"
    )

    results = results.sort_values("consensus_weight", ascending=False)

    print(f"\n  Top 15 KPIs by consensus weight:")
    display_cols = ["rank", "feature_name", "consensus_weight", "category",
                    "direction", "p_value", "effect_size"]
    print(results[display_cols].head(15).to_string(index=False))

    return results


def validate_model(df_model, feature_cols):
    """5-fold stratified CV to assess model quality with weighted features."""
    print("\n" + "=" * 70)
    print("MODEL VALIDATION (5-fold stratified CV)")
    print("=" * 70)

    X = df_model[feature_cols].copy().fillna(df_model[feature_cols].median())
    y = (df_model["status"] == "PLAYS").astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    n_pos = y.sum()
    n_neg = len(y) - n_pos

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42,
            max_depth=6, min_samples_leaf=5
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            scale_pos_weight=n_neg / max(n_pos, 1),
            random_state=42, eval_metric="logloss", verbosity=0,
        ),
    }

    for name, model in models.items():
        auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        f1_scores = cross_val_score(model, X, y, cv=cv, scoring="f1")
        print(f"\n  {name}:")
        print(f"    AUC-ROC: {auc_scores.mean():.3f} (+/- {auc_scores.std():.3f})")
        print(f"    F1:      {f1_scores.mean():.3f} (+/- {f1_scores.std():.3f})")


def plot_weights(weights_df):
    """Generate visualizations of KPI weights."""
    print("\n  Generating visualizations...")

    # --- Plot 1: Consensus weight bar chart (top 20) ---
    perf = weights_df[weights_df["category"] == "performance"].head(20).copy()
    perf = perf.iloc[::-1]  # reverse for horizontal bar

    fig, ax = plt.subplots(figsize=(14, 10))
    colors = []
    for _, row in perf.iterrows():
        p = row["p_value"]
        if p < 0.01:
            colors.append("#1a9641")
        elif p < 0.05:
            colors.append("#a6d96a")
        else:
            colors.append("#bdbdbd")

    ax.barh(range(len(perf)), perf["consensus_weight"], color=colors)
    ax.set_yticks(range(len(perf)))
    labels = []
    for _, row in perf.iterrows():
        arrow = "+" if row["direction"] == "higher" else "-"
        labels.append(f"({arrow}) {row['feature_name']}")
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Consensus Weight (normalized)")
    ax.set_title("KPI Weights for Goalkeeper Progression Prediction\n"
                 "(+) = higher is better | (-) = lower is better\n"
                 "Dark green = p<0.01 | Light green = p<0.05 | Grey = not significant",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT / "kpi_consensus_weights.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Plot 2: Method comparison heatmap (top 15) ---
    top15 = weights_df.head(15).copy()
    method_cols = ["rf_importance_norm", "xgb_importance_norm", "lr_abs_coef_norm",
                   "mutual_info_norm", "perm_importance_norm", "effect_size_norm"]
    available_methods = [c for c in method_cols if c in top15.columns]

    heatmap_data = top15.set_index("feature_name")[available_methods]
    heatmap_data.columns = [c.replace("_norm", "").replace("_", " ").title() for c in available_methods]

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlOrRd",
                ax=ax, linewidths=0.5)
    ax.set_title("Top 15 KPIs — Normalized Importance Across 6 Methods", fontsize=13)
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(OUTPUT / "kpi_method_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Plot 3: All KPIs weight distribution ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(weights_df["consensus_weight"], bins=30, color="steelblue",
            edgecolor="white", alpha=0.8)
    ax.axvline(weights_df["consensus_weight"].median(), color="red",
               linestyle="--", label=f"Median: {weights_df['consensus_weight'].median():.4f}")
    ax.set_xlabel("Consensus Weight")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of KPI Weights")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT / "kpi_weight_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("  Saved: kpi_consensus_weights.png, kpi_method_comparison.png, kpi_weight_distribution.png")


def main():
    print("=" * 70)
    print("PHASE 1 — KPI DISCOVERY & WEIGHTING MODULE")
    print("=" * 70)

    # Load definitions and data
    score_defs, score_labels, kpi_defs = load_definitions()
    cache = get_cache_path()
    dataset, df = load_and_aggregate_data(score_defs, cache_path=cache)

    # Feature selection
    df_model, feature_cols, gk_cols, general_cols = select_features(df, score_defs)

    # KPI Discovery
    disc_df = discover_kpis(df_model, score_defs, feature_cols, gk_cols, general_cols)

    # KPI Weighting
    weights_df = compute_weights(df_model, feature_cols)

    # Save full weight table
    output_cols = ["rank", "feature", "feature_name", "consensus_weight", "category",
                   "direction", "p_value", "effect_size",
                   "rf_importance", "xgb_importance", "lr_abs_coef",
                   "mutual_info", "perm_importance"]
    available = [c for c in output_cols if c in weights_df.columns]
    weights_df[available].to_csv(OUTPUT / "kpi_weights_full.csv", index=False)
    print(f"\nSaved: kpi_weights_full.csv ({len(weights_df)} KPIs)")

    # Validation
    validate_model(df_model, feature_cols)

    # Visualizations
    plot_weights(weights_df)

    print(f"\n{'=' * 70}")
    print(f"All Phase 1 outputs saved to: {OUTPUT.resolve()}")
    print(f"{'=' * 70}")

    return weights_df


if __name__ == "__main__":
    main()
