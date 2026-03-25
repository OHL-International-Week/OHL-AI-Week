#!/usr/bin/env python3
"""Step 2 — Assign data-driven weights to every KPI.

Uses the full KPI dataset (built by build_kpi_dataset.py) to determine
how important each of the ~1,000+ KPIs is for predicting goalkeeper
progression from lower leagues to top leagues.

Methods:
1. XGBoost feature importance (gain)
2. Random Forest feature importance (Gini)
3. Permutation importance
4. Mutual information
5. Mann-Whitney U effect size (Cohen's d)

Consensus weight = normalized average across all methods.
Every KPI gets a weight — no KPI is left unweighted.

Output:
  - kpi_weights_all.csv       Full weight table for every KPI
  - kpi_weights_top100.csv    Top 100 KPIs for quick reference
  - weight_bar_chart.png      Top 40 KPIs visualization
  - weight_heatmap.png        Method agreement heatmap (top 30)
  - weight_distribution.png   Distribution of all weights

Run: python3 KPI_Weights/assign_weights.py
"""

import json
import sys
from pathlib import Path

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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

PROJECT = Path(__file__).resolve().parent.parent
GK_DATA = PROJECT / "GK_Data"
OUTPUT = Path(__file__).resolve().parent / "output"
CACHE = OUTPUT / "keeper_all_kpis.parquet"


def load_kpi_metadata():
    """Load KPI definitions for annotation."""
    with open(GK_DATA / "player_kpi_definitions.json") as f:
        raw = json.load(f).get("data", [])
    meta = {}
    for k in raw:
        details = k.get("details") or {}
        parent = k.get("parentKpi") or {}
        context = k.get("context") or {}
        meta[k["name"]] = {
            "kpi_id": k["id"],
            "label": details.get("label") or k["name"],
            "definition": (details.get("definition") or "")[:300],
            "meaning": (details.get("meaning") or "")[:200],
            "parent": parent.get("name", ""),
            "context": context.get("name", ""),
            "context_label": context.get("label", ""),
            "inverted": k.get("inverted", False),
        }
    return meta


def prepare_features(df):
    """Prepare feature matrix: drop high-missing, fill NaN, return X, y, names."""
    mean_cols = sorted([c for c in df.columns if c.startswith("mean_")])
    print(f"  Raw KPI features: {len(mean_cols)}")

    # Drop features with >60% missing
    missing = df[mean_cols].isnull().mean()
    keep = missing[missing <= 0.6].index.tolist()
    dropped = len(mean_cols) - len(keep)
    print(f"  Dropped (>60% missing): {dropped}")
    print(f"  Retained features: {len(keep)}")

    # Add meta features
    meta = ["age", "origin_median", "n_matches_loaded"]
    feature_cols = keep + [m for m in meta if m in df.columns]

    # Fill NaN with median
    X = df[feature_cols].copy()
    for col in feature_cols:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())

    y = (df["status"] == "PLAYS").astype(int)

    return X, y, feature_cols


def compute_all_weights(X, y, feature_cols):
    """Run 5 weighting methods and produce consensus."""
    n_features = len(feature_cols)
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    print(f"\n  Target: PLAYS ({n_pos}) vs REST ({n_neg})")
    print(f"  Features: {n_features}")

    results = pd.DataFrame({"feature": feature_cols})

    # --- 1. XGBoost ---
    print("\n  [1/5] XGBoost importance...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        scale_pos_weight=n_neg / max(n_pos, 1),
        random_state=42, eval_metric="logloss", verbosity=0,
        colsample_bytree=0.7, subsample=0.8,
    )
    xgb_model.fit(X, y)
    results["xgb_importance"] = xgb_model.feature_importances_

    # --- 2. Random Forest ---
    print("  [2/5] Random Forest importance...")
    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42,
        max_depth=8, min_samples_leaf=5, max_features="sqrt",
    )
    rf.fit(X, y)
    results["rf_importance"] = rf.feature_importances_

    # --- 3. Permutation importance ---
    print("  [3/5] Permutation importance (RF)...")
    perm = permutation_importance(rf, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    results["perm_importance"] = perm.importances_mean.clip(min=0)

    # --- 4. Mutual information ---
    print("  [4/5] Mutual information...")
    mi = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
    results["mutual_info"] = mi

    # --- 5. Effect size ---
    print("  [5/5] Mann-Whitney U effect sizes...")
    plays_mask = y == 1
    effect_sizes = []
    p_values = []
    directions = []
    for col in feature_cols:
        p_vals = X.loc[plays_mask, col].dropna()
        r_vals = X.loc[~plays_mask, col].dropna()
        if len(p_vals) < 5 or len(r_vals) < 5:
            effect_sizes.append(0)
            p_values.append(1.0)
            directions.append("neutral")
            continue
        try:
            _, pval = stats.mannwhitneyu(p_vals, r_vals, alternative="two-sided")
        except Exception:
            pval = 1.0
        pooled_std = pd.concat([p_vals, r_vals]).std()
        d = abs(p_vals.mean() - r_vals.mean()) / max(pooled_std, 1e-10)
        effect_sizes.append(d)
        p_values.append(pval)
        directions.append("higher" if p_vals.mean() > r_vals.mean() else "lower")

    results["effect_size"] = effect_sizes
    results["p_value"] = p_values
    results["direction"] = directions

    # --- Consensus ---
    print("\n  Computing consensus weights...")
    method_cols = ["xgb_importance", "rf_importance", "perm_importance",
                   "mutual_info", "effect_size"]

    for col in method_cols:
        mn = results[col].min()
        mx = results[col].max()
        rng = mx - mn
        if rng > 0:
            results[f"{col}_norm"] = (results[col] - mn) / rng
        else:
            results[f"{col}_norm"] = 0.0

    norm_cols = [f"{c}_norm" for c in method_cols]
    results["consensus_weight"] = results[norm_cols].mean(axis=1)

    # Normalize to sum to 1
    total = results["consensus_weight"].sum()
    if total > 0:
        results["consensus_weight"] = results["consensus_weight"] / total

    results["rank"] = results["consensus_weight"].rank(ascending=False).astype(int)

    return results


def annotate_results(results, kpi_meta):
    """Add human-readable labels and metadata to results."""
    results["kpi_name"] = results["feature"].str.replace("mean_", "", regex=False)

    labels = []
    definitions = []
    parents = []
    contexts = []
    categories = []

    meta_features = {"age", "origin_median", "n_matches_loaded"}

    for _, row in results.iterrows():
        name = row["kpi_name"]
        if name in meta_features:
            labels.append(name)
            definitions.append("Context feature (not a KPI)")
            parents.append("")
            contexts.append("")
            categories.append("context")
            continue

        m = kpi_meta.get(name, {})
        labels.append(m.get("label", name))
        definitions.append(m.get("definition", ""))
        parents.append(m.get("parent", ""))
        contexts.append(m.get("context_label", ""))

        # Categorize
        if name.startswith("GK_") or "GOALKEEPER" in name.upper():
            categories.append("GK-specific")
        elif m.get("parent"):
            categories.append(f"Contextual ({m.get('parent', '')[:30]})")
        else:
            categories.append("Base KPI")

    results["label"] = labels
    results["definition"] = definitions
    results["parent_kpi"] = parents
    results["context"] = contexts
    results["category"] = categories

    return results


def validate_model(X, y, top_n=50):
    """Quick validation: how well do top-N KPIs predict PLAYS?"""
    print(f"\n  Validating model with top {top_n} features...")
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        scale_pos_weight=n_neg / max(n_pos, 1),
        random_state=42, eval_metric="logloss", verbosity=0,
    )
    auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    f1 = cross_val_score(model, X, y, cv=cv, scoring="f1")
    print(f"    AUC-ROC: {auc.mean():.3f} (+/- {auc.std():.3f})")
    print(f"    F1:      {f1.mean():.3f} (+/- {f1.std():.3f})")
    return auc.mean(), f1.mean()


def plot_top_weights(results, n=40):
    """Bar chart of top N KPIs by consensus weight."""
    top = results.head(n).copy().iloc[::-1]

    fig, ax = plt.subplots(figsize=(14, max(10, n * 0.3)))
    colors = []
    for _, row in top.iterrows():
        p = row.get("p_value", 1.0)
        if p < 0.001:
            colors.append("#1a9641")
        elif p < 0.01:
            colors.append("#66bd63")
        elif p < 0.05:
            colors.append("#a6d96a")
        elif p < 0.1:
            colors.append("#fee08b")
        else:
            colors.append("#bdbdbd")

    ax.barh(range(len(top)), top["consensus_weight"], color=colors)
    ax.set_yticks(range(len(top)))

    labels = []
    for _, row in top.iterrows():
        arrow = "+" if row.get("direction") == "higher" else ("-" if row.get("direction") == "lower" else " ")
        name = row["kpi_name"]
        # Shorten long names
        if len(name) > 45:
            name = name[:42] + "..."
        labels.append(f"({arrow}) {name}")

    ax.set_yticklabels(labels, fontsize=7.5, fontfamily="monospace")
    ax.set_xlabel("Consensus Weight (normalized across 5 methods)")
    ax.set_title(
        f"Top {n} KPIs for Goalkeeper Progression Prediction\n"
        "(+) PLAYS higher  (-) PLAYS lower\n"
        "Green = p<0.01  Light green = p<0.05  Yellow = p<0.10  Grey = not significant",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT / "weight_bar_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: weight_bar_chart.png")


def plot_method_heatmap(results, n=30):
    """Heatmap showing agreement across methods for top N KPIs."""
    top = results.head(n).copy()
    method_norm = ["xgb_importance_norm", "rf_importance_norm",
                   "perm_importance_norm", "mutual_info_norm", "effect_size_norm"]
    available = [c for c in method_norm if c in top.columns]

    heatmap_data = top.set_index("kpi_name")[available].copy()
    heatmap_data.columns = [c.replace("_norm", "").replace("_", " ").title() for c in available]

    fig, ax = plt.subplots(figsize=(10, max(8, n * 0.35)))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlOrRd",
                ax=ax, linewidths=0.5, vmin=0, vmax=1)
    ax.set_title(f"Top {n} KPIs — Method Agreement\n(1.0 = highest within method, 0.0 = lowest)",
                 fontsize=12)
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(OUTPUT / "weight_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: weight_heatmap.png")


def plot_weight_distribution(results):
    """Distribution of consensus weights."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.hist(results["consensus_weight"], bins=50, color="steelblue",
            edgecolor="white", alpha=0.8)
    ax.set_xlabel("Consensus Weight")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of KPI Weights (all KPIs)")
    ax.axvline(results["consensus_weight"].quantile(0.9), color="red",
               linestyle="--", label=f"90th pctile: {results['consensus_weight'].quantile(0.9):.5f}")
    ax.legend()

    # Cumulative weight
    ax = axes[1]
    sorted_w = results.sort_values("consensus_weight", ascending=False)["consensus_weight"]
    cumsum = sorted_w.cumsum()
    ax.plot(range(1, len(cumsum) + 1), cumsum.values, linewidth=2, color="steelblue")
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="50% of total weight")
    ax.axhline(0.8, color="orange", linestyle="--", alpha=0.5, label="80% of total weight")
    # Find how many features capture 50% and 80%
    n_50 = (cumsum <= 0.5).sum() + 1
    n_80 = (cumsum <= 0.8).sum() + 1
    ax.axvline(n_50, color="red", linestyle=":", alpha=0.3)
    ax.axvline(n_80, color="orange", linestyle=":", alpha=0.3)
    ax.set_xlabel("Number of KPIs (ranked by weight)")
    ax.set_ylabel("Cumulative Weight")
    ax.set_title(f"Cumulative Weight — Top {n_50} KPIs = 50%, Top {n_80} = 80%")
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT / "weight_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: weight_distribution.png")


def plot_category_breakdown(results):
    """Weight distribution by KPI category."""
    # Group KPIs into broad categories
    def broad_category(row):
        name = row["kpi_name"]
        cat = row.get("category", "")
        if cat == "context":
            return "Context"
        if "GK" in name or "PREVENTED" in name or "CAUGHT" in name or "LAUNCH" in name:
            return "GK-Specific"
        if "PASS" in name or "DIAGONAL" in name or "LOW_PASS" in name or "GOAL_KICK" in name:
            return "Distribution"
        if "BYPASSED" in name or "PACKING" in name or "PXT" in name:
            return "Packing / xT"
        if "BALL_LOSS" in name or "BALL_WIN" in name or "DUEL" in name:
            return "Ball Wins/Losses"
        if "TOUCH" in name or "OFFENSIVE" in name or "DEFENSIVE" in name:
            return "Involvement"
        if "SHOT" in name or "GOAL" in name or "XG" in name:
            return "Shooting / Goals"
        return "Other"

    results["broad_category"] = results.apply(broad_category, axis=1)

    cat_weights = results.groupby("broad_category")["consensus_weight"].agg(["sum", "count"])
    cat_weights = cat_weights.sort_values("sum", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    ax.barh(range(len(cat_weights)), cat_weights["sum"], color="steelblue")
    ax.set_yticks(range(len(cat_weights)))
    ax.set_yticklabels([f"{idx} ({row['count']:.0f} KPIs)" for idx, row in cat_weights.iterrows()], fontsize=10)
    ax.set_xlabel("Total Weight (sum of consensus weights in category)")
    ax.set_title("Weight Distribution by KPI Category")

    ax = axes[1]
    ax.barh(range(len(cat_weights)), cat_weights["sum"] / cat_weights["count"], color="darkorange")
    ax.set_yticks(range(len(cat_weights)))
    ax.set_yticklabels(cat_weights.index, fontsize=10)
    ax.set_xlabel("Average Weight per KPI in Category")
    ax.set_title("Average Importance per KPI by Category")

    plt.tight_layout()
    plt.savefig(OUTPUT / "weight_by_category.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: weight_by_category.png")


def main():
    print("=" * 70)
    print("KPI WEIGHTS — Assign weights to ALL goalkeeper KPIs")
    print("=" * 70)

    # Load cached data
    if not CACHE.exists():
        print(f"ERROR: Run build_kpi_dataset.py first to create {CACHE}")
        sys.exit(1)

    df = pd.read_parquet(CACHE)
    print(f"Loaded: {len(df)} keepers")

    # Load metadata
    kpi_meta = load_kpi_metadata()
    print(f"KPI definitions loaded: {len(kpi_meta)}")

    # Prepare features
    X, y, feature_cols = prepare_features(df)

    # Compute weights
    print("\n" + "=" * 70)
    print("COMPUTING WEIGHTS (5 methods)")
    print("=" * 70)
    results = compute_all_weights(X, y, feature_cols)

    # Annotate with metadata
    results = annotate_results(results, kpi_meta)
    results = results.sort_values("rank")

    # Print top 30
    print(f"\n  Top 30 KPIs by consensus weight:")
    display_cols = ["rank", "kpi_name", "consensus_weight", "direction",
                    "p_value", "effect_size", "category"]
    print(results[display_cols].head(30).to_string(index=False))

    # Save full table
    save_cols = ["rank", "kpi_name", "label", "consensus_weight",
                 "category", "direction", "p_value", "effect_size",
                 "xgb_importance", "rf_importance", "perm_importance",
                 "mutual_info", "definition", "parent_kpi", "context"]
    available = [c for c in save_cols if c in results.columns]
    results[available].to_csv(OUTPUT / "kpi_weights_all.csv", index=False)
    print(f"\n  Saved: kpi_weights_all.csv ({len(results)} KPIs)")

    # Top 100
    results[available].head(100).to_csv(OUTPUT / "kpi_weights_top100.csv", index=False)
    print(f"  Saved: kpi_weights_top100.csv")

    # Validation: full model vs top-50
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    print("\n  All features:")
    auc_all, f1_all = validate_model(X, y)

    top50_cols = results.head(50)["feature"].tolist()
    top50_available = [c for c in top50_cols if c in X.columns]
    print(f"\n  Top 50 features only:")
    auc_50, f1_50 = validate_model(X[top50_available], y, top_n=50)

    top20_cols = results.head(20)["feature"].tolist()
    top20_available = [c for c in top20_cols if c in X.columns]
    print(f"\n  Top 20 features only:")
    auc_20, f1_20 = validate_model(X[top20_available], y, top_n=20)

    # Plots
    print("\n" + "=" * 70)
    print("VISUALIZATIONS")
    print("=" * 70)
    plot_top_weights(results, n=40)
    plot_method_heatmap(results, n=30)
    plot_weight_distribution(results)
    plot_category_breakdown(results)

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)

    n_sig_001 = (results["p_value"] < 0.01).sum()
    n_sig_005 = (results["p_value"] < 0.05).sum()
    n_sig_01 = (results["p_value"] < 0.1).sum()
    cumw = results.sort_values("rank")["consensus_weight"].cumsum()
    n_50pct = (cumw <= 0.5).sum() + 1
    n_80pct = (cumw <= 0.8).sum() + 1

    print(f"  Total KPIs weighted: {len(results)}")
    print(f"  Significant (p<0.01): {n_sig_001}")
    print(f"  Significant (p<0.05): {n_sig_005}")
    print(f"  Significant (p<0.10): {n_sig_01}")
    print(f"  Top {n_50pct} KPIs capture 50% of total weight")
    print(f"  Top {n_80pct} KPIs capture 80% of total weight")
    print(f"\n  Model AUC (all features): {auc_all:.3f}")
    print(f"  Model AUC (top 50):       {auc_50:.3f}")
    print(f"  Model AUC (top 20):       {auc_20:.3f}")
    print(f"\n  All outputs in: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
