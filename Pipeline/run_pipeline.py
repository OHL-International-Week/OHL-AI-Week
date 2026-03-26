#!/usr/bin/env python3
"""
GK Scouting Pipeline — Finding the New Number 1
=================================================
OH Leuven International Week, March 2026

Central question:
  "Which measurable on-pitch performances of a goalkeeper in a lower league
   predict whether he will succeed at a higher level?"

Pipeline:
  Step 1 — Identify the most important KPIs from 462 performance features
            using 6 methods (XGBoost, Random Forest, Lasso, Mann-Whitney,
            Boruta, Bootstrap Stability)
  Step 2 — Weight KPIs by consensus across all methods
  Step 3 — Score every goalkeeper 1-100 on predicted progression potential

The number of KPIs used is DATA-DRIVEN: determined by the elbow detection
in kpi_experiments.py and saved in optimal_n_features.txt.

NOTE: Only on-pitch performance KPIs are used. Context variables like
league strength (origin_median), age, and match count are excluded —
they're not performance metrics a scout can act on.

Data: 693 goalkeepers, 40+ leagues, 2019-2026
Run:  python pipeline/run_pipeline.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="colorblind")

OUTPUT = Path(__file__).resolve().parent / "output"
OUTPUT.mkdir(exist_ok=True)

COLORS = {
    "PLAYS": "#2ecc71", "BENCH": "#f39c12",
    "STAYED": "#3498db", "DROPPED": "#e74c3c",
}
STATUS_ORDER = ["PLAYS", "BENCH", "STAYED", "DROPPED"]

n_top_features = 20  # default, overridden by main() from optimal_n_features.txt


def _load_optimal_n():
    """Load the data-driven optimal N from kpi_experiments.py output."""
    optimal_file = OUTPUT / "optimal_n_features.txt"
    if optimal_file.exists():
        n = int(optimal_file.read_text().strip())
        print(f"Loaded optimal N = {n} features (from kpi_experiments.py)")
        return n
    print("WARNING: optimal_n_features.txt not found, using default 20.")
    print("         Run 'python pipeline/kpi_experiments.py' first for data-driven N.")
    return 20


# =========================================================================
#  DATA LOADING
# =========================================================================

def load_data():
    """Load the full KPI dataset built by build_full_kpi_dataset.py."""
    parquet = OUTPUT / "keeper_all_kpis.parquet"
    if not parquet.exists():
        print("Full KPI dataset not found. Building it now...")
        from pipeline.build_full_kpi_dataset import build_dataset
        build_dataset()
    df = pd.read_parquet(parquet)
    print(f"Loaded: {len(df)} keepers, {sum(1 for c in df.columns if c.startswith('mean_'))} KPI features")
    return df


def prepare_features(df, min_coverage=0.5):
    """Pre-filter to performance KPIs only (no context variables)."""
    mean_cols = sorted([c for c in df.columns if c.startswith("mean_")])

    # Coverage filter
    coverage = df[mean_cols].notna().mean()
    keep = coverage[coverage >= min_coverage].index.tolist()

    # Variance filter (remove bottom 5%)
    variances = df[keep].var()
    keep = variances[variances > variances.quantile(0.05)].index.tolist()

    # Redundancy filter (|r| > 0.95)
    X_temp = df[keep].fillna(df[keep].median())
    corr_matrix = X_temp.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        for hc in upper.index[upper[col] > 0.95].tolist():
            if variances.get(hc, 0) < variances.get(col, 0):
                to_drop.add(hc)
            else:
                to_drop.add(col)
    keep = [c for c in keep if c not in to_drop]

    X = df[keep].copy().fillna(df[keep].median())
    y = (df["status"] == "PLAYS").astype(int)

    print(f"After filtering: {len(keep)} performance KPIs (no context variables)")
    print(f"Samples: {len(y)} ({y.sum()} PLAYS, {(~y.astype(bool)).sum()} REST)")

    return X, y, keep


# =========================================================================
#  STEP 1 — Identify Important KPIs (4 methods)
# =========================================================================

def step1_find_important_kpis(X, y, feature_names):
    """Use 4 complementary methods to identify which KPIs predict progression.

    Methods:
      1. XGBoost importance — captures non-linear interactions
      2. Mann-Whitney U + FDR — statistical evidence of real difference
      3. Boruta — "is this KPI better than random noise?"
      4. Bootstrap stability — "is this KPI consistently selected?"
    """
    print("\n" + "=" * 70)
    print("STEP 1 — IDENTIFY IMPORTANT KPIs (4 methods, 462 features)")
    print("=" * 70)

    n_pos, n_neg = y.sum(), len(y) - y.sum()
    results = pd.DataFrame({"feature": feature_names})

    # Method 1: XGBoost
    print("\n  [1/4] XGBoost importance...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        scale_pos_weight=n_neg / max(n_pos, 1), random_state=42,
        eval_metric="logloss", verbosity=0, colsample_bytree=0.5, subsample=0.8)
    xgb_model.fit(X, y)
    results["xgb_importance"] = xgb_model.feature_importances_

    # Method 2: Mann-Whitney U
    print("  [2/4] Mann-Whitney U + Cohen's d...")
    effect_sizes, p_values, directions = [], [], []
    for col in feature_names:
        p_vals = X.loc[y == 1, col].dropna()
        r_vals = X.loc[y == 0, col].dropna()
        if len(p_vals) < 5 or len(r_vals) < 5:
            effect_sizes.append(0); p_values.append(1.0); directions.append("higher"); continue
        _, pv = stats.mannwhitneyu(p_vals, r_vals, alternative="two-sided")
        pooled_std = np.sqrt(((len(p_vals)-1)*p_vals.std()**2 + (len(r_vals)-1)*r_vals.std()**2)
                              / (len(p_vals) + len(r_vals) - 2))
        d = (p_vals.mean() - r_vals.mean()) / max(pooled_std, 1e-10)
        effect_sizes.append(d); p_values.append(pv)
        directions.append("higher" if d > 0 else "lower")

    from statsmodels.stats.multitest import multipletests
    _, p_adj, _, _ = multipletests(p_values, method="fdr_bh")
    results["effect_size"] = effect_sizes
    results["abs_effect_size"] = np.abs(effect_sizes)
    results["p_value"] = p_values
    results["p_fdr"] = p_adj
    results["significant"] = p_adj < 0.05
    results["direction"] = directions

    # Method 3: Boruta
    print("  [3/4] Boruta shadow feature selection (50 iterations)...")
    n_features = len(feature_names)
    hit_counts = np.zeros(n_features)
    for iteration in range(50):
        if (iteration + 1) % 10 == 0:
            print(f"        Iteration {iteration + 1}/50...")
        X_shadow = X.copy()
        for col in X_shadow.columns:
            X_shadow[col] = np.random.permutation(X_shadow[col].values)
        X_shadow.columns = [f"shadow_{c}" for c in X_shadow.columns]
        X_combined = pd.concat([X, X_shadow], axis=1)
        model = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                       random_state=iteration, max_depth=6,
                                       min_samples_leaf=5, n_jobs=-1)
        model.fit(X_combined, y)
        imp = model.feature_importances_
        hit_counts += (imp[:n_features] > imp[n_features:].max()).astype(int)
    results["boruta_fraction"] = hit_counts / 50
    results["boruta_confirmed"] = results["boruta_fraction"] > 0.5

    # Method 4: Bootstrap stability
    print("  [4/4] Bootstrap stability (50 resamples)...")
    selection_counts = np.zeros(n_features)
    for b in range(50):
        if (b + 1) % 10 == 0:
            print(f"        Bootstrap {b + 1}/50...")
        idx = np.random.choice(len(X), size=len(X), replace=True)
        m = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
            scale_pos_weight=n_neg / max(n_pos, 1), random_state=b,
            eval_metric="logloss", verbosity=0, colsample_bytree=0.5, subsample=0.8)
        m.fit(X.iloc[idx], y.iloc[idx])
        top_idx = np.argsort(m.feature_importances_)[::-1][:n_top_features]
        selection_counts[top_idx] += 1
    results["stability_pct"] = selection_counts / 50 * 100

    # Feature name
    results["feature_name"] = results["feature"].str.replace("mean_", "", regex=False)

    n_sig = results["significant"].sum()
    n_boruta = results["boruta_confirmed"].sum()
    print(f"\n  FDR significant: {n_sig} / {len(results)}")
    print(f"  Boruta confirmed: {n_boruta} / {len(results)}")

    return results


# =========================================================================
#  STEP 2 — Weight KPIs by Consensus
# =========================================================================

def step2_weight_kpis(results, df, feature_names):
    """Normalize 4 methods and compute consensus weight."""
    print("\n" + "=" * 70)
    print("STEP 2 — WEIGHT KPIs BY CONSENSUS (4 methods)")
    print("=" * 70)

    # Normalize each method to [0, 1]
    for col in ["xgb_importance", "abs_effect_size", "boruta_fraction", "stability_pct"]:
        mn, mx = results[col].min(), results[col].max()
        results[f"{col}_norm"] = (results[col] - mn) / (mx - mn) if mx > mn else 0.0

    norm_cols = [f"{c}_norm" for c in ["xgb_importance", "abs_effect_size", "boruta_fraction", "stability_pct"]]
    results["consensus_weight"] = results[norm_cols].mean(axis=1)
    total = results["consensus_weight"].sum()
    if total > 0:
        results["consensus_weight"] /= total
    results["rank"] = results["consensus_weight"].rank(ascending=False).astype(int)

    results = results.sort_values("consensus_weight", ascending=False)

    # Save
    save_cols = ["rank", "feature", "feature_name", "consensus_weight", "direction",
                 "p_value", "p_fdr", "significant", "effect_size", "boruta_fraction",
                 "boruta_confirmed", "stability_pct",
                 "xgb_importance"]
    results[save_cols].to_csv(OUTPUT / "kpi_weights.csv", index=False)

    print(f"\n  Top {n_top_features} KPIs:")
    for _, row in results.head(n_top_features).iterrows():
        sig = "***" if row["significant"] else "   "
        boruta = " B" if row["boruta_confirmed"] else "  "
        arrow = "+" if row["direction"] == "higher" else "-"
        print(f"    #{row['rank']:3d}  ({arrow}) {row['feature_name']:<50s}  "
              f"stab={row['stability_pct']:4.0f}%  {sig}{boruta}")

    # Plot
    perf = results.head(n_top_features).copy().iloc[::-1]
    fig, ax = plt.subplots(figsize=(16, 10))
    bar_colors = ["#1a9641" if (row["significant"] and row["boruta_confirmed"])
                  else "#a6d96a" if row["significant"]
                  else "#fee08b" if row["boruta_confirmed"]
                  else "#bdbdbd"
                  for _, row in perf.iterrows()]
    ax.barh(range(len(perf)), perf["consensus_weight"], color=bar_colors)
    ax.set_yticks(range(len(perf)))
    labels = [(("(+) " if d == "higher" else "(-) ") + n[:45])
              for d, n in zip(perf["direction"], perf["feature_name"])]
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Consensus Weight (4 methods)")
    ax.set_title(f"Top {n_top_features} Performance KPI Weights\n"
                 "Green = significant + Boruta | Light green = significant | "
                 "Yellow = Boruta confirmed | Grey = model/stability only")
    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor="#1a9641", label="Significant + Boruta"),
        Patch(facecolor="#a6d96a", label="Statistically significant"),
        Patch(facecolor="#fee08b", label="Boruta confirmed"),
        Patch(facecolor="#bdbdbd", label="Model / stability only"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT / "kpi_weights.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: kpi_weights.png")

    return results


# =========================================================================
#  STEP 3 — Score Goalkeepers 1-100
# =========================================================================

def step3_score_goalkeepers(df, X, y, weights_df):
    """Select top KPIs, train model, score every keeper 1-100."""
    print("\n" + "=" * 70)
    print(f"STEP 3 — SCORE GOALKEEPERS (1-100, top {n_top_features} KPIs)")
    print("=" * 70)

    top = weights_df.head(n_top_features)
    selected = top["feature"].tolist()

    top[["rank", "feature_name", "consensus_weight", "direction", "significant",
         "boruta_confirmed", "stability_pct"]].to_csv(OUTPUT / "selected_features.csv", index=False)

    X_sel = X[selected].copy()
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Random Forest (best model for performance-only KPIs)
    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42,
        max_depth=6, min_samples_leaf=5, n_jobs=-1)
    rf_proba = cross_val_predict(rf, X_sel, y, cv=cv, method="predict_proba")[:, 1]
    rf_auc = roc_auc_score(y, rf_proba)
    rf_f1 = f1_score(y, (rf_proba > 0.5).astype(int))
    print(f"  Random Forest      AUC={rf_auc:.3f}  F1={rf_f1:.3f}")

    # XGBoost for comparison
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        scale_pos_weight=n_neg / max(n_pos, 1), random_state=42,
        eval_metric="logloss", verbosity=0, colsample_bytree=0.5, subsample=0.8)
    xgb_proba = cross_val_predict(xgb_model, X_sel, y, cv=cv, method="predict_proba")[:, 1]
    xgb_auc = roc_auc_score(y, xgb_proba)
    xgb_f1 = f1_score(y, (xgb_proba > 0.5).astype(int))
    print(f"  XGBoost            AUC={xgb_auc:.3f}  F1={xgb_f1:.3f}")

    # Ensemble
    ensemble_proba = 0.5 * rf_proba + 0.5 * xgb_proba
    ens_auc = roc_auc_score(y, ensemble_proba)
    ens_f1 = f1_score(y, (ensemble_proba > 0.5).astype(int))
    print(f"  Ensemble           AUC={ens_auc:.3f}  F1={ens_f1:.3f}")
    print(f"  Baseline: {y.mean():.1%} PLAYS rate")

    # Weighted KPI performance score
    feat_weights = top.set_index("feature")["consensus_weight"]
    X_norm = pd.DataFrame(MinMaxScaler().fit_transform(X_sel), columns=X_sel.columns, index=X_sel.index)
    kpi_pct = X_norm.multiply(feat_weights, axis=1).sum(axis=1).rank(pct=True)

    # Final score: 60% model + 40% KPI performance, percentile-ranked to 1-100
    raw_score = 0.6 * ensemble_proba + 0.4 * kpi_pct.values
    scouting_score = pd.Series(raw_score).rank(pct=True).values * 99 + 1

    # Build output
    score_df = df[["playerId", "name", "status", "age", "origin_team",
                    "origin_comp", "origin_median", "origin_matches"]].copy()
    score_df["raw_probability"] = np.round(ensemble_proba, 4)
    score_df["scouting_score"] = np.round(scouting_score, 1)
    score_df = score_df.sort_values("scouting_score", ascending=False)

    print(f"\n  Scouting score by status:")
    for status in STATUS_ORDER:
        subset = score_df[score_df["status"] == status]
        if len(subset) > 0:
            print(f"    {status:10s}  median={subset['scouting_score'].median():.0f}  n={len(subset)}")

    # Save
    score_df.to_csv(OUTPUT / "scouting_scores.csv", index=False)
    targets = score_df[score_df["status"] == "STAYED"].head(20)
    targets.to_csv(OUTPUT / "scouting_targets.csv", index=False)
    print(f"\n  Saved: scouting_scores.csv ({len(score_df)} keepers)")
    print(f"  Saved: scouting_targets.csv (top 20 hidden gems)")

    # ── Visualizations ──
    # Score distribution
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    for status in STATUS_ORDER:
        subset = score_df[score_df["status"] == status]["scouting_score"]
        ax.hist(subset, bins=20, alpha=0.5, label=f"{status} (n={len(subset)})", color=COLORS.get(status))
    ax.set_xlabel("Scouting Score (1-100)"); ax.set_ylabel("Count")
    ax.set_title("Score Distribution by Status"); ax.legend()

    ax = axes[1]
    data_box = [score_df[score_df["status"] == s]["scouting_score"].values for s in STATUS_ORDER]
    bp = ax.boxplot(data_box, labels=STATUS_ORDER, patch_artist=True)
    for patch, status in zip(bp["boxes"], STATUS_ORDER):
        patch.set_facecolor(COLORS[status]); patch.set_alpha(0.6)
    ax.set_ylabel("Scouting Score"); ax.set_title("Score by Status")
    plt.tight_layout()
    plt.savefig(OUTPUT / "score_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Feature importance (from RF, the better model)
    rf.fit(X_sel, y)
    feat_imp = pd.DataFrame({
        "feature": selected, "feature_name": [f.replace("mean_", "") for f in selected],
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)
    feat_imp.to_csv(OUTPUT / "feature_importance.csv", index=False)

    top_fi = feat_imp.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.barh(range(len(top_fi)), top_fi["importance"], color="#2196F3")
    ax.set_yticks(range(len(top_fi)))
    ax.set_yticklabels([f[:45] for f in top_fi["feature_name"]], fontsize=8)
    ax.set_xlabel("Feature Importance (Random Forest)")
    ax.set_title(f"Top {min(20, len(top_fi))} Features in Final Scouting Model")
    plt.tight_layout()
    plt.savefig(OUTPUT / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Radar chart
    radar_features = top.head(8)["feature"].tolist()
    radar_labels = [f.replace("mean_", "")[:22] for f in radar_features]
    all_v = X[radar_features]
    plays_vals = X.loc[y == 1, radar_features].median().values
    rest_vals = X.loc[y == 0, radar_features].median().values
    plays_pctiles = [(all_v[c] < plays_vals[i]).mean() for i, c in enumerate(radar_features)]
    rest_pctiles = [(all_v[c] < rest_vals[i]).mean() for i, c in enumerate(radar_features)]
    n = len(radar_labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist() + [0]
    plays_pctiles += plays_pctiles[:1]; rest_pctiles += rest_pctiles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.plot(angles, plays_pctiles, "o-", linewidth=2.5, color=COLORS["PLAYS"],
            label="Progressors (PLAYS)", markersize=8)
    ax.fill(angles, plays_pctiles, alpha=0.15, color=COLORS["PLAYS"])
    ax.plot(angles, rest_pctiles, "o-", linewidth=2.5, color="#95a5a6",
            label="Non-progressors (REST)", markersize=8)
    ax.fill(angles, rest_pctiles, alpha=0.1, color="#95a5a6")
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(radar_labels, fontsize=8)
    ax.set_ylim(0, 1); ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(["25th", "50th", "75th"], fontsize=7)
    ax.set_title("Progression Profile (Top 8 Weighted KPIs)", fontsize=14, pad=30)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(OUTPUT / "radar_profile.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y, ensemble_proba)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f"Ensemble (AUC={ens_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC Curve: Predicting Goalkeeper Progression\n"
                 "(performance KPIs only — no league strength)"); ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("  Saved: score_distribution.png, feature_importance.png, radar_profile.png, roc_curve.png")
    return score_df


# =========================================================================
#  MAIN
# =========================================================================

def main():
    global n_top_features
    n_top_features = _load_optimal_n()

    print("=" * 70)
    print("  GOALKEEPER SCOUTING PIPELINE — Finding the New Number 1")
    print(f"  462 performance KPIs → top {n_top_features} (data-driven) → score 1-100")
    print("=" * 70)

    df = load_data()
    X, y, feature_names = prepare_features(df)
    results = step1_find_important_kpis(X, y, feature_names)
    weights = step2_weight_kpis(results, df, feature_names)
    scores = step3_score_goalkeepers(df, X, y, weights)

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n  Outputs: {OUTPUT.resolve()}")
    for f in sorted(OUTPUT.iterdir()):
        if f.suffix in [".csv", ".png"]:
            print(f"    {f.name}")

    print(f"\n  Top 5 hidden gems (STAYED keepers):")
    targets = scores[scores["status"] == "STAYED"].head(5)
    for _, r in targets.iterrows():
        print(f"    Score {r['scouting_score']:5.1f}  {r['name']:<30s}  {r['origin_comp']}")


if __name__ == "__main__":
    main()
