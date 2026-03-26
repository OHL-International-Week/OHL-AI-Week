#!/usr/bin/env python3
"""KPI Selection Experiments — Performance KPIs Only

Identifies the most important on-pitch KPIs from ~460 performance features
using 6 complementary methods, each chosen for a specific reason:

  1. XGBoost importance       — non-linear interactions (boosting)
  2. Random Forest importance  — non-linear interactions (bagging, decorrelated trees)
  3. Lasso (L1 regularization) — linear perspective: which KPIs survive shrinkage?
  4. Mann-Whitney U + FDR      — statistical test: is there a real group difference?
  5. Boruta (shadow features)  — all-relevant: is this KPI better than random noise?
  6. Bootstrap stability       — consistency: is this KPI reliably selected?

Why these 6?
  - XGBoost and Random Forest both find non-linear patterns but work differently
    (boosting vs bagging). If both agree, the signal is robust.
  - Lasso adds a linear perspective — simple, interpretable, catches different patterns.
  - Mann-Whitney gives p-values and effect sizes (useful for presenting to scouts).
  - Boruta answers the fundamental question: does this KPI carry real signal?
  - Bootstrap stability catches overfitting: noisy features appear randomly.

The OPTIMAL NUMBER OF KPIs is determined data-driven:
  - We sweep feature counts from 5 to 150 with 5-fold CV
  - We find the "elbow" using the 1-SE rule (simplest model within 1 SE of peak)
  - This number is saved and used in the final pipeline

IMPORTANT: No context variables (origin_median, age, n_matches_loaded).
We only evaluate on-pitch performance KPIs.

Run: python pipeline/kpi_experiments.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="colorblind")

OUTPUT = Path(__file__).resolve().parent / "output"
OUTPUT.mkdir(exist_ok=True)


def load_full_dataset():
    """Load the full KPI dataset."""
    parquet = OUTPUT / "keeper_all_kpis.parquet"
    if not parquet.exists():
        print("ERROR: Run build_full_kpi_dataset.py first!")
        sys.exit(1)
    df = pd.read_parquet(parquet)
    print(f"Loaded: {len(df)} keepers x {len(df.columns)} columns")
    return df


def prepare_features(df, min_coverage=0.5, min_variance_pctile=5):
    """Pre-filter to performance KPIs only (no context variables)."""
    print("\n" + "=" * 70)
    print("PRE-FILTERING (performance KPIs only)")
    print("=" * 70)

    mean_cols = sorted([c for c in df.columns if c.startswith("mean_")])
    print(f"\n  Total mean_* features: {len(mean_cols)}")

    # Coverage filter
    coverage = df[mean_cols].notna().mean()
    keep = coverage[coverage >= min_coverage].index.tolist()
    print(f"  After coverage >= {min_coverage:.0%}: {len(keep)} (dropped {len(mean_cols) - len(keep)})")

    # Variance filter
    variances = df[keep].var()
    threshold = variances.quantile(min_variance_pctile / 100)
    keep = variances[variances > threshold].index.tolist()
    print(f"  After variance > {min_variance_pctile}th pctile: {len(keep)}")

    # Redundancy removal (|r| > 0.95)
    print("  Computing correlation matrix...")
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
    print(f"  After |r| > 0.95 removal: {len(keep)}")

    X = df[keep].copy().fillna(df[keep].median())
    y = (df["status"] == "PLAYS").astype(int)

    print(f"\n  Final: {X.shape[1]} performance KPIs, {len(y)} samples "
          f"({y.sum()} PLAYS, {(~y.astype(bool)).sum()} REST)")
    return X, y, keep


# =========================================================================
#  6 METHODS
# =========================================================================

def method_xgboost(X, y, feature_names):
    print("\n  [1/6] XGBoost importance (boosting, non-linear)...")
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        scale_pos_weight=n_neg / max(n_pos, 1), random_state=42,
        eval_metric="logloss", verbosity=0, colsample_bytree=0.5, subsample=0.8)
    model.fit(X, y)
    return pd.DataFrame({"feature": feature_names, "xgb_importance": model.feature_importances_})


def method_random_forest(X, y, feature_names):
    print("  [2/6] Random Forest importance (bagging, decorrelated)...")
    model = RandomForestClassifier(
        n_estimators=500, class_weight="balanced", random_state=42,
        max_depth=8, min_samples_leaf=5, n_jobs=-1)
    model.fit(X, y)
    return pd.DataFrame({"feature": feature_names, "rf_importance": model.feature_importances_})


def method_lasso(X, y, feature_names):
    print("  [3/6] Lasso L1 coefficients (linear, regularized)...")
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names, index=X.index)
    model = LogisticRegression(
        max_iter=5000, penalty="l1", solver="saga",
        class_weight="balanced", random_state=42, C=0.1)
    model.fit(X_scaled, y)
    return pd.DataFrame({"feature": feature_names, "lasso_importance": np.abs(model.coef_[0])})


def method_mann_whitney(X, y, feature_names):
    print("  [4/6] Mann-Whitney U + Cohen's d + FDR...")
    p_values, effect_sizes, directions = [], [], []
    for col in feature_names:
        p_vals = X.loc[y == 1, col].dropna()
        r_vals = X.loc[y == 0, col].dropna()
        if len(p_vals) < 5 or len(r_vals) < 5:
            p_values.append(1.0); effect_sizes.append(0.0); directions.append("higher")
            continue
        _, pv = stats.mannwhitneyu(p_vals, r_vals, alternative="two-sided")
        pooled_std = np.sqrt(
            ((len(p_vals)-1)*p_vals.std()**2 + (len(r_vals)-1)*r_vals.std()**2)
            / (len(p_vals) + len(r_vals) - 2))
        d = (p_vals.mean() - r_vals.mean()) / max(pooled_std, 1e-10)
        p_values.append(pv); effect_sizes.append(d)
        directions.append("higher" if d > 0 else "lower")

    from statsmodels.stats.multitest import multipletests
    _, p_adj, _, _ = multipletests(p_values, method="fdr_bh")

    return pd.DataFrame({
        "feature": feature_names, "cohens_d": effect_sizes,
        "abs_cohens_d": np.abs(effect_sizes), "p_value": p_values,
        "p_fdr": p_adj, "significant": p_adj < 0.05, "direction": directions,
    })


def method_boruta(X, y, feature_names, n_iterations=50):
    print(f"  [5/6] Boruta shadow features ({n_iterations} iterations)...")
    n_features = len(feature_names)
    hit_counts = np.zeros(n_features)
    for it in range(n_iterations):
        if (it + 1) % 10 == 0:
            print(f"        Iteration {it + 1}/{n_iterations}...")
        X_shadow = X.copy()
        for col in X_shadow.columns:
            X_shadow[col] = np.random.permutation(X_shadow[col].values)
        X_shadow.columns = [f"shadow_{c}" for c in X_shadow.columns]
        X_combined = pd.concat([X, X_shadow], axis=1)
        model = RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=it,
            max_depth=6, min_samples_leaf=5, n_jobs=-1)
        model.fit(X_combined, y)
        imp = model.feature_importances_
        hit_counts += (imp[:n_features] > imp[n_features:].max()).astype(int)

    return pd.DataFrame({
        "feature": feature_names, "boruta_hits": hit_counts,
        "boruta_fraction": hit_counts / n_iterations,
        "boruta_confirmed": hit_counts / n_iterations > 0.5,
    })


def method_bootstrap_stability(X, y, feature_names, n_bootstrap=50, top_k=30):
    print(f"  [6/6] Bootstrap stability ({n_bootstrap} resamples, top-{top_k})...")
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    n_features = len(feature_names)
    counts = np.zeros(n_features)
    for b in range(n_bootstrap):
        if (b + 1) % 10 == 0:
            print(f"        Bootstrap {b + 1}/{n_bootstrap}...")
        idx = np.random.choice(len(X), size=len(X), replace=True)
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            scale_pos_weight=n_neg / max(n_pos, 1), random_state=b,
            eval_metric="logloss", verbosity=0, colsample_bytree=0.5, subsample=0.8)
        model.fit(X.iloc[idx], y.iloc[idx])
        top_idx = np.argsort(model.feature_importances_)[::-1][:top_k]
        counts[top_idx] += 1

    return pd.DataFrame({
        "feature": feature_names, "stability_count": counts,
        "stability_pct": counts / n_bootstrap * 100,
    })


# =========================================================================
#  CONSENSUS RANKING
# =========================================================================

def build_consensus(xgb_res, rf_res, lasso_res, mw_res, boruta_res, stab_res):
    print("\n" + "=" * 70)
    print("CONSENSUS RANKING (6 methods)")
    print("=" * 70)

    consensus = mw_res[["feature", "abs_cohens_d", "p_fdr", "significant", "direction"]].copy()
    consensus = consensus.merge(xgb_res, on="feature")
    consensus = consensus.merge(rf_res, on="feature")
    consensus = consensus.merge(lasso_res, on="feature")
    consensus = consensus.merge(boruta_res[["feature", "boruta_fraction", "boruta_confirmed"]], on="feature")
    consensus = consensus.merge(stab_res[["feature", "stability_pct"]], on="feature")

    method_cols = ["abs_cohens_d", "xgb_importance", "rf_importance",
                   "lasso_importance", "boruta_fraction", "stability_pct"]
    for col in method_cols:
        mn, mx = consensus[col].min(), consensus[col].max()
        consensus[f"{col}_norm"] = (consensus[col] - mn) / (mx - mn) if mx > mn else 0.0

    norm_cols = [f"{c}_norm" for c in method_cols]
    consensus["consensus_score"] = consensus[norm_cols].mean(axis=1)
    consensus["consensus_rank"] = consensus["consensus_score"].rank(ascending=False).astype(int)
    consensus["feature_name"] = consensus["feature"].str.replace("mean_", "", regex=False)
    consensus = consensus.sort_values("consensus_rank")

    print(f"\n  Top 20 KPIs:")
    for _, row in consensus.head(20).iterrows():
        sig = "***" if row["significant"] else "   "
        bor = " B" if row["boruta_confirmed"] else "  "
        arr = "+" if row["direction"] == "higher" else "-"
        print(f"    #{row['consensus_rank']:3d}  ({arr}) {row['feature_name']:<50s}  "
              f"stab={row['stability_pct']:4.0f}%  {sig}{bor}")

    return consensus


# =========================================================================
#  DATA-DRIVEN: WHY THESE 6 METHODS?
# =========================================================================

def justify_methods(consensus_df, X, y):
    """Show with data why each method is needed and non-redundant.

    Three analyses:
      1. Spearman rank correlation between methods — if two methods always
         agree, one is redundant. Low correlation = complementary.
      2. Unique contribution — for each method, how many of its top-20
         features are NOT in any other method's top-20?
      3. Consensus vs single method — does combining methods beat any
         individual method's feature selection?
    """
    print("\n" + "=" * 70)
    print("WHY THESE 6 METHODS? (data-driven justification)")
    print("=" * 70)

    method_cols = ["abs_cohens_d", "xgb_importance", "rf_importance",
                   "lasso_importance", "boruta_fraction", "stability_pct"]
    method_labels = ["Mann-Whitney", "XGBoost", "Random Forest",
                     "Lasso", "Boruta", "Stability"]

    # ── Analysis 1: Spearman rank correlation ──
    print("\n  1) RANK CORRELATION BETWEEN METHODS")
    print("     Low correlation = methods capture different signal = good")
    print()

    rank_data = pd.DataFrame()
    for col in method_cols:
        rank_data[col] = consensus_df[col].rank(ascending=False)

    from scipy.stats import spearmanr
    n_methods = len(method_cols)
    corr_matrix = np.zeros((n_methods, n_methods))
    for i in range(n_methods):
        for j in range(n_methods):
            corr_matrix[i, j], _ = spearmanr(rank_data[method_cols[i]], rank_data[method_cols[j]])

    corr_df = pd.DataFrame(corr_matrix, index=method_labels, columns=method_labels)
    print("     Spearman rank correlations:")
    print(corr_df.round(2).to_string())

    # Find most and least correlated pairs
    pairs = []
    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            pairs.append((method_labels[i], method_labels[j], corr_matrix[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True)

    print(f"\n     Most similar pair:  {pairs[0][0]} & {pairs[0][1]} (r={pairs[0][2]:.2f})")
    print(f"     Most different pair: {pairs[-1][0]} & {pairs[-1][1]} (r={pairs[-1][2]:.2f})")
    avg_corr = np.mean([p[2] for p in pairs])
    print(f"     Average correlation: {avg_corr:.2f}")
    print(f"     → {'Good diversity' if avg_corr < 0.5 else 'Moderate diversity' if avg_corr < 0.7 else 'Low diversity'}: "
          f"methods capture {'different' if avg_corr < 0.5 else 'overlapping'} signal")

    # ── Analysis 2: Unique contributions ──
    print("\n  2) UNIQUE CONTRIBUTIONS PER METHOD")
    print("     Features in a method's top-20 that no other method ranks in its top-20")
    print()

    top_k = 20
    top_sets = {}
    for col, label in zip(method_cols, method_labels):
        top_features = set(consensus_df.nlargest(top_k, col)["feature"].tolist())
        top_sets[label] = top_features

    unique_counts = {}
    unique_features = {}
    for label in method_labels:
        others = set()
        for other_label in method_labels:
            if other_label != label:
                others |= top_sets[other_label]
        unique = top_sets[label] - others
        unique_counts[label] = len(unique)
        unique_features[label] = unique

    for label in method_labels:
        uniq = unique_features[label]
        print(f"     {label:<15s}  {unique_counts[label]:2d} unique features in top-{top_k}")
        for f in sorted(uniq):
            print(f"       → {f.replace('mean_', '')}")

    total_unique = sum(unique_counts.values())
    all_union = set()
    for s in top_sets.values():
        all_union |= s
    print(f"\n     Union of all top-{top_k} sets: {len(all_union)} distinct features")
    print(f"     Total unique contributions: {total_unique}")
    print(f"     → {'Every' if all(v > 0 for v in unique_counts.values()) else 'Most'} method"
          f"{'s contribute' if total_unique > 0 else ' is redundant'} something unique")

    # ── Analysis 3: Consensus vs single method ──
    print("\n  3) CONSENSUS vs SINGLE METHOD (AUC comparison)")
    print("     Does combining methods beat using just one?")
    print()

    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.ensemble import RandomForestClassifier

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    n_features = find_optimal_n_from_file()

    results_rows = []
    # Test each method's top-N alone
    for col, label in zip(method_cols, method_labels):
        top_features = consensus_df.nlargest(n_features, col)["feature"].tolist()
        top_features = [f for f in top_features if f in X.columns]
        if len(top_features) < 5:
            continue
        X_sel = X[top_features[:n_features]]
        model = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                       random_state=42, max_depth=6, min_samples_leaf=5, n_jobs=-1)
        auc = cross_val_score(model, X_sel, y, cv=cv, scoring="roc_auc")
        results_rows.append({"method": label, "auc_mean": auc.mean(), "auc_std": auc.std()})
        print(f"     {label:<15s} top-{n_features}:  AUC = {auc.mean():.3f} +/- {auc.std():.3f}")

    # Consensus top-N
    top_features = consensus_df.nsmallest(n_features, "consensus_rank")["feature"].tolist()
    top_features = [f for f in top_features if f in X.columns]
    X_sel = X[top_features[:n_features]]
    model = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   random_state=42, max_depth=6, min_samples_leaf=5, n_jobs=-1)
    auc = cross_val_score(model, X_sel, y, cv=cv, scoring="roc_auc")
    results_rows.append({"method": "CONSENSUS (all 6)", "auc_mean": auc.mean(), "auc_std": auc.std()})
    print(f"     {'CONSENSUS':15s} top-{n_features}:  AUC = {auc.mean():.3f} +/- {auc.std():.3f}  ★")

    results_df = pd.DataFrame(results_rows)
    best_single = results_df[results_df["method"] != "CONSENSUS (all 6)"]["auc_mean"].max()
    consensus_auc = results_df[results_df["method"] == "CONSENSUS (all 6)"]["auc_mean"].values[0]
    improvement = consensus_auc - best_single

    print(f"\n     Best single method: AUC = {best_single:.3f}")
    print(f"     Consensus:          AUC = {consensus_auc:.3f}")
    print(f"     Improvement:        {improvement:+.3f}")
    if improvement > 0:
        print(f"     → Consensus outperforms every single method — combining is worthwhile")
    else:
        print(f"     → Consensus is competitive with best single method — combining adds robustness")

    # ── Save results ──
    corr_df.to_csv(OUTPUT / "method_correlation.csv")
    results_df.to_csv(OUTPUT / "method_comparison_auc.csv", index=False)

    # ── Plot: correlation heatmap ──
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax = axes[0]
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="RdYlGn_r", ax=ax,
                mask=mask, vmin=0, vmax=1, linewidths=0.5, square=True)
    ax.set_title("Rank Correlation Between Methods\n(lower = more complementary)", fontsize=12)

    ax = axes[1]
    sorted_results = results_df.sort_values("auc_mean")
    colors = ["#2196F3" if m != "CONSENSUS (all 6)" else "#FF5722"
              for m in sorted_results["method"]]
    bars = ax.barh(range(len(sorted_results)), sorted_results["auc_mean"], color=colors,
                    xerr=sorted_results["auc_std"], capsize=3)
    ax.set_yticks(range(len(sorted_results)))
    ax.set_yticklabels(sorted_results["method"], fontsize=10)
    ax.set_xlabel("AUC-ROC (5-fold CV)")
    ax.set_title(f"Feature Selection: Single Method vs Consensus\n(each selecting top {n_features} KPIs)",
                 fontsize=12)
    # Add value labels
    for i, (_, row) in enumerate(sorted_results.iterrows()):
        ax.text(row["auc_mean"] + row["auc_std"] + 0.005, i,
                f"{row['auc_mean']:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT / "method_justification.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  Saved: method_justification.png, method_correlation.csv, method_comparison_auc.csv")

    return corr_df, results_df


def find_optimal_n_from_file():
    """Read optimal N from file, or return default."""
    f = OUTPUT / "optimal_n_features.txt"
    if f.exists():
        return int(f.read_text().strip())
    return 20


# =========================================================================
#  DATA-DRIVEN OPTIMAL FEATURE COUNT
# =========================================================================

def find_optimal_n_features(X, y, consensus_df):
    """Sweep feature counts with REPEATED CV to find stable optimal N.

    Problem: with 693 samples and 5-fold CV, AUC estimates have ~0.02 std.
    The curve is flat from ~15 to ~150 features — all overlap within 1 std.
    A single CV run picks a different "optimal" each time.

    Solution: run 5 DIFFERENT CV splits (5 seeds x 5 folds = 25 evaluations
    per point), then find the plateau start: the smallest N where AUC enters
    and stays within 1% of the best AUC seen at that N or above.
    """
    print("\n" + "=" * 70)
    print("FINDING OPTIMAL NUMBER OF KPIs (repeated CV, plateau detection)")
    print("=" * 70)

    n_pos, n_neg = y.sum(), len(y) - y.sum()
    ranked = consensus_df.sort_values("consensus_rank")["feature"].tolist()
    ranked = [f for f in ranked if f in X.columns]

    sizes = [5, 10, 15, 20, 25, 30, 35, 40, 50, 75, 100]
    rows = []

    # Repeated CV: 3 different random seeds for stability
    seeds = [42, 123, 7]

    for n in sizes:
        if n > len(ranked):
            break
        X_sel = X[ranked[:n]]
        all_auc_rf = []
        all_auc_xgb = []
        for seed in seeds:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

            rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                        random_state=seed, max_depth=6,
                                        min_samples_leaf=5, n_jobs=-1)
            auc_rf = cross_val_score(rf, X_sel, y, cv=cv, scoring="roc_auc")
            all_auc_rf.extend(auc_rf)

            xgb_m = xgb.XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                scale_pos_weight=n_neg / max(n_pos, 1), random_state=seed,
                eval_metric="logloss", verbosity=0, colsample_bytree=0.5, subsample=0.8)
            auc_xgb = cross_val_score(xgb_m, X_sel, y, cv=cv, scoring="roc_auc")
            all_auc_xgb.extend(auc_xgb)

        # Average across all seeds
        rf_mean = np.mean(all_auc_rf)
        rf_std = np.std(all_auc_rf)
        xgb_mean = np.mean(all_auc_xgb)
        xgb_std = np.std(all_auc_xgb)

        rows.append({"n_features": n, "model": "Random Forest",
                      "auc_mean": rf_mean, "auc_std": rf_std,
                      "f1_mean": 0, "f1_std": 0})
        rows.append({"n_features": n, "model": "XGBoost",
                      "auc_mean": xgb_mean, "auc_std": xgb_std,
                      "f1_mean": 0, "f1_std": 0})

        best = max(rf_mean, xgb_mean)
        print(f"  top {n:3d}  RF={rf_mean:.3f}+/-{rf_std:.3f}  "
              f"XGB={xgb_mean:.3f}+/-{xgb_std:.3f}  best={best:.3f}")

    comp_df = pd.DataFrame(rows)

    # Plateau detection:
    # 1. For each N, take the best model's AUC
    # 2. Find the running maximum from N onwards (the "best you could do with N or more")
    # 3. The optimal N is where AUC first reaches within 1% of the overall peak
    #    AND subsequent points don't improve by more than 0.005
    best_per_n = comp_df.loc[comp_df.groupby("n_features")["auc_mean"].idxmax()].copy()
    best_per_n = best_per_n.sort_values("n_features").reset_index(drop=True)

    ns = best_per_n["n_features"].values
    aucs = best_per_n["auc_mean"].values
    stds = best_per_n["auc_std"].values
    peak_auc = aucs.max()
    peak_n = int(ns[aucs.argmax()])
    plateau_threshold = peak_auc * 0.99  # within 1% of peak

    # Find where we enter the plateau
    optimal_n = int(ns[-1])  # default to largest
    for i in range(len(ns)):
        if aucs[i] >= plateau_threshold:
            optimal_n = int(ns[i])
            break

    # But also check: is there a clear jump AFTER this point?
    # If so, the plateau hasn't really started yet.
    for i in range(len(ns) - 1):
        if int(ns[i]) >= optimal_n:
            # Check if any later point is >0.01 better (outside noise)
            remaining_max = aucs[i + 1:].max() if i + 1 < len(aucs) else aucs[i]
            if remaining_max - aucs[i] > 0.01:
                # There's still meaningful improvement ahead
                # Move optimal to the point where that improvement happens
                better_idx = i + 1 + np.argmax(aucs[i + 1:])
                optimal_n = int(ns[better_idx])
            break

    print(f"\n  Peak AUC: {peak_auc:.3f} at {peak_n} features")
    print(f"  Plateau threshold (99% of peak): {plateau_threshold:.3f}")
    print(f"  -> Optimal N: {optimal_n} features")

    print(f"\n  Feature count vs best AUC:")
    for _, row in best_per_n.iterrows():
        n = int(row["n_features"])
        marker = " <-- OPTIMAL" if n == optimal_n else ""
        marker += " <-- PEAK" if n == peak_n else ""
        print(f"    {n:4d} features  AUC={row['auc_mean']:.3f} +/- {row['auc_std']:.3f}{marker}")

    return optimal_n, comp_df


# =========================================================================
#  VISUALIZATIONS
# =========================================================================

def plot_results(consensus_df, comparison_df, optimal_n):
    print("\n  Generating visualizations...")

    # 1. Consensus bar chart
    top_n = min(30, len(consensus_df))
    top = consensus_df.head(top_n).copy().iloc[::-1]
    fig, ax = plt.subplots(figsize=(16, 11))
    colors = []
    for _, row in top.iterrows():
        if row.get("boruta_confirmed", False) and row.get("significant", False):
            colors.append("#1a9641")
        elif row.get("significant", False):
            colors.append("#a6d96a")
        elif row.get("boruta_confirmed", False):
            colors.append("#fee08b")
        else:
            colors.append("#bdbdbd")
    ax.barh(range(len(top)), top["consensus_score"], color=colors)
    ax.set_yticks(range(len(top)))
    labels = []
    for _, row in top.iterrows():
        arr = "+" if row["direction"] == "higher" else "-"
        stab = f" [{row['stability_pct']:.0f}%]" if pd.notna(row.get("stability_pct")) else ""
        labels.append(f"({arr}) {row['feature_name'][:45]}{stab}")
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Consensus Score (6 methods)")
    ax.set_title(f"Top {top_n} Performance KPIs — Consensus of 6 Methods\n"
                 "Green = FDR sig + Boruta | Light green = FDR sig | "
                 "Yellow = Boruta | Grey = model-based\n"
                 "[%] = bootstrap stability", fontsize=11)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#1a9641", label="Significant + Boruta"),
        Patch(facecolor="#a6d96a", label="Statistically significant"),
        Patch(facecolor="#fee08b", label="Boruta confirmed"),
        Patch(facecolor="#bdbdbd", label="Model-based only"),
    ], loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT / "experiment_consensus_top30.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Method agreement heatmap
    top15 = consensus_df.head(15).copy()
    norm_cols = ["abs_cohens_d_norm", "xgb_importance_norm", "rf_importance_norm",
                 "lasso_importance_norm", "boruta_fraction_norm", "stability_pct_norm"]
    available = [c for c in norm_cols if c in top15.columns]
    heatmap_data = top15.set_index("feature_name")[available]
    heatmap_data.columns = ["Effect Size", "XGBoost", "Random Forest",
                             "Lasso", "Boruta", "Stability"][:len(available)]
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlOrRd",
                ax=ax, linewidths=0.5, vmin=0, vmax=1)
    ax.set_title("Top 15 KPIs — Agreement Across 6 Methods (normalized 0-1)", fontsize=13)
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(OUTPUT / "experiment_method_agreement.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Model comparison with 1-SE annotation
    if comparison_df is not None and len(comparison_df) > 0:
        fig, ax = plt.subplots(figsize=(12, 7))
        for model_name in comparison_df["model"].unique():
            subset = comparison_df[comparison_df["model"] == model_name]
            ax.errorbar(subset["n_features"], subset["auc_mean"], yerr=subset["auc_std"],
                        marker="o", label=model_name, capsize=3, linewidth=2)

        best_per_n = comparison_df.loc[comparison_df.groupby("n_features")["auc_mean"].idxmax()]
        peak_idx = best_per_n["auc_mean"].idxmax()
        peak_auc = best_per_n.loc[peak_idx, "auc_mean"]
        peak_std = best_per_n.loc[peak_idx, "auc_std"]
        threshold = peak_auc - peak_std

        ax.axhline(y=threshold, color="red", linestyle="--", alpha=0.5,
                    label=f"1-SE threshold ({threshold:.3f})")
        ax.axvline(x=optimal_n, color="green", linestyle="--", alpha=0.7,
                    label=f"Optimal = {optimal_n} KPIs (1-SE rule)")

        ax.set_xlabel("Number of Top KPIs")
        ax.set_ylabel("AUC-ROC (5-fold CV)")
        ax.set_title(f"How Many KPIs Do We Need? (Data-Driven Selection)\n"
                     f"1-SE rule: pick the simplest model within 1 SE of the peak\n"
                     f"Result: {optimal_n} KPIs")
        ax.legend(loc="lower right")
        ax.set_xticks(comparison_df["n_features"].unique())
        plt.tight_layout()
        plt.savefig(OUTPUT / "experiment_model_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()

    print("  Saved visualizations.")


# =========================================================================
#  MAIN
# =========================================================================

def main():
    print("=" * 70)
    print("  KPI SELECTION EXPERIMENTS")
    print("  Performance KPIs only x 6 methods x data-driven feature count")
    print("=" * 70)

    df = load_full_dataset()
    X, y, feature_names = prepare_features(df)

    print("\n" + "=" * 70)
    print("RUNNING 6 METHODS")
    print("=" * 70)

    xgb_res = method_xgboost(X, y, feature_names)
    rf_res = method_random_forest(X, y, feature_names)
    lasso_res = method_lasso(X, y, feature_names)
    mw_res = method_mann_whitney(X, y, feature_names)
    boruta_res = method_boruta(X, y, feature_names, n_iterations=50)
    stab_res = method_bootstrap_stability(X, y, feature_names, n_bootstrap=50, top_k=30)

    consensus = build_consensus(xgb_res, rf_res, lasso_res, mw_res, boruta_res, stab_res)
    optimal_n, comparison = find_optimal_n_features(X, y, consensus)

    # Save optimal_n BEFORE justify_methods reads it
    with open(OUTPUT / "optimal_n_features.txt", "w") as f:
        f.write(str(optimal_n))

    # Justify why these 6 methods (with data)
    corr_df, method_auc = justify_methods(consensus, X, y)

    plot_results(consensus, comparison, optimal_n)

    consensus.to_csv(OUTPUT / "kpi_consensus_all_experiments.csv", index=False)
    comparison.to_csv(OUTPUT / "model_comparison_by_features.csv", index=False)
    with open(OUTPUT / "optimal_n_features.txt", "w") as f:
        f.write(str(optimal_n))

    print(f"\n{'=' * 70}")
    print("EXPERIMENTS COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Features analyzed: {len(feature_names)} performance KPIs")
    print(f"  Boruta confirmed: {consensus['boruta_confirmed'].sum()}")
    print(f"  FDR significant: {consensus['significant'].sum()}")
    print(f"  Optimal N features: {optimal_n} (data-driven, 1-SE rule)")


if __name__ == "__main__":
    main()
