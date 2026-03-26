#!/usr/bin/env python3
"""Central Question — Definitive Answer

Synthesizes Phase 1 (KPI weights), Q1 (discriminating metrics), Q2 (prediction),
and Q3 (signal vs noise) into a unified analysis with:
- Consolidated cross-reference table
- Core/Supporting/Discarded metric profiles
- Threshold analysis for actionable scouting
- Validation with case studies
- Visual scorecards

Run: python -m Central_Question.analysis
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from shared.data_utils import (
    load_definitions, load_and_aggregate_data, select_features,
    get_cache_path, STATUS_ORDER, META_COLS, PROJECT_ROOT
)

OUTPUT = Path(__file__).resolve().parent / "results"
OUTPUT.mkdir(exist_ok=True)

# ── Color palette ──────────────────────────────────────────────────────
COLORS = {
    "PLAYS": "#2ecc71", "BENCH": "#f39c12",
    "STAYED": "#3498db", "DROPPED": "#e74c3c",
    "core": "#1a9641", "supporting": "#fdae61", "discarded": "#d7191c",
}


# =========================================================================
# STEP 1 — Synthesize all prior findings
# =========================================================================

def load_prior_results():
    """Load outputs from Phase 1, Q1, Q2, Q3."""
    base = PROJECT_ROOT

    weights = pd.read_csv(base / "kpi_weighting" / "output" / "kpi_weights_full.csv")
    q1_mw = pd.read_csv(base / "Q1_discriminating_metrics" / "results" / "mann_whitney_plays_vs_rest.csv")
    q2_shap = pd.read_csv(base / "Q2_progression_prediction" / "results" / "shap_feature_importance.csv")
    q3_tiers = pd.read_csv(base / "Q3_signal_vs_noise" / "results" / "metric_tier_list.csv")

    return weights, q1_mw, q2_shap, q3_tiers


def build_consolidated_table(weights, q1_mw, q2_shap, q3_tiers):
    """Cross-reference all four analyses into one master table."""
    print("\n" + "=" * 70)
    print("STEP 1 — CONSOLIDATED CROSS-REFERENCE TABLE")
    print("=" * 70)

    # Start with Phase 1 weights (includes all KPIs)
    master = weights[["feature_name", "consensus_weight", "rank", "category", "direction"]].copy()
    master = master.rename(columns={
        "consensus_weight": "phase1_weight",
        "rank": "phase1_rank",
    })

    # Merge Q1 statistical tests
    q1_cols = q1_mw[["feature", "cohens_d", "p_value_raw", "p_value_fdr",
                       "significant_fdr_005", "plays_higher"]].copy()
    q1_cols = q1_cols.rename(columns={
        "feature": "feature_name",
        "cohens_d": "q1_cohens_d",
        "p_value_raw": "q1_p_raw",
        "p_value_fdr": "q1_p_fdr",
        "significant_fdr_005": "q1_significant",
        "plays_higher": "q1_plays_higher",
    })
    master = master.merge(q1_cols, on="feature_name", how="left")

    # Merge Q2 SHAP importance
    q2_shap_clean = q2_shap.copy()
    # Extract feature_name from 'feature' column (strip mean_ and GK_ prefix)
    q2_shap_clean["feature_name"] = q2_shap_clean["feature"].str.replace("mean_", "", regex=False)
    q2_shap_clean["q2_shap_rank"] = range(1, len(q2_shap_clean) + 1)
    q2_shap_clean = q2_shap_clean.rename(columns={"mean_abs_shap": "q2_shap_value"})
    master = master.merge(
        q2_shap_clean[["feature_name", "q2_shap_value", "q2_shap_rank"]],
        on="feature_name", how="left"
    )

    # Merge Q3 tier list
    q3_cols = q3_tiers[["feature", "coeff_variation", "icc_approx", "tier"]].copy()
    q3_cols = q3_cols.rename(columns={
        "feature": "feature_name",
        "coeff_variation": "q3_cv",
        "icc_approx": "q3_icc",
        "tier": "q3_tier",
    })
    master = master.merge(q3_cols, on="feature_name", how="left")

    # Compute composite scouting score per KPI
    # Normalize each dimension to [0,1] and weight them
    def safe_norm(series):
        mn, mx = series.min(), series.max()
        if mx - mn > 0:
            return (series - mn) / (mx - mn)
        return series * 0

    master["norm_weight"] = safe_norm(master["phase1_weight"].fillna(0))
    master["norm_effect"] = safe_norm(master["q1_cohens_d"].abs().fillna(0))
    master["norm_shap"] = safe_norm(master["q2_shap_value"].fillna(0))

    # For reliability, lower CV is better → invert
    cv_filled = master["q3_cv"].fillna(master["q3_cv"].max())
    master["norm_reliability"] = 1 - safe_norm(cv_filled.clip(upper=100))

    # Composite: 25% weight + 25% effect + 25% SHAP + 25% reliability
    master["composite_score"] = (
        0.25 * master["norm_weight"] +
        0.25 * master["norm_effect"] +
        0.25 * master["norm_shap"] +
        0.25 * master["norm_reliability"]
    )

    # Filter to performance KPIs only for ranking
    perf = master[master["category"] == "performance"].copy()
    perf["composite_rank"] = perf["composite_score"].rank(ascending=False).astype(int)
    master = master.merge(perf[["feature_name", "composite_rank"]], on="feature_name", how="left")

    master = master.sort_values("composite_score", ascending=False)

    # Display
    display_cols = ["feature_name", "phase1_rank", "q1_cohens_d", "q1_significant",
                    "q2_shap_rank", "q3_tier", "q3_cv", "composite_score", "composite_rank"]
    available = [c for c in display_cols if c in master.columns]
    print("\n  Performance KPIs ranked by composite scouting score:")
    perf_display = master[master["category"] == "performance"][available]
    print(perf_display.to_string(index=False))

    master.to_csv(OUTPUT / "consolidated_table.csv", index=False)
    print(f"\n  Saved: consolidated_table.csv ({len(master)} KPIs)")

    return master


# =========================================================================
# STEP 2 — Definitive scouting profile
# =========================================================================

def classify_metrics(master):
    """Classify KPIs into Core, Supporting, and Discarded."""
    print("\n" + "=" * 70)
    print("STEP 2 — DEFINITIVE SCOUTING PROFILE")
    print("=" * 70)

    perf = master[master["category"] == "performance"].copy()

    def classify(row):
        sig = row.get("q1_significant", False)
        tier = row.get("q3_tier", "Tier 3 (Noise)")
        composite = row.get("composite_score", 0)
        cv = row.get("q3_cv", 100)

        # Core: significant in Q1, Tier 1 in Q3, top composite
        if sig and "Tier 1" in str(tier):
            return "Core Predictor"
        # Core: very high composite even if borderline significance
        if composite > 0.55 and cv < 1.0 and sig:
            return "Core Predictor"
        # Supporting: Tier 1 or 2, moderate composite
        if ("Tier 1" in str(tier) or "Tier 2" in str(tier)) and composite > 0.3:
            return "Supporting Indicator"
        if sig and composite > 0.35:
            return "Supporting Indicator"
        # Discarded: everything else
        return "Discarded"

    perf["classification"] = perf.apply(classify, axis=1)

    for cls in ["Core Predictor", "Supporting Indicator", "Discarded"]:
        subset = perf[perf["classification"] == cls]
        print(f"\n  {cls}s ({len(subset)}):")
        for _, row in subset.iterrows():
            d = row.get("q1_cohens_d", 0)
            cv = row.get("q3_cv", 0)
            tier = row.get("q3_tier", "?")
            sig = "***" if row.get("q1_significant", False) else ""
            print(f"    {row['feature_name']:<50s}  d={d:+.3f}  CV={cv:.1f}  {tier}  {sig}")

    perf.to_csv(OUTPUT / "metric_classification.csv", index=False)
    return perf


# =========================================================================
# STEP 3 — Threshold analysis
# =========================================================================

def threshold_analysis(df_model, classified):
    """Compute actionable thresholds for core predictors."""
    print("\n" + "=" * 70)
    print("STEP 3 — THRESHOLD ANALYSIS")
    print("=" * 70)

    core = classified[classified["classification"] == "Core Predictor"]
    plays = df_model[df_model["status"] == "PLAYS"]
    rest = df_model[df_model["status"] != "PLAYS"]

    thresholds = []
    for _, row in core.iterrows():
        fname = row["feature_name"]
        col = f"mean_{fname}"
        if col not in df_model.columns:
            continue

        p_vals = plays[col].dropna()
        r_vals = rest[col].dropna()
        all_vals = df_model[col].dropna()

        # Percentile-based thresholds
        p25_plays = p_vals.quantile(0.25)
        p50_plays = p_vals.median()
        p75_plays = p_vals.quantile(0.75)
        p50_rest = r_vals.median()

        # Optimal threshold via Youden's J statistic
        y_true = (df_model[col].notna() & (df_model["status"] == "PLAYS")).astype(int)
        y_true = y_true[df_model[col].notna()]
        x_vals = df_model.loc[df_model[col].notna(), col]

        # Simple threshold sweep
        best_j = -1
        best_thresh = p50_plays
        for pctile in np.arange(0.1, 0.9, 0.05):
            thresh = all_vals.quantile(pctile)
            pred = (x_vals >= thresh).astype(int)
            tp = ((pred == 1) & (y_true == 1)).sum()
            fp = ((pred == 1) & (y_true == 0)).sum()
            tn = ((pred == 0) & (y_true == 0)).sum()
            fn = ((pred == 0) & (y_true == 1)).sum()
            if (tp + fn) > 0 and (fp + tn) > 0:
                sens = tp / (tp + fn)
                spec = tn / (fp + tn)
                j = sens + spec - 1
                if j > best_j:
                    best_j = j
                    best_thresh = thresh

        # What percentile of the overall population is the threshold?
        thresh_pctile = (all_vals < best_thresh).mean()

        thresholds.append({
            "feature": fname,
            "plays_median": p50_plays,
            "plays_p25": p25_plays,
            "plays_p75": p75_plays,
            "rest_median": p50_rest,
            "optimal_threshold": best_thresh,
            "threshold_percentile": thresh_pctile,
            "youden_j": best_j,
            "direction": row.get("direction", "higher"),
        })

        print(f"\n  {fname}:")
        print(f"    PLAYS median: {p50_plays:.4f}  (IQR: {p25_plays:.4f} - {p75_plays:.4f})")
        print(f"    REST median:  {p50_rest:.4f}")
        print(f"    Optimal threshold: {best_thresh:.4f} ({thresh_pctile:.0%} percentile)")
        print(f"    Youden's J: {best_j:.3f}")

    thresh_df = pd.DataFrame(thresholds)
    thresh_df.to_csv(OUTPUT / "thresholds.csv", index=False)
    print(f"\n  Saved: thresholds.csv")

    # Plot threshold distributions
    n = len(thresholds)
    if n == 0:
        return thresh_df

    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
    axes = axes.flatten() if n > ncols else ([axes] if n == 1 else axes.flatten())

    for i, t in enumerate(thresholds):
        ax = axes[i]
        col = f"mean_{t['feature']}"
        for status in ["PLAYS", "REST"]:
            if status == "PLAYS":
                data = plays[col].dropna()
                color = COLORS["PLAYS"]
            else:
                data = rest[col].dropna()
                color = "#95a5a6"
            ax.hist(data, bins=25, alpha=0.6, color=color, label=status, density=True)

        ax.axvline(t["optimal_threshold"], color="red", linestyle="--", linewidth=2,
                   label=f"Threshold: {t['optimal_threshold']:.3f}")
        ax.axvline(t["plays_median"], color=COLORS["PLAYS"], linestyle="-", linewidth=1.5,
                   label=f"PLAYS median: {t['plays_median']:.3f}")

        name = t["feature"].replace("GK_", "")[:30]
        ax.set_title(f"{name}\n(J={t['youden_j']:.2f}, threshold at {t['threshold_percentile']:.0%}ile)",
                     fontsize=10)
        ax.legend(fontsize=7)
        ax.set_xlabel("Value")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Core Predictor Thresholds: PLAYS vs REST Distributions", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT / "threshold_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: threshold_distributions.png")

    return thresh_df


# =========================================================================
# STEP 4 — Validation & confidence
# =========================================================================

def validate_core_model(df_model, classified, feature_cols):
    """Train a model using ONLY core + supporting predictors and validate."""
    print("\n" + "=" * 70)
    print("STEP 4 — VALIDATION WITH CORE + SUPPORTING PREDICTORS")
    print("=" * 70)

    core_support = classified[classified["classification"].isin(
        ["Core Predictor", "Supporting Indicator"]
    )]
    core_features = [f"mean_{f}" for f in core_support["feature_name"]
                     if f"mean_{f}" in df_model.columns]
    # Also include context features
    model_features = core_features + [c for c in META_COLS if c in df_model.columns]

    print(f"\n  Core + Supporting features: {len(core_features)} performance + {len(META_COLS)} context")
    for f in core_features:
        print(f"    - {f.replace('mean_', '')}")

    X = df_model[model_features].copy().fillna(df_model[model_features].median())
    y = (df_model["status"] == "PLAYS").astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    n_pos = y.sum()
    n_neg = len(y) - n_pos

    # Compare: full model vs core-only model
    print("\n  --- Full Model (all 30 features) ---")
    X_full = df_model[feature_cols].copy().fillna(df_model[feature_cols].median())
    xgb_full = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        scale_pos_weight=n_neg / max(n_pos, 1),
        random_state=42, eval_metric="logloss", verbosity=0,
    )
    y_pred_full = cross_val_predict(xgb_full, X_full, y, cv=cv)
    y_proba_full = cross_val_predict(xgb_full, X_full, y, cv=cv, method="predict_proba")[:, 1]
    auc_full = roc_auc_score(y, y_proba_full)
    f1_full = f1_score(y, y_pred_full)
    print(f"    AUC-ROC: {auc_full:.3f}  F1: {f1_full:.3f}")

    print(f"\n  --- Core+Supporting Model ({len(model_features)} features) ---")
    xgb_core = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        scale_pos_weight=n_neg / max(n_pos, 1),
        random_state=42, eval_metric="logloss", verbosity=0,
    )
    y_pred_core = cross_val_predict(xgb_core, X, y, cv=cv)
    y_proba_core = cross_val_predict(xgb_core, X, y, cv=cv, method="predict_proba")[:, 1]
    auc_core = roc_auc_score(y, y_proba_core)
    f1_core = f1_score(y, y_pred_core)
    print(f"    AUC-ROC: {auc_core:.3f}  F1: {f1_core:.3f}")
    print(f"\n    Performance retained: {auc_core/auc_full:.1%} of full model AUC")

    # ROC curves comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    fpr_full, tpr_full, _ = roc_curve(y, y_proba_full)
    fpr_core, tpr_core, _ = roc_curve(y, y_proba_core)
    ax.plot(fpr_full, tpr_full, label=f"Full model (AUC={auc_full:.3f})", linewidth=2)
    ax.plot(fpr_core, tpr_core, label=f"Core+Support (AUC={auc_core:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Full Model vs Core+Supporting Predictors Only")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT / "roc_full_vs_core.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: roc_full_vs_core.png")

    # Confusion matrix for core model
    fig, ax = plt.subplots(figsize=(7, 5))
    cm = confusion_matrix(y, y_pred_core)
    ConfusionMatrixDisplay(cm, display_labels=["REST", "PLAYS"]).plot(ax=ax, cmap="Blues")
    ax.set_title(f"Core+Supporting Model\nAUC={auc_core:.3f}, F1={f1_core:.3f}")
    plt.tight_layout()
    plt.savefig(OUTPUT / "confusion_matrix_core.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: confusion_matrix_core.png")

    return {
        "auc_full": auc_full, "f1_full": f1_full,
        "auc_core": auc_core, "f1_core": f1_core,
        "y_proba_core": y_proba_core, "y_pred_core": y_pred_core,
        "model_features": model_features,
    }


def case_studies(df_model, validation, classified):
    """Show real examples from the data."""
    print("\n" + "=" * 70)
    print("CASE STUDIES")
    print("=" * 70)

    y = (df_model["status"] == "PLAYS").astype(int)
    proba = validation["y_proba_core"]

    df_cases = df_model[["playerId", "name", "age", "status", "origin_team",
                          "origin_comp", "origin_median"]].copy()
    df_cases["predicted_proba"] = proba
    df_cases["predicted_plays"] = (proba > 0.5).astype(int)
    df_cases["actual_plays"] = y.values

    # True positives: correctly identified PLAYS keepers
    tp = df_cases[(df_cases["actual_plays"] == 1) & (df_cases["predicted_proba"] > 0.3)]
    tp = tp.sort_values("predicted_proba", ascending=False)
    print(f"\n  TRUE POSITIVES — PLAYS keepers the model correctly flagged (top 10):")
    display_cols = ["name", "age", "origin_comp", "origin_median", "predicted_proba", "status"]
    print(tp[display_cols].head(10).to_string(index=False))

    # False negatives: PLAYS keepers the model missed
    fn = df_cases[(df_cases["actual_plays"] == 1) & (df_cases["predicted_proba"] < 0.15)]
    fn = fn.sort_values("predicted_proba")
    print(f"\n  FALSE NEGATIVES — PLAYS keepers the model MISSED ({len(fn)}):")
    if len(fn) > 0:
        print(fn[display_cols].head(10).to_string(index=False))

    # Top predictions among STAYED keepers (potential undiscovered talent)
    stayed_high = df_cases[
        (df_cases["status"] == "STAYED") & (df_cases["predicted_proba"] > 0.3)
    ].sort_values("predicted_proba", ascending=False)
    print(f"\n  HIDDEN GEMS — STAYED keepers with high progression probability ({len(stayed_high)}):")
    if len(stayed_high) > 0:
        print(stayed_high[display_cols].head(10).to_string(index=False))

    df_cases.to_csv(OUTPUT / "case_studies.csv", index=False)
    print(f"\n  Saved: case_studies.csv")

    return df_cases


# =========================================================================
# STEP 5 — Visual scorecards & radar charts
# =========================================================================

def radar_chart(df_model, classified, thresh_df):
    """Radar chart comparing PLAYS vs REST profiles on core metrics."""
    print("\n" + "=" * 70)
    print("STEP 5 — VISUAL SCORECARDS")
    print("=" * 70)

    core = classified[classified["classification"] == "Core Predictor"]
    core_features = [f"mean_{f}" for f in core["feature_name"] if f"mean_{f}" in df_model.columns]

    if len(core_features) < 3:
        print("  Not enough core features for radar chart")
        return

    plays = df_model[df_model["status"] == "PLAYS"]
    rest = df_model[df_model["status"] != "PLAYS"]

    # Compute percentile scores for each group
    labels = []
    plays_pctiles = []
    rest_pctiles = []

    for col in core_features:
        all_vals = df_model[col]
        p_pctile = (all_vals < plays[col].median()).mean()
        r_pctile = (all_vals < rest[col].median()).mean()
        labels.append(col.replace("mean_", "").replace("GK_", "")[:20])
        plays_pctiles.append(p_pctile)
        rest_pctiles.append(r_pctile)

    # Close the polygon
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    plays_pctiles += plays_pctiles[:1]
    rest_pctiles += rest_pctiles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.plot(angles, plays_pctiles, "o-", linewidth=2.5, color=COLORS["PLAYS"],
            label="PLAYS (progressors)", markersize=8)
    ax.fill(angles, plays_pctiles, alpha=0.15, color=COLORS["PLAYS"])
    ax.plot(angles, rest_pctiles, "o-", linewidth=2.5, color="#95a5a6",
            label="REST (non-progressors)", markersize=8)
    ax.fill(angles, rest_pctiles, alpha=0.1, color="#95a5a6")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75])
    ax.set_yticklabels(["25th %ile", "50th %ile", "75th %ile"], fontsize=8)
    ax.set_title("Goalkeeper Progression Profile\n(Core Predictors — Percentile Ranks)",
                 fontsize=14, pad=30)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUT / "radar_progression_profile.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: radar_progression_profile.png")


def scorecard_heatmap(df_model, classified, thresh_df):
    """Heatmap scorecard: how many core thresholds each status group passes."""
    core = classified[classified["classification"] == "Core Predictor"]

    if thresh_df is None or len(thresh_df) == 0:
        return

    # For each keeper, count how many core thresholds they exceed
    score_cols = []
    for _, t in thresh_df.iterrows():
        col = f"mean_{t['feature']}"
        if col in df_model.columns:
            threshold = t["optimal_threshold"]
            pass_col = f"pass_{t['feature'][:20]}"
            df_model[pass_col] = (df_model[col] >= threshold).astype(int)
            score_cols.append(pass_col)

    if not score_cols:
        return

    df_model["n_thresholds_passed"] = df_model[score_cols].sum(axis=1)
    n_core = len(score_cols)

    # Summary by status
    print(f"\n  Thresholds passed (out of {n_core} core metrics):")
    for status in STATUS_ORDER:
        subset = df_model[df_model["status"] == status]
        if len(subset) > 0:
            mean_passed = subset["n_thresholds_passed"].mean()
            median_passed = subset["n_thresholds_passed"].median()
            print(f"    {status:10s}  mean={mean_passed:.1f}  median={median_passed:.0f}  n={len(subset)}")

    # Heatmap: pass rates per status per metric
    pass_rates = []
    for status in STATUS_ORDER:
        subset = df_model[df_model["status"] == status]
        rates = {}
        for col in score_cols:
            rates[col.replace("pass_", "")] = subset[col].mean()
        pass_rates.append(rates)

    rate_df = pd.DataFrame(pass_rates, index=STATUS_ORDER)

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(rate_df, annot=True, fmt=".0%", cmap="RdYlGn", ax=ax,
                vmin=0, vmax=1, linewidths=1)
    ax.set_title("Percentage of Keepers Exceeding Core Thresholds by Status", fontsize=13)
    ax.set_ylabel("Career Status")
    plt.tight_layout()
    plt.savefig(OUTPUT / "scorecard_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: scorecard_heatmap.png")

    # Distribution of thresholds passed
    fig, ax = plt.subplots(figsize=(10, 6))
    for status in STATUS_ORDER:
        subset = df_model[df_model["status"] == status]
        counts = subset["n_thresholds_passed"].value_counts().sort_index()
        pcts = counts / len(subset) * 100
        ax.plot(pcts.index, pcts.values, "o-", label=f"{status} (n={len(subset)})",
                linewidth=2, markersize=6)
    ax.set_xlabel(f"Number of Core Thresholds Passed (out of {n_core})")
    ax.set_ylabel("% of Keepers")
    ax.set_title("How Many Core Thresholds Do Keepers Pass?")
    ax.legend()
    ax.set_xticks(range(n_core + 1))
    plt.tight_layout()
    plt.savefig(OUTPUT / "thresholds_passed_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: thresholds_passed_distribution.png")


def plot_consolidated_overview(master):
    """Visual overview of the consolidated table."""
    perf = master[master["category"] == "performance"].sort_values("composite_score", ascending=True).copy()

    fig, ax = plt.subplots(figsize=(14, 10))

    # Color by classification approach
    colors = []
    for _, row in perf.iterrows():
        sig = row.get("q1_significant", False)
        tier = str(row.get("q3_tier", ""))
        if sig and "Tier 1" in tier:
            colors.append(COLORS["core"])
        elif "Tier 1" in tier or "Tier 2" in tier:
            colors.append(COLORS["supporting"])
        else:
            colors.append(COLORS["discarded"])

    ax.barh(range(len(perf)), perf["composite_score"], color=colors)
    ax.set_yticks(range(len(perf)))
    labels = [f.replace("GK_", "")[:35] for f in perf["feature_name"]]
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Composite Scouting Score (weight + effect + SHAP + reliability)")
    ax.set_title("All Performance KPIs: Composite Scouting Score\n"
                 "Green = Core | Orange = Supporting | Red = Discarded", fontsize=13)

    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor=COLORS["core"], label="Core Predictor"),
        Patch(facecolor=COLORS["supporting"], label="Supporting Indicator"),
        Patch(facecolor=COLORS["discarded"], label="Discarded"),
    ]
    ax.legend(handles=legend, loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT / "consolidated_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: consolidated_overview.png")


# =========================================================================
# MAIN
# =========================================================================

def main():
    print("=" * 70)
    print("CENTRAL QUESTION — DEFINITIVE ANSWER")
    print("Which measurable performances of a goalkeeper in a lower league")
    print("predict whether he will succeed at a higher level?")
    print("=" * 70)

    # Load data
    score_defs, _, _ = load_definitions()
    cache = get_cache_path()
    dataset, df = load_and_aggregate_data(score_defs, cache_path=cache)
    df_model, feature_cols, gk_cols, general_cols = select_features(df, score_defs)

    # Step 1: Synthesize
    weights, q1_mw, q2_shap, q3_tiers = load_prior_results()
    master = build_consolidated_table(weights, q1_mw, q2_shap, q3_tiers)
    plot_consolidated_overview(master)

    # Step 2: Classify
    classified = classify_metrics(master)

    # Step 3: Thresholds
    thresh_df = threshold_analysis(df_model, classified)

    # Step 4: Validate
    validation = validate_core_model(df_model, classified, feature_cols)
    cases = case_studies(df_model, validation, classified)

    # Step 5: Visuals
    radar_chart(df_model, classified, thresh_df)
    scorecard_heatmap(df_model, classified, thresh_df)

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)

    core = classified[classified["classification"] == "Core Predictor"]
    print(f"\n  {len(core)} Core Predictors identified:")
    for _, row in core.iterrows():
        print(f"    - {row['feature_name']}")

    print(f"\n  Core+Supporting model retains {validation['auc_core']/validation['auc_full']:.0%} "
          f"of full model performance (AUC {validation['auc_core']:.3f} vs {validation['auc_full']:.3f})")

    print(f"\n  All outputs saved to: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
