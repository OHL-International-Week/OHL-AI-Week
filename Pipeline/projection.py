#!/usr/bin/env python3
"""Project lower-league KPIs to higher-league performance.

Uses 130 keepers who already transferred up to learn how each KPI changes.

Approach (3 layers, each validated against the next):

  Layer 1 — RETENTION RATE: for each KPI, what fraction of the lower-league
            value do keepers typically maintain at the higher level?
            Simple, robust, no overfitting. (e.g., "keepers retain 85% of
            their passing volume when moving up")

  Layer 2 — INDIVIDUAL REGRESSION: current_KPI = a * origin_KPI + b
            Two parameters per KPI. Captures that some keepers improve
            (high origin → even higher current) while others regress.

  Layer 3 — LEAGUE-GAP ADJUSTMENT: add the gap in league strength as a
            predictor. Bigger step up → bigger adjustment.

We pick the best layer per KPI based on cross-validated R².

Output:
  - projected_profiles.csv      (all STAYED keepers)
  - projection_accuracy.csv     (per-KPI R² and retention rates)
  - projection_accuracy.png     (visualization)
  - projection_radar.png        (top 5 targets)
  - projection_scatter.png      (origin vs projected)
  - kpi_retention_rates.csv     (how each KPI changes when stepping up)

Run: python Pipeline/projection.py
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
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SKPipeline
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="colorblind")

OUTPUT = Path(__file__).resolve().parent / "output"
GK_DATA = PROJECT_ROOT / "GK_Data"

INTERPRETATIONS = {
    'SUCCESSFUL_PASSES_BY_FOOT_RIGHT': 'Right-foot passing',
    'SUCCESSFUL_PASSES_BY_FOOT_LEFT': 'Left-foot passing',
    'UNSUCCESSFUL_PASSES_BY_FOOT_LEFT': 'Weak foot attempts',
    'BYPASSED_DEFENDERS_AT_PHASE_SECOND_BALL': '2nd ball defending',
    'UNSUCCESSFUL_PASSES_BY_FOOT_RIGHT': 'Right-foot attempts',
    'BYPASSED_OPPONENTS_NUMBER_AT_PHASE_ATTACKING_TRANSITION': 'Counter-attacking',
    'SUCCESSFUL_PASSES_BY_ACTION_DIAGONAL_PASS': 'Diagonal passes',
    'SUCCESSFUL_PASSES': 'Total passes',
    'DISTANCE_TO_GOAL_COVERED_DRIBBLE': 'Dribbling (off-line)',
    'SECOND_BALL_WIN': 'Second ball wins',
    'SUCCESSFUL_PASSES_FROM_PITCH_POSITION_FIRST_THIRD': 'Passes from own third',
    'BYPASSED_OPPONENTS_BY_ACTION_LOW_PASS': 'Low pass bypasses',
}


def load_data():
    origin = pd.read_parquet(OUTPUT / "keeper_all_kpis.parquet")
    current = pd.read_parquet(OUTPUT / "keeper_current_kpis.parquet")
    dataset = pd.read_csv(GK_DATA / "gk_dataset_final.csv")

    merged = origin.merge(
        current[["playerId"] + [c for c in current.columns if c.startswith("cur_")]],
        on="playerId", how="inner"
    )
    merged = merged.merge(
        dataset[["playerId", "current_median"]].drop_duplicates(),
        on="playerId", how="left"
    )
    print(f"Keepers with before + after data: {len(merged)}")
    return merged, origin, dataset


def analyze_kpi_changes(merged):
    """For each KPI, compute how it changes when keepers move up."""
    print("\nAnalyzing KPI changes (lower league → higher league)...")

    selected = pd.read_csv(OUTPUT / "selected_features.csv")
    top_kpis = selected.head(30)["feature_name"].tolist()

    results = []
    models = {}

    for kpi in top_kpis:
        o_col = f"mean_{kpi}"
        c_col = f"cur_{kpi}"
        if o_col not in merged.columns or c_col not in merged.columns:
            continue

        valid = merged[[o_col, c_col, "origin_median", "current_median"]].dropna()
        if len(valid) < 20:
            continue

        origin_vals = valid[o_col].values
        current_vals = valid[c_col].values

        # Layer 1: Retention rate (median ratio, robust to outliers)
        ratios = current_vals / np.where(np.abs(origin_vals) > 1e-6, origin_vals, np.nan)
        ratios = ratios[np.isfinite(ratios) & (np.abs(ratios) < 10)]  # clip extreme
        retention = np.median(ratios) if len(ratios) > 10 else 1.0

        # Layer 2: Simple linear regression (2 params)
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        X_simple = origin_vals.reshape(-1, 1)
        cv_r2_simple = cross_val_score(lr, X_simple, current_vals, cv=5, scoring="r2")
        lr.fit(X_simple, current_vals)
        slope = lr.coef_[0]
        intercept = lr.intercept_

        # Layer 3: Add league gap
        league_gap = (valid["current_median"] - valid["origin_median"]).values.reshape(-1, 1)
        X_gap = np.column_stack([origin_vals, league_gap])
        lr_gap = LinearRegression()
        cv_r2_gap = cross_val_score(lr_gap, X_gap, current_vals, cv=5, scoring="r2")
        lr_gap.fit(X_gap, current_vals)

        # Pick best layer
        r2_simple = cv_r2_simple.mean()
        r2_gap = cv_r2_gap.mean()

        if r2_gap > r2_simple and r2_gap > 0:
            best_r2 = r2_gap
            best_model = "league-adjusted"
            model_obj = lr_gap
            n_params = 3
        elif r2_simple > 0:
            best_r2 = r2_simple
            best_model = "linear"
            model_obj = lr
            n_params = 2
        else:
            best_r2 = 0
            best_model = "retention"
            model_obj = None
            n_params = 1

        # Paired t-test: does the KPI significantly change?
        t_stat, p_val = stats.ttest_rel(origin_vals, current_vals)
        direction = "increases" if current_vals.mean() > origin_vals.mean() else "decreases"

        models[kpi] = {
            "model": model_obj,
            "best_method": best_model,
            "retention": retention,
            "slope": slope,
            "intercept": intercept,
        }

        results.append({
            "kpi": kpi,
            "football_name": INTERPRETATIONS.get(kpi, kpi[:40]),
            "n_keepers": len(valid),
            "origin_mean": origin_vals.mean(),
            "current_mean": current_vals.mean(),
            "retention_rate": retention,
            "change_direction": direction,
            "change_pct": (current_vals.mean() - origin_vals.mean()) / max(abs(origin_vals.mean()), 1e-6) * 100,
            "paired_t_pval": p_val,
            "significant": p_val < 0.05,
            "r2_simple": r2_simple,
            "r2_league_adj": r2_gap,
            "best_r2": best_r2,
            "best_method": best_model,
        })

        sig = "***" if p_val < 0.05 else "   "
        print(f"  {kpi[:45]:<47s}  retention={retention:.2f}  R²={best_r2:.3f} ({best_model})  {sig}")

    results_df = pd.DataFrame(results).sort_values("best_r2", ascending=False)
    results_df.to_csv(OUTPUT / "projection_accuracy.csv", index=False)
    results_df.to_csv(OUTPUT / "kpi_retention_rates.csv", index=False)

    n_sig = results_df["significant"].sum()
    print(f"\n  {n_sig}/{len(results_df)} KPIs significantly change when moving up")
    print(f"  Median retention rate: {results_df['retention_rate'].median():.2f}")

    return models, results_df


def project_stayed_keepers(origin_df, dataset, models, results_df):
    """Apply projection to STAYED keepers."""
    print("\nProjecting STAYED keepers...")

    stayed = origin_df[origin_df["status"] == "STAYED"].copy()
    plays_median = dataset[dataset["status"] == "PLAYS"]["current_median"].median()
    print(f"  Target league: median={plays_median:.3f}")

    projected_rows = []
    for _, keeper in stayed.iterrows():
        proj = {
            "playerId": keeper["playerId"],
            "name": keeper["name"],
            "age": keeper["age"],
            "origin_team": keeper.get("origin_team", ""),
            "origin_comp": keeper.get("origin_comp", ""),
            "origin_median": keeper["origin_median"],
            "target_median": plays_median,
        }

        for kpi, info in models.items():
            o_col = f"mean_{kpi}"
            origin_val = keeper.get(o_col, np.nan)
            proj[f"origin_{kpi}"] = origin_val

            if pd.isna(origin_val):
                proj[f"projected_{kpi}"] = np.nan
                continue

            if info["best_method"] == "retention":
                proj[f"projected_{kpi}"] = origin_val * info["retention"]
            elif info["best_method"] == "linear" and info["model"] is not None:
                proj[f"projected_{kpi}"] = info["slope"] * origin_val + info["intercept"]
            elif info["best_method"] == "league-adjusted" and info["model"] is not None:
                gap = plays_median - keeper.get("origin_median", 0.5)
                proj[f"projected_{kpi}"] = info["model"].predict(
                    np.array([[origin_val, gap]]))[0]
            else:
                proj[f"projected_{kpi}"] = origin_val * info["retention"]

        projected_rows.append(proj)

    proj_df = pd.DataFrame(projected_rows)

    # Add scouting score
    scores = pd.read_csv(OUTPUT / "scouting_scores.csv")
    proj_df = proj_df.merge(scores[["playerId", "scouting_score"]], on="playerId", how="left")
    proj_df = proj_df.sort_values("scouting_score", ascending=False)
    proj_df.to_csv(OUTPUT / "projected_profiles.csv", index=False)
    print(f"  Saved: {len(proj_df)} projected profiles")
    return proj_df


def plot_results(merged, proj_df, models, results_df):
    """Generate visualizations."""
    print("\nGenerating visualizations...")

    # 1. KPI retention rates — how each KPI changes
    plot_df = results_df.sort_values("retention_rate").copy()
    fig, ax = plt.subplots(figsize=(14, 9))
    colors = ["#2ecc71" if r >= 1.0 else "#e74c3c" if r < 0.8 else "#f39c12"
              for r in plot_df["retention_rate"]]
    bars = ax.barh(range(len(plot_df)), plot_df["retention_rate"], color=colors,
                   edgecolor="white", linewidth=0.5)
    ax.axvline(x=1.0, color="black", linestyle="--", alpha=0.5, label="No change (1.0)")
    ax.set_yticks(range(len(plot_df)))
    labels = []
    for _, r in plot_df.iterrows():
        sig = " ***" if r["significant"] else ""
        labels.append(f"{r['football_name'][:30]}{sig}")
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Retention Rate (current / origin)")
    ax.set_title("How Do KPIs Change When Moving to a Higher League?\n"
                 "Green = increases | Orange = slight decrease | Red = drops significantly\n"
                 "*** = statistically significant change (paired t-test, p<0.05)")
    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor="#2ecc71", label="Increases (>=1.0)"),
        Patch(facecolor="#f39c12", label="Slight decrease (0.8-1.0)"),
        Patch(facecolor="#e74c3c", label="Drops (<0.8)"),
    ]
    ax.legend(handles=legend, loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT / "projection_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Radar: top 5 targets — origin vs projected
    top5 = proj_df.head(5)
    radar_kpis = results_df.head(8)["kpi"].tolist()
    radar_kpis = [k for k in radar_kpis if f"projected_{k}" in proj_df.columns][:8]

    if len(radar_kpis) >= 4:
        n_keepers = min(5, len(top5))
        fig, axes = plt.subplots(1, n_keepers, figsize=(4 * n_keepers, 4.5),
                                 subplot_kw=dict(polar=True))
        if n_keepers == 1:
            axes = [axes]

        # Compute percentiles from all STAYED keepers for normalization
        for idx, (_, keeper) in enumerate(top5.head(n_keepers).iterrows()):
            ax = axes[idx]
            n = len(radar_kpis)
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist() + [0]

            origin_vals = []
            proj_vals = []
            for kpi in radar_kpis:
                ov = keeper.get(f"origin_{kpi}", 0) or 0
                pv = keeper.get(f"projected_{kpi}", 0) or 0

                # Normalize against all STAYED keepers
                all_vals = proj_df[f"origin_{kpi}"].dropna()
                if len(all_vals) > 0 and all_vals.max() > all_vals.min():
                    ov_pct = (all_vals < ov).mean()
                    pv_pct = (all_vals < pv).mean()
                else:
                    ov_pct = 0.5
                    pv_pct = 0.5
                origin_vals.append(ov_pct)
                proj_vals.append(pv_pct)

            origin_vals += origin_vals[:1]
            proj_vals += proj_vals[:1]

            ax.plot(angles, origin_vals, "o-", linewidth=1.5, color="#3498db",
                    label="Lower league", markersize=4)
            ax.fill(angles, origin_vals, alpha=0.1, color="#3498db")
            ax.plot(angles, proj_vals, "o-", linewidth=2, color="#2ecc71",
                    label="Projected", markersize=4)
            ax.fill(angles, proj_vals, alpha=0.15, color="#2ecc71")

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([INTERPRETATIONS.get(k, k[:12]) for k in radar_kpis], fontsize=5)
            ax.set_ylim(0, 1.1)
            ax.set_yticks([0.25, 0.5, 0.75])
            ax.set_yticklabels(["25th", "50th", "75th"], fontsize=5)
            name = str(keeper.get("name", ""))[:18]
            score = keeper.get("scouting_score", 0)
            ax.set_title(f"{name}\nscore={score:.0f}", fontsize=9, pad=15)
            if idx == 0:
                ax.legend(fontsize=6, loc="upper right", bbox_to_anchor=(1.4, 1.15))

        fig.suptitle("Projected Higher-League Profiles — Top Scouting Targets",
                     fontsize=13, y=1.05)
        plt.tight_layout()
        plt.savefig(OUTPUT / "projection_radar.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 3. Scatter: origin vs current for best-predicted KPI
    best_kpi = results_df.iloc[0]["kpi"]
    o_col = f"mean_{best_kpi}"
    c_col = f"cur_{best_kpi}"
    if o_col in merged.columns and c_col in merged.columns:
        valid = merged[[o_col, c_col]].dropna()
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.scatter(valid[o_col], valid[c_col], c="#2196F3", s=40, alpha=0.6, edgecolors="white")
        mn = min(valid[o_col].min(), valid[c_col].min())
        mx = max(valid[o_col].max(), valid[c_col].max())
        ax.plot([mn, mx], [mn, mx], "k--", alpha=0.3, label="No change")

        # Add regression line
        z = np.polyfit(valid[o_col], valid[c_col], 1)
        p = np.poly1d(z)
        x_line = np.linspace(mn, mx, 100)
        ax.plot(x_line, p(x_line), "r-", alpha=0.5, linewidth=2,
                label=f"Trend (slope={z[0]:.2f})")

        r = np.corrcoef(valid[o_col], valid[c_col])[0, 1]
        fname = INTERPRETATIONS.get(best_kpi, best_kpi[:35])
        ax.set_xlabel(f"Lower league: {fname}")
        ax.set_ylabel(f"Higher league: {fname}")
        ax.set_title(f"Does Lower-League Performance Carry Over?\n"
                     f"{fname} (r={r:.2f}, n={len(valid)})")
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT / "projection_scatter.png", dpi=150, bbox_inches="tight")
        plt.close()

    print("  Saved: projection_accuracy.png, projection_radar.png, projection_scatter.png")


def main():
    print("=" * 70)
    print("  PROJECTION: Lower League → Higher League Performance")
    print("=" * 70)

    merged, origin_df, dataset = load_data()
    models, results_df = analyze_kpi_changes(merged)

    if len(models) == 0:
        print("No projectable KPIs found.")
        return

    proj_df = project_stayed_keepers(origin_df, dataset, models, results_df)
    plot_results(merged, proj_df, models, results_df)

    print(f"\n{'=' * 70}")
    print("  PROJECTION COMPLETE")
    print(f"{'=' * 70}")
    n_sig = results_df["significant"].sum()
    print(f"  KPIs analyzed: {len(results_df)}")
    print(f"  Significantly change: {n_sig}")
    print(f"  Median retention: {results_df['retention_rate'].median():.2f}")
    print(f"  STAYED keepers projected: {len(proj_df)}")


if __name__ == "__main__":
    main()
