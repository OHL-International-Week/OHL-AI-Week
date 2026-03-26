#!/usr/bin/env python3
"""Generate additional demo assets that the pipeline doesn't produce.

- Exploratory box plots of key KPIs by status
- Full model metrics (confusion matrix, precision, recall, F1)
- Correlation heatmap of top features
- Case study: individual keeper profile

Run after run_pipeline.py.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="colorblind")

OUTPUT = Path(__file__).resolve().parent / "output"
GK_DATA = PROJECT_ROOT / "GK_Data"

COLORS = {"PLAYS": "#2ecc71", "BENCH": "#f39c12", "STAYED": "#3498db", "DROPPED": "#e74c3c"}
STATUS_ORDER = ["PLAYS", "BENCH", "STAYED", "DROPPED"]


def load_data():
    df = pd.read_parquet(OUTPUT / "keeper_all_kpis.parquet")
    selected = pd.read_csv(OUTPUT / "selected_features.csv")
    return df, selected


def generate_exploration_plots(df, selected):
    """Box plots + violin plots of top KPIs by status group."""
    print("Generating exploratory visualizations...")

    top8 = selected.head(8)
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    for idx, (_, row) in enumerate(top8.iterrows()):
        ax = axes[idx // 4][idx % 4]
        col = f"mean_{row['feature_name']}"
        if col not in df.columns:
            continue

        plot_data = []
        for status in STATUS_ORDER:
            vals = df[df["status"] == status][col].dropna()
            for v in vals:
                plot_data.append({"Status": status, "Value": v})
        plot_df = pd.DataFrame(plot_data)

        sns.boxplot(data=plot_df, x="Status", y="Value", order=STATUS_ORDER,
                    palette=COLORS, ax=ax, fliersize=2)
        name = row["feature_name"][:30]
        arrow = "+" if row["direction"] == "higher" else "-"
        ax.set_title(f"({arrow}) {name}", fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.suptitle("Top 8 Discriminating KPIs by Career Status\n"
                 "(+) = higher for progressors | (-) = lower for progressors",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT / "exploration_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Violin plots for top 4 (more detail)
    top4 = selected.head(4)
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    for idx, (_, row) in enumerate(top4.iterrows()):
        ax = axes[idx]
        col = f"mean_{row['feature_name']}"
        if col not in df.columns:
            continue

        plot_data = []
        for status in STATUS_ORDER:
            vals = df[df["status"] == status][col].dropna()
            for v in vals:
                plot_data.append({"Status": status, "Value": v})
        plot_df = pd.DataFrame(plot_data)

        sns.violinplot(data=plot_df, x="Status", y="Value", order=STATUS_ORDER,
                       palette=COLORS, ax=ax, inner="box", cut=0)
        name = row["feature_name"][:35]
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("")

    fig.suptitle("Distribution of Top 4 KPIs — How Groups Overlap",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT / "exploration_violins.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: exploration_boxplots.png, exploration_violins.png")


def generate_correlation_heatmap(df, selected):
    """Correlation heatmap of selected features."""
    print("Generating correlation heatmap...")
    top20 = selected.head(20)
    cols = [f"mean_{n}" for n in top20["feature_name"] if f"mean_{n}" in df.columns]
    corr = df[cols].corr()
    corr.index = [c.replace("mean_", "")[:30] for c in corr.index]
    corr.columns = [c.replace("mean_", "")[:30] for c in corr.columns]

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, ax=ax,
                annot=True, fmt=".2f", linewidths=0.5, square=True,
                vmin=-1, vmax=1, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Between Top 20 Selected KPIs\n"
                 "(multicollinearity check — high correlation = redundant signal)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT / "exploration_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: exploration_correlation.png")


def generate_full_model_metrics(df, selected):
    """Full model evaluation: confusion matrix, precision, recall, F1, AUC."""
    print("Generating full model metrics...")

    optimal_n = int((OUTPUT / "optimal_n_features.txt").read_text().strip())
    top = selected.head(optimal_n)
    feature_cols = [f"mean_{n}" for n in top["feature_name"] if f"mean_{n}" in df.columns]
    # Add non-mean features if present
    for f in top["feature_name"]:
        if f in df.columns and f not in feature_cols:
            feature_cols.append(f)

    X = df[feature_cols].copy().fillna(df[feature_cols].median())
    y = (df["status"] == "PLAYS").astype(int)
    n_pos, n_neg = y.sum(), len(y) - y.sum()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=42, max_depth=6, min_samples_leaf=5, n_jobs=-1)
    rf_proba = cross_val_predict(rf, X, y, cv=cv, method="predict_proba")[:, 1]

    # XGBoost
    xgb_m = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
        scale_pos_weight=n_neg / max(n_pos, 1), random_state=42,
        eval_metric="logloss", verbosity=0, colsample_bytree=0.5, subsample=0.8)
    xgb_proba = cross_val_predict(xgb_m, X, y, cv=cv, method="predict_proba")[:, 1]

    ensemble_proba = 0.5 * rf_proba + 0.5 * xgb_proba

    # Find optimal threshold (maximize F1)
    thresholds = np.arange(0.1, 0.5, 0.01)
    f1s = [f1_score(y, (ensemble_proba > t).astype(int)) for t in thresholds]
    best_t = thresholds[np.argmax(f1s)]
    y_pred = (ensemble_proba > best_t).astype(int)

    auc = roc_auc_score(y, ensemble_proba)
    f1 = f1_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)

    # Save metrics
    metrics = {
        "auc": auc, "f1": f1, "precision": prec, "recall": rec,
        "threshold": best_t, "n_features": len(feature_cols),
        "n_samples": len(y), "n_plays": int(y.sum()),
        "n_rest": int(len(y) - y.sum()),
    }
    pd.DataFrame([metrics]).to_csv(OUTPUT / "model_metrics.csv", index=False)

    # Confusion matrix plot
    cm = confusion_matrix(y, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Predicted: REST", "Predicted: PLAYS"],
                yticklabels=["Actual: REST", "Actual: PLAYS"])
    ax.set_title(f"Confusion Matrix (threshold={best_t:.2f})")

    # ROC with threshold marker
    ax = axes[1]
    fpr, tpr, threshs = roc_curve(y, ensemble_proba)
    ax.plot(fpr, tpr, linewidth=2, label=f"Ensemble (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC=0.500)")
    # Mark operating point
    idx = np.argmin(np.abs(threshs - best_t))
    ax.plot(fpr[idx], tpr[idx], "ro", markersize=10,
            label=f"Operating point (F1={f1:.2f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Ensemble Model")
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT / "model_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Classification report
    report = classification_report(y, y_pred, target_names=["REST", "PLAYS"])
    with open(OUTPUT / "classification_report.txt", "w") as f:
        f.write(f"Ensemble Model — PLAYS vs REST\n")
        f.write(f"Features: {len(feature_cols)}, Threshold: {best_t:.2f}\n")
        f.write(f"AUC: {auc:.3f}\n\n")
        f.write(report)

    print(f"  AUC={auc:.3f}  F1={f1:.3f}  Precision={prec:.3f}  Recall={rec:.3f}")
    print(f"  Threshold: {best_t:.2f}")
    print(f"  Saved: model_evaluation.png, model_metrics.csv, classification_report.txt")


def generate_multiclass_analysis(df, selected):
    """Show how the model distinguishes all 4 status groups, not just binary."""
    print("Generating multi-class analysis...")

    optimal_n = int((OUTPUT / "optimal_n_features.txt").read_text().strip())
    top = selected.head(optimal_n)
    feature_cols = [f"mean_{n}" for n in top["feature_name"] if f"mean_{n}" in df.columns]

    X = df[feature_cols].copy().fillna(df[feature_cols].median())
    y_binary = (df["status"] == "PLAYS").astype(int)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    n_pos, n_neg = y_binary.sum(), len(y_binary) - y_binary.sum()

    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=42, max_depth=6, min_samples_leaf=5, n_jobs=-1)
    xgb_m = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
        scale_pos_weight=n_neg / max(n_pos, 1), random_state=42,
        eval_metric="logloss", verbosity=0, colsample_bytree=0.5, subsample=0.8)

    rf_proba = cross_val_predict(rf, X, y_binary, cv=cv, method="predict_proba")[:, 1]
    xgb_proba = cross_val_predict(xgb_m, X, y_binary, cv=cv, method="predict_proba")[:, 1]
    ensemble_proba = 0.5 * rf_proba + 0.5 * xgb_proba

    df_plot = df[["status"]].copy()
    df_plot["probability"] = ensemble_proba

    fig, ax = plt.subplots(figsize=(10, 6))
    for status in STATUS_ORDER:
        subset = df_plot[df_plot["status"] == status]["probability"]
        ax.hist(subset, bins=30, alpha=0.5, label=f"{status} (n={len(subset)})",
                color=COLORS[status], density=True)
    ax.set_xlabel("Model Probability (PLAYS)")
    ax.set_ylabel("Density")
    ax.set_title("Model Output Distribution by Status Group\n"
                 "The model separates PLAYS from REST, with BENCH/DROPPED in between")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT / "multiclass_probability.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: multiclass_probability.png")


def main():
    print("=" * 60)
    print("  Generating demo assets")
    print("=" * 60)

    df, selected = load_data()
    generate_exploration_plots(df, selected)
    generate_correlation_heatmap(df, selected)
    generate_full_model_metrics(df, selected)
    generate_multiclass_analysis(df, selected)

    print("\nAll demo assets generated.")


if __name__ == "__main__":
    main()
