#!/usr/bin/env python3
"""Q2 — Can we predict progression?

Builds classification models to predict whether a goalkeeper will progress,
with a focus on interpretability (SHAP plots).

Models:
- Logistic Regression (L2, balanced weights)
- Random Forest (balanced weights)
- XGBoost (scale_pos_weight)

Targets:
- Primary: PLAYS vs REST
- Secondary: UP (PLAYS+BENCH) vs NOT UP (STAYED+DROPPED)

Run: python -m Q2_progression_prediction.model
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    f1_score, ConfusionMatrixDisplay, precision_score, recall_score,
    roc_curve, accuracy_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("WARNING: shap not installed. SHAP plots will be skipped.")

from shared.data_utils import (
    load_definitions, load_and_aggregate_data, select_features,
    get_cache_path, STATUS_ORDER
)

OUTPUT = Path(__file__).resolve().parent / "results"
OUTPUT.mkdir(exist_ok=True)


def train_binary_models(X, X_scaled, y, label_names, target_name, cv):
    """Train 3 binary classifiers with CV and return results."""
    print(f"\n--- Binary Classification: {target_name} ---")
    print(f"  Positive class: {label_names[1]} (n={y.sum()})")
    print(f"  Negative class: {label_names[0]} (n=(y==0).sum())")
    print(f"  Baseline (random): {y.mean():.1%}")

    n_pos = y.sum()
    n_neg = len(y) - n_pos

    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=42, C=0.1),
            True  # use scaled
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   random_state=42, max_depth=6, min_samples_leaf=5),
            False
        ),
        "XGBoost": (
            xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                               scale_pos_weight=n_neg / max(n_pos, 1),
                               random_state=42, eval_metric="logloss", verbosity=0),
            False
        ),
    }

    results = {}
    for name, (model, use_scaled) in models.items():
        X_in = X_scaled if use_scaled else X
        y_pred = cross_val_predict(model, X_in, y, cv=cv)
        y_proba = cross_val_predict(model, X_in, y, cv=cv, method="predict_proba")[:, 1]

        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred)
        try:
            auc = roc_auc_score(y, y_proba)
        except ValueError:
            auc = 0.0

        print(f"\n  {name}:")
        print(f"    Accuracy:  {acc:.3f}  (baseline: {1 - y.mean():.3f})")
        print(f"    Precision: {prec:.3f}")
        print(f"    Recall:    {rec:.3f}")
        print(f"    F1:        {f1:.3f}")
        print(f"    AUC-ROC:   {auc:.3f}")

        results[name] = {
            "accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "auc": auc, "y_pred": y_pred, "y_proba": y_proba,
        }

    return results


def plot_confusion_matrices(y, results, label_names, target_name):
    """Plot confusion matrices for all models side by side."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y, res["y_pred"])
        ConfusionMatrixDisplay(cm, display_labels=label_names).plot(ax=ax, cmap="Blues")
        ax.set_title(f"{name}\nF1={res['f1']:.3f}, AUC={res['auc']:.3f}")

    plt.suptitle(f"Confusion Matrices: {target_name}", fontsize=14)
    plt.tight_layout()
    fname = f"confusion_matrices_{target_name.lower().replace(' ', '_')}.png"
    plt.savefig(OUTPUT / fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def plot_roc_curves(y, results, target_name):
    """Plot ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y, res["y_proba"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})")

    ax.plot([0, 1], [0, 1], "k--", label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves: {target_name}")
    ax.legend()
    plt.tight_layout()
    fname = f"roc_curves_{target_name.lower().replace(' ', '_')}.png"
    plt.savefig(OUTPUT / fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def run_shap_analysis(X, y, feature_cols):
    """Train XGBoost and generate SHAP plots."""
    if not HAS_SHAP:
        print("  SHAP not available — skipping")
        return

    print("\n--- SHAP Analysis (XGBoost) ---")

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        scale_pos_weight=n_neg / max(n_pos, 1),
        random_state=42, eval_metric="logloss", verbosity=0,
    )
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Clean feature names for plots
    clean_names = [c.replace("mean_", "").replace("GK_", "")[:35] for c in feature_cols]
    X_display = X.copy()
    X_display.columns = clean_names

    # SHAP summary (beeswarm)
    fig, ax = plt.subplots(figsize=(12, 10))
    shap.summary_plot(shap_values, X_display, show=False, max_display=20)
    plt.title("SHAP Values: Feature Impact on PLAYS Prediction", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT / "shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: shap_summary_beeswarm.png")

    # SHAP bar plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_display, plot_type="bar", show=False, max_display=20)
    plt.title("SHAP Feature Importance (mean |SHAP value|)", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT / "shap_importance_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: shap_importance_bar.png")

    # SHAP dependence plots for top 4 features
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:4]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, idx in enumerate(top_idx):
        ax = axes[i // 2][i % 2]
        shap.dependence_plot(idx, shap_values, X_display, ax=ax, show=False)
        ax.set_title(f"SHAP Dependence: {clean_names[idx]}", fontsize=10)

    plt.suptitle("SHAP Dependence Plots — Top 4 Features", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT / "shap_dependence_top4.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: shap_dependence_top4.png")

    # Save mean absolute SHAP per feature
    mean_shap = pd.DataFrame({
        "feature": feature_cols,
        "feature_name": clean_names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    mean_shap.to_csv(OUTPUT / "shap_feature_importance.csv", index=False)
    print("  Saved: shap_feature_importance.csv")


def run_multiclass(X, y_multi, class_names, cv):
    """Multi-class classification (all 4 categories)."""
    print("\n--- Multi-class Classification ---")

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42,
            max_depth=6, min_samples_leaf=5
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            random_state=42, eval_metric="mlogloss", verbosity=0,
        ),
    }

    for name, model in models.items():
        print(f"\n  {name}:")
        y_pred = cross_val_predict(model, X, y_multi, cv=cv)
        f1_macro = f1_score(y_multi, y_pred, average="macro")
        f1_weighted = f1_score(y_multi, y_pred, average="weighted")
        print(f"    F1 (macro):    {f1_macro:.3f}")
        print(f"    F1 (weighted): {f1_weighted:.3f}")
        print(classification_report(y_multi, y_pred, target_names=class_names))

        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_multi, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap="Blues")
        ax.set_title(f"Multi-class: {name}\nF1(macro)={f1_macro:.3f}")
        plt.tight_layout()
        fname = f"multiclass_cm_{name.lower().replace(' ', '_')}.png"
        plt.savefig(OUTPUT / fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {fname}")


def save_results_summary(binary_results, up_results):
    """Save a CSV summary of all model results."""
    rows = []
    for target, results in [("PLAYS vs REST", binary_results), ("UP vs NOT UP", up_results)]:
        for name, res in results.items():
            rows.append({
                "target": target,
                "model": name,
                "accuracy": res["accuracy"],
                "precision": res["precision"],
                "recall": res["recall"],
                "f1": res["f1"],
                "auc_roc": res["auc"],
            })
    pd.DataFrame(rows).to_csv(OUTPUT / "model_results_summary.csv", index=False)
    print("  Saved: model_results_summary.csv")


def main():
    print("=" * 70)
    print("Q2 — CAN WE PREDICT PROGRESSION?")
    print("=" * 70)

    # Load data
    score_defs, _, _ = load_definitions()
    cache = get_cache_path()
    dataset, df = load_and_aggregate_data(score_defs, cache_path=cache)
    df_model, feature_cols, gk_cols, general_cols = select_features(df, score_defs)

    X = df_model[feature_cols].copy().fillna(df_model[feature_cols].median())
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols, index=X.index)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # --- Binary: PLAYS vs REST ---
    y_binary = (df_model["status"] == "PLAYS").astype(int)
    binary_results = train_binary_models(X, X_scaled, y_binary,
                                          ["REST", "PLAYS"], "PLAYS vs REST", cv)
    plot_confusion_matrices(y_binary, binary_results, ["REST", "PLAYS"], "PLAYS vs REST")
    plot_roc_curves(y_binary, binary_results, "PLAYS vs REST")

    # --- Binary: UP vs NOT UP ---
    y_up = df_model["direction"].map({"UP": 1, "DOWN": 0, "NONE": 0}).astype(int)
    up_results = train_binary_models(X, X_scaled, y_up,
                                      ["NOT UP", "UP"], "UP vs NOT UP", cv)
    plot_confusion_matrices(y_up, up_results, ["NOT UP", "UP"], "UP vs NOT UP")
    plot_roc_curves(y_up, up_results, "UP vs NOT UP")

    # --- SHAP analysis ---
    run_shap_analysis(X, y_binary, feature_cols)

    # --- Multi-class ---
    le = LabelEncoder()
    y_multi = le.fit_transform(df_model["status"])
    run_multiclass(X, y_multi, le.classes_, cv)

    # --- Save summary ---
    save_results_summary(binary_results, up_results)

    print(f"\n{'=' * 70}")
    print("Q2 SUMMARY")
    print("=" * 70)
    print(f"\n  PLAYS vs REST (baseline: {y_binary.mean():.1%} positive rate):")
    for name, res in binary_results.items():
        print(f"    {name:25s}  F1={res['f1']:.3f}  AUC={res['auc']:.3f}")
    print(f"\n  UP vs NOT UP (baseline: {y_up.mean():.1%} positive rate):")
    for name, res in up_results.items():
        print(f"    {name:25s}  F1={res['f1']:.3f}  AUC={res['auc']:.3f}")
    print(f"\n  All outputs saved to: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
