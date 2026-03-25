"""Classification models — binary and multi-class."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb

from .config import OUTPUT


def run_models(df_model, feature_cols_clean):
    """Train and evaluate all classification models.

    Returns
    -------
    binary_results : dict
    up_results : dict
    """
    print("\n" + "=" * 70)
    print("6. CLASSIFICATION MODELS")
    print("=" * 70)

    X = df_model[feature_cols_clean].copy().fillna(df_model[feature_cols_clean].median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── 6a. Binary: PLAYS vs REST ─────────────────────────────────────
    print("\n--- 6a. Binary Classification: PLAYS vs REST ---")
    y_binary = (df_model["status"] == "PLAYS").astype(int)

    models_binary = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42, C=0.1
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42,
            max_depth=6, min_samples_leaf=5
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            scale_pos_weight=len(y_binary[y_binary == 0]) / max(len(y_binary[y_binary == 1]), 1),
            random_state=42, eval_metric="logloss", verbosity=0,
        ),
    }

    binary_results = {}
    for name, model in models_binary.items():
        print(f"\n  {name}:")
        use_scaled = name == "Logistic Regression"
        X_in = X_scaled if use_scaled else X

        y_pred = cross_val_predict(model, X_in, y_binary, cv=cv)
        y_proba = cross_val_predict(model, X_in, y_binary, cv=cv, method="predict_proba")[:, 1]

        f1 = f1_score(y_binary, y_pred, average="binary")
        try:
            auc = roc_auc_score(y_binary, y_proba)
        except ValueError:
            auc = 0.0

        report = classification_report(y_binary, y_pred, target_names=["REST", "PLAYS"])
        print(f"    F1 (PLAYS): {f1:.3f}")
        print(f"    AUC-ROC:    {auc:.3f}")
        print(report)
        binary_results[name] = {"f1": f1, "auc": auc, "y_pred": y_pred, "y_proba": y_proba}

    # Confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, res) in zip(axes, binary_results.items()):
        cm = confusion_matrix(y_binary, res["y_pred"])
        ConfusionMatrixDisplay(cm, display_labels=["REST", "PLAYS"]).plot(ax=ax, cmap="Blues")
        ax.set_title(f"{name}\nF1={res['f1']:.3f}, AUC={res['auc']:.3f}")
    plt.suptitle("Binary Classification: PLAYS vs REST", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT / "07_binary_confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 07_binary_confusion_matrices.png")

    # ── 6b. Binary: UP vs NOT UP ──────────────────────────────────────
    print("\n--- 6b. Binary Classification: UP vs NOT UP ---")
    y_up = df_model["direction"].map({"UP": 1, "DOWN": 0, "NONE": 0}).astype(int)

    models_up = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42, C=0.1
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42,
            max_depth=6, min_samples_leaf=5
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            scale_pos_weight=len(y_up[y_up == 0]) / max(len(y_up[y_up == 1]), 1),
            random_state=42, eval_metric="logloss", verbosity=0,
        ),
    }

    up_results = {}
    for name, model in models_up.items():
        print(f"\n  {name}:")
        use_scaled = name == "Logistic Regression"
        X_in = X_scaled if use_scaled else X

        y_pred = cross_val_predict(model, X_in, y_up, cv=cv)
        y_proba = cross_val_predict(model, X_in, y_up, cv=cv, method="predict_proba")[:, 1]

        f1 = f1_score(y_up, y_pred, average="binary")
        try:
            auc = roc_auc_score(y_up, y_proba)
        except ValueError:
            auc = 0.0

        report = classification_report(y_up, y_pred, target_names=["NOT UP", "UP"])
        print(f"    F1 (UP):  {f1:.3f}")
        print(f"    AUC-ROC:  {auc:.3f}")
        print(report)
        up_results[name] = {"f1": f1, "auc": auc, "y_pred": y_pred, "y_proba": y_proba}

    # ── 6c. Multi-class ───────────────────────────────────────────────
    print("\n--- 6c. Multi-class Classification ---")
    le = LabelEncoder()
    y_multi = le.fit_transform(df_model["status"])
    class_names = le.classes_

    models_multi = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42,
            max_depth=6, min_samples_leaf=5
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            random_state=42, eval_metric="mlogloss", verbosity=0,
        ),
    }

    for name, model in models_multi.items():
        print(f"\n  {name}:")
        y_pred = cross_val_predict(model, X, y_multi, cv=cv)
        f1_macro = f1_score(y_multi, y_pred, average="macro")
        f1_weighted = f1_score(y_multi, y_pred, average="weighted")
        report = classification_report(y_multi, y_pred, target_names=class_names)
        print(f"    F1 (macro):    {f1_macro:.3f}")
        print(f"    F1 (weighted): {f1_weighted:.3f}")
        print(report)

        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_multi, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap="Blues")
        ax.set_title(f"Multi-class: {name}\nF1(macro)={f1_macro:.3f}")
        plt.tight_layout()
        plt.savefig(OUTPUT / f"08_multiclass_cm_{name.lower().replace(' ', '_')}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: 08_multiclass_cm_{name.lower().replace(' ', '_')}.png")

    return binary_results, up_results
