"""Feature importance analysis — LR, RF, XGBoost, permutation, consensus."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import xgboost as xgb

from .config import OUTPUT


def run_feature_importance(df_model, feature_cols_clean):
    """Compute feature importance with 4 methods and build consensus ranking.

    Returns
    -------
    consensus_df : DataFrame
    """
    print("\n" + "=" * 70)
    print("7. FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)

    X = df_model[feature_cols_clean].copy().fillna(df_model[feature_cols_clean].median())
    y_binary = (df_model["status"] == "PLAYS").astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Logistic Regression coefficients
    print("\n--- Logistic Regression Coefficients ---")
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42, C=0.1)
    lr.fit(X_scaled, y_binary)
    lr_importance = pd.DataFrame({
        "feature": feature_cols_clean,
        "coefficient": lr.coef_[0],
        "abs_coefficient": np.abs(lr.coef_[0]),
    }).sort_values("abs_coefficient", ascending=False)
    print(lr_importance.head(15).to_string(index=False))

    # Random Forest
    print("\n--- Random Forest Feature Importance ---")
    rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42,
        max_depth=6, min_samples_leaf=5
    )
    rf.fit(X, y_binary)
    rf_importance = pd.DataFrame({
        "feature": feature_cols_clean,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)
    print(rf_importance.head(15).to_string(index=False))

    # XGBoost
    print("\n--- XGBoost Feature Importance ---")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        scale_pos_weight=len(y_binary[y_binary == 0]) / max(len(y_binary[y_binary == 1]), 1),
        random_state=42, eval_metric="logloss", verbosity=0,
    )
    xgb_model.fit(X, y_binary)
    xgb_importance = pd.DataFrame({
        "feature": feature_cols_clean,
        "importance": xgb_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print(xgb_importance.head(15).to_string(index=False))

    # Permutation importance
    print("\n--- Permutation Importance (Random Forest) ---")
    perm_imp = permutation_importance(rf, X, y_binary, n_repeats=20, random_state=42, n_jobs=-1)
    perm_df = pd.DataFrame({
        "feature": feature_cols_clean,
        "importance_mean": perm_imp.importances_mean,
        "importance_std": perm_imp.importances_std,
    }).sort_values("importance_mean", ascending=False)
    print(perm_df.head(15).to_string(index=False))

    # Consensus ranking
    print("\n--- Consensus Top Features (appear in top 10 across methods) ---")
    top_n = 10
    top_lr = set(lr_importance.head(top_n)["feature"])
    top_rf = set(rf_importance.head(top_n)["feature"])
    top_xgb = set(xgb_importance.head(top_n)["feature"])
    top_perm = set(perm_df.head(top_n)["feature"])

    all_top = top_lr | top_rf | top_xgb | top_perm
    consensus = []
    for feat in all_top:
        count = sum([feat in top_lr, feat in top_rf, feat in top_xgb, feat in top_perm])
        consensus.append({"feature": feat, "methods_in_top10": count})

    consensus_df = pd.DataFrame(consensus).sort_values("methods_in_top10", ascending=False)
    print(consensus_df.to_string(index=False))

    # Feature importance plots
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))

    ax = axes[0, 0]
    top15 = lr_importance.head(15)
    colors = ["green" if c > 0 else "red" for c in top15["coefficient"]]
    short_names = [f.replace("mean_", "").replace("GK_", "")[:30] for f in top15["feature"]]
    ax.barh(range(len(top15)), top15["coefficient"], color=colors)
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_title("Logistic Regression Coefficients")
    ax.invert_yaxis()

    ax = axes[0, 1]
    top15 = rf_importance.head(15)
    short_names = [f.replace("mean_", "").replace("GK_", "")[:30] for f in top15["feature"]]
    ax.barh(range(len(top15)), top15["importance"], color="steelblue")
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_title("Random Forest Feature Importance")
    ax.invert_yaxis()

    ax = axes[1, 0]
    top15 = xgb_importance.head(15)
    short_names = [f.replace("mean_", "").replace("GK_", "")[:30] for f in top15["feature"]]
    ax.barh(range(len(top15)), top15["importance"], color="darkorange")
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_title("XGBoost Feature Importance")
    ax.invert_yaxis()

    ax = axes[1, 1]
    top15 = perm_df.head(15)
    short_names = [f.replace("mean_", "").replace("GK_", "")[:30] for f in top15["feature"]]
    ax.barh(range(len(top15)), top15["importance_mean"], color="purple",
            xerr=top15["importance_std"])
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_title("Permutation Importance (RF)")
    ax.invert_yaxis()

    fig.suptitle("Feature Importance: PLAYS vs REST (4 Methods)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT / "09_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 09_feature_importance.png")

    # Save CSVs
    lr_importance.to_csv(OUTPUT / "importance_logistic_regression.csv", index=False)
    rf_importance.to_csv(OUTPUT / "importance_random_forest.csv", index=False)
    xgb_importance.to_csv(OUTPUT / "importance_xgboost.csv", index=False)
    perm_df.to_csv(OUTPUT / "importance_permutation.csv", index=False)
    consensus_df.to_csv(OUTPUT / "importance_consensus.csv", index=False)

    return consensus_df
