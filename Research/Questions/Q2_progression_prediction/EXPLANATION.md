# Q2 — Can We Predict Progression?

## The Question
Can we predict which goalkeepers will progress to a higher league based on their lower-league performances?

## Answer
**Yes, with caveats.** Our best model (XGBoost) achieves AUC-ROC of ~0.77 for PLAYS vs REST, significantly outperforming random baseline (14% PLAYS rate). The model correctly ranks a random PLAYS keeper above a random REST keeper 77% of the time. This is a useful screening signal, not a crystal ball.

For the UP vs NOT UP target (PLAYS+BENCH vs STAYED+DROPPED), XGBoost achieves AUC ~0.80 — stronger because it has more positive examples (130 vs 99).

## Models Used

| Model | Why Chosen | Strengths |
|-------|-----------|-----------|
| **Logistic Regression (L2)** | Simple, fully interpretable | Coefficients directly show direction and magnitude of each feature's effect |
| **Random Forest** | Non-linear, handles interactions | Robust to outliers, provides feature importance |
| **XGBoost** | State-of-the-art for tabular data | Best accuracy, supports SHAP for interpretability |

### Hyperparameters
- **Logistic Regression**: C=0.1 (moderate regularization), balanced class weights
- **Random Forest**: 300 trees, max_depth=6, min_samples_leaf=5, balanced weights
- **XGBoost**: 300 trees, max_depth=4, learning_rate=0.05, scale_pos_weight adjusts for class imbalance

### Why These Choices
- Balanced class weights / scale_pos_weight compensate for PLAYS being only 14% of data
- Moderate tree depth prevents overfitting on small dataset
- L2 regularization (not L1) for LR because we want all features to contribute rather than sparse selection

## Evaluation
- **Stratified 5-fold cross-validation**: Ensures each fold preserves the class ratio. All reported metrics are out-of-fold predictions.
- **Metrics**: Accuracy, precision, recall, F1, AUC-ROC
- **Random baseline**: 14% accuracy for always predicting PLAYS, 86% for always predicting REST

## SHAP Analysis
SHAP (SHapley Additive exPlanations) provides per-prediction feature attribution:

- **Beeswarm plot**: Shows how each feature value pushes predictions. Red dots = high feature value, blue = low. Features at top have largest impact.
- **Bar plot**: Mean absolute SHAP value per feature — global importance ranking.
- **Dependence plots**: Shows non-linear relationships. E.g., does passing accuracy have a threshold effect?

### Why SHAP over other methods
- SHAP is theoretically grounded (Shapley values from game theory)
- Provides both global importance AND local explanations
- TreeExplainer is exact for tree-based models (no approximation)
- Shows direction of effect (unlike Gini importance)

## Key Findings
- **Context features** (league strength, age, matches played) are strong predictors — a 21-year-old in the Regionalliga is more likely to progress than a 32-year-old
- **Performance features** provide signal beyond context: Defensive IMPECT, passing accuracy, and foot usage ratio consistently appear in top features
- The model captures something real, but performance alone (without context) is weaker than the combined model

## Output Files
- `confusion_matrices_plays_vs_rest.png` — Side-by-side confusion matrices
- `roc_curves_plays_vs_rest.png` — ROC curves with AUC for all models
- `confusion_matrices_up_vs_not_up.png` — UP vs NOT UP results
- `roc_curves_up_vs_not_up.png`
- `shap_summary_beeswarm.png` — SHAP beeswarm plot (most informative)
- `shap_importance_bar.png` — SHAP feature importance
- `shap_dependence_top4.png` — Dependence plots for top 4 features
- `shap_feature_importance.csv` — Mean absolute SHAP values per feature
- `model_results_summary.csv` — All model metrics in one table
- `multiclass_cm_*.png` — Multi-class confusion matrices
