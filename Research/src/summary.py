"""Print final summary and conclusions."""


def print_summary(df_model, feature_cols_clean, stat_df, consensus_df, binary_results):
    """Print a human-readable summary of all findings."""
    print("\n" + "=" * 70)
    print("10. SUMMARY & CONCLUSIONS")
    print("=" * 70)

    print(f"""
DATASET:
  - {len(df_model)} goalkeepers with usable data
  - {len(feature_cols_clean)} features used
  - Status: {dict(df_model['status'].value_counts())}

STATISTICAL FINDINGS:
  - {stat_df['significant_005'].sum()} features significantly different between PLAYS and REST (p<0.05)
  - {stat_df['significant_001'].sum()} features highly significant (p<0.01)

TOP DISCRIMINATING FEATURES (by statistical test):
""")
    for _, row in stat_df.head(10).iterrows():
        direction = "higher" if row["diff"] > 0 else "lower"
        print(f"  {row['feature'][:45]:45s} p={row['p_value']:.4f}  PLAYS {direction} (d={row['effect_size_d']:.3f})")

    print(f"""
CONSENSUS FEATURES (in top 10 across multiple importance methods):
""")
    for _, row in consensus_df[consensus_df["methods_in_top10"] >= 2].iterrows():
        print(f"  {row['feature'][:50]:50s}  ({row['methods_in_top10']}/4 methods)")

    print(f"""
MODEL PERFORMANCE (5-fold CV, binary PLAYS vs REST):
  Logistic Regression:  F1={binary_results['Logistic Regression']['f1']:.3f}, AUC={binary_results['Logistic Regression']['auc']:.3f}
  Random Forest:        F1={binary_results['Random Forest']['f1']:.3f}, AUC={binary_results['Random Forest']['auc']:.3f}
  XGBoost:              F1={binary_results['XGBoost']['f1']:.3f}, AUC={binary_results['XGBoost']['auc']:.3f}
""")

    print("Done!")
