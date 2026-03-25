"""
GK Talent Identification — OH Leuven
=====================================
Analyses which measurable goalkeeper performances in lower leagues
predict career progression to a higher level.

Run: python3 main.py
"""

from src.config import OUTPUT, load_definitions
from src.data_loading import load_data
from src.feature_selection import select_features
from src.eda import run_eda
from src.statistical_tests import run_statistical_tests
from src.modeling import run_models
from src.feature_importance import run_feature_importance
from src.reliability import run_reliability_analysis
from src.scouting_score import compute_scouting_scores
from src.summary import print_summary


def main():
    # 1–2. Load definitions and data
    score_defs, score_labels, kpi_defs = load_definitions()
    dataset, df = load_data(score_defs)

    # 3. Feature selection
    df_model, feature_cols_clean, gk_score_cols, general_score_cols = select_features(df, score_defs)

    # 4. Exploratory data analysis
    run_eda(df_model, feature_cols_clean, gk_score_cols, general_score_cols)

    # 5. Statistical testing
    stat_df, kw_df = run_statistical_tests(df_model, feature_cols_clean)

    # 6. Classification models
    binary_results, up_results = run_models(df_model, feature_cols_clean)

    # 7. Feature importance
    consensus_df = run_feature_importance(df_model, feature_cols_clean)

    # 8–9. Reliability / signal vs noise
    rel_df = run_reliability_analysis(df_model, feature_cols_clean, stat_df)

    # 10. GK Scouting Score (1–100)
    scores_df = compute_scouting_scores(df_model, feature_cols_clean, stat_df, rel_df)

    # 11. Summary
    print_summary(df_model, feature_cols_clean, stat_df, consensus_df, binary_results)

    print(f"\nAll outputs saved to: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
